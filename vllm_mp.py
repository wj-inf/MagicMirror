#!/usr/bin/env python3
"""
Multi-GPU Image Quality Assessment using vLLM

This script processes JSONL files containing subject detection results and performs
quality assessment on generated images using a fine-tuned vision-language model.

"""

import os
import sys
import math
import json
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from typing import List, Dict, Any, Tuple

from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from base_info import get_prompt

DEFAULT_MODEL_PATH = "wj-inf/MagicData340k"

def ensure_image_1024(image_path: str) -> str:
    """
    Ensure image resolution is suitable for processing.
    If not 1024x1024, resize the shorter side to 1024 while maintaining aspect ratio.
    Save to a directory with '_1024' suffix and return the new path.
    If already 1024x1024, return original path.
    
    Args:
        image_path (str): Path to the original image
        
    Returns:
        str: Path to the processed image
    """
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} does not exist.")
        return image_path
    
    try:
        img = Image.open(image_path)
        w, h = img.size
        
        # If already 1024x1024, return original path
        if w == 1024 and h == 1024:
            return image_path

        # Calculate new dimensions
        if w < h:
            new_w = 1024
            new_h = int(h * 1024 / w)
        else:
            new_h = 1024
            new_w = int(w * 1024 / h)
        
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)

        # Determine new path
        dirname, basename = os.path.split(image_path)
        parent, folder = os.path.split(dirname)
        new_folder = folder + "_1024"
        new_dir = os.path.join(parent, new_folder)
        os.makedirs(new_dir, exist_ok=True)
        new_path = os.path.join(new_dir, basename)

        # Save if file doesn't exist
        if not os.path.exists(new_path):
            img_resized.save(new_path)
        
        return new_path
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return image_path

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process images with vLLM model for quality assessment using multi-GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python vllm_mp.py --input_file output/flux/subject_detection_flux.jsonl
  
  # Process multiple files
  python vllm_mp.py --input_files output/flux/subject_detection_flux.jsonl output/sd3/subject_detection_sd3.jsonl
  
  # Use custom model and settings
  python vllm_mp.py --input_file data.jsonl --model_path /path/to/model --batch_size 16 --world_size 4
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_file', type=str,
                           help='Single JSONL file containing subject detection results')
    input_group.add_argument('--input_files', nargs='+', 
                           help='Multiple JSONL files containing subject detection results')
    
    # Model options
    parser.add_argument('--model_name', type=str, default='wj-inf/MagicData340k')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--output_dir', type=str, default='./output')
    
    # Performance options
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count())
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--max_tokens', type=int, default=2048)
    
    # Debug options
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Convert single file to list for uniform processing
    if args.input_file:
        args.input_files = [args.input_file]
    
    # Validate input files
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            parser.error(f"Input file does not exist: {input_file}")
    
    # Validate world_size
    available_gpus = torch.cuda.device_count()
    if args.world_size > available_gpus:
        print(f"Warning: Requested world_size ({args.world_size}) > available GPUs ({available_gpus})")
        args.world_size = available_gpus
    
    return args

def setup_distributed(rank: int, world_size: int, port: str = '12356') -> None:
    """Initialize distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed() -> None:
    """Clean up distributed environment."""
    dist.destroy_process_group()

def parse_jsonl_data(input_path: str, rank: int, world_size: int) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse JSONL file and distribute data across processes.
    
    Args:
        input_path (str): Path to input JSONL file
        rank (int): Current process rank
        world_size (int): Total number of processes
        
    Returns:
        Tuple[List[Dict], List[Dict]]: (data_for_current_rank, all_data)
    """
    assert os.path.isfile(input_path), f"File not found: {input_path}"
    
    all_data = []
    
    # Read all data
    with open(input_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            try:
                item = json.loads(line.strip())
                all_data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num} in {input_path}: {e}")
                continue
    
    # Calculate data partition for current rank
    total_items = len(all_data)
    items_per_rank = math.ceil(total_items / world_size)
    start_idx = rank * items_per_rank
    end_idx = min((rank + 1) * items_per_rank, total_items)
    
    print(f'Rank {rank}: Processing items {start_idx}-{end_idx-1} from {input_path} (total: {total_items})')
    
    return all_data[start_idx:end_idx], all_data

def process_batch_data(batch_data: List[Dict], processor, llm, sampling_params, verbose: bool = False) -> List[Dict]:
    """
    Process a batch of data for quality assessment.
    
    Args:
        batch_data (List[Dict]): Batch of data items
        processor: Model processor
        llm: Language model
        sampling_params: Sampling parameters
        verbose (bool): Enable verbose output
        
    Returns:
        List[Dict]: Processed batch data with quality assessment results
    """
    batch_inputs = []
    
    # Prepare inputs for each data item
    for data in batch_data:
        prompt_t2i = data.get("prompt_t2i", "")
        
        # Apply prompt transformation using get_prompt function
        try:
            processed_prompt = get_prompt(prompt_t2i)
        except Exception as e:
            if verbose:
                print(f"Warning: Error processing prompt '{prompt_t2i[:50]}...': {e}")
            processed_prompt = prompt_t2i
        
        # Get image path
        gen_images = data.get("gen_images", [])
        if not gen_images or gen_images == []:
            image_path = None
        else:
            image_path = gen_images[0]
            image_path = ensure_image_1024(image_path)
        
        # Prepare content based on image availability
        if image_path is None or not os.path.exists(image_path):
            content = [{"type": "text", "text": processed_prompt}]
            if verbose:
                print(f"Warning: No valid image found for prompt: {prompt_t2i[:50]}...")
        else:
            content = [
                {
                    "type": "image",
                    "image": image_path,
                    "min_pixels": 224 * 224,
                    "max_pixels": 1280 * 28 * 28,
                },
                {"type": "text", "text": processed_prompt},
            ]
        
        # Build messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]
        
        # Process messages
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }
        
        batch_inputs.append(llm_inputs)
    
    # Generate responses
    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
    
    # Process outputs
    for j, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        batch_data[j]["response"] = generated_text
    
    return batch_data

def process_with_vllm(rank: int, world_size: int, args: argparse.Namespace) -> None:
    """
    Main processing function for each process.
    
    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        args (argparse.Namespace): Command line arguments
    """
    # Setup distributed environment if using multiple processes
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # Set device for current process
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Load model
    print(f"Rank {rank}: Loading model from {args.model_path}")
    
    try:
        # Create LLM instance
        llm = LLM(
            model=args.model_path,
            limit_mm_per_prompt={"image": 10, "video": 10},
            tensor_parallel_size=1,  # Each process uses 1 GPU
            device=f"cuda:{rank}"
        )
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(args.model_path)
        
        print(f"Rank {rank}: Model loaded successfully")
        
    except Exception as e:
        print(f"Rank {rank}: Error loading model: {e}")
        if world_size > 1:
            cleanup_distributed()
        return
    
    # Process each input file
    for input_file in args.input_files:
        print(f"Rank {rank}: Processing file {input_file}")
        
        # Extract model name from input path
        # e.g., "output/flux/subject_detection_flux.jsonl" -> "flux"
        path_parts = input_file.split('/')
        if len(path_parts) >= 2:
            model_name = path_parts[-2]  # Second to last part
        else:
            model_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Create output directory
        output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Set output file name
        output_file = os.path.join(output_dir, f"merged_result_{model_name}_vllm.jsonl")
        
        try:
            # Parse input data
            data_for_this_rank, all_data = parse_jsonl_data(input_file, rank, world_size)
            
            if not data_for_this_rank:
                print(f"Rank {rank}: No data to process for {input_file}")
                continue
            
            # Process in batches
            total_batches = math.ceil(len(data_for_this_rank) / args.batch_size)
            
            for i in range(0, len(data_for_this_rank), args.batch_size):
                batch_data = data_for_this_rank[i:i+args.batch_size]
                batch_num = i // args.batch_size + 1
                
                print(f"Rank {rank}: Processing {model_name} batch {batch_num}/{total_batches}")
                
                # Process batch
                processed_batch = process_batch_data(
                    batch_data, processor, llm, sampling_params, args.verbose
                )
                
                # Write batch results to temporary file
                temp_output = f"{os.path.splitext(output_file)[0]}-rank{rank}.jsonl"
                with open(temp_output, 'a', encoding='utf-8') as file:
                    for item in processed_batch:
                        file.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                # Clear memory
                del processed_batch
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Rank {rank}: Error processing {input_file}: {e}")
            continue
        
        # Wait for all processes to complete current file
        if world_size > 1:
            dist.barrier()
        
        # Merge results (only on rank 0)
        if rank == 0:
            merge_results(output_file, world_size)
    
    # Cleanup
    if world_size > 1:
        cleanup_distributed()

def merge_results(output_file: str, world_size: int) -> None:
    """
    Merge all result files into one.
    
    Args:
        output_file (str): Final output file path
        world_size (int): Number of processes
    """
    base_path = os.path.splitext(output_file)[0]
    
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for rank in range(world_size):
                rank_file = f"{base_path}-rank{rank}.jsonl"
                if os.path.exists(rank_file):
                    with open(rank_file, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                    # Remove temporary file
                    os.remove(rank_file)
        
        print(f"Results merged and saved to {output_file}")
        
    except Exception as e:
        print(f"Error merging results: {e}")

def main():
    """Main function."""
    args = parse_args()
    
    # Validate CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires GPU support.")
        sys.exit(1)
    
    world_size = min(args.world_size, torch.cuda.device_count())
    
    print("=" * 60)
    print("Image Quality Assessment with vLLM")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Input files: {args.input_files}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of processes: {world_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if world_size > 1:
        # Use PyTorch multiprocessing to spawn processes
        mp.spawn(
            process_with_vllm,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        process_with_vllm(0, world_size, args)
    
    print("Artifacts assessment completed!")

if __name__ == "__main__":
    main()
