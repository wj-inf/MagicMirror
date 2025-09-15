import json
import os
import argparse
from tqdm import tqdm
from base_info import _check_format, union_abnormal_labels
from mathruler.grader import extract_boxed_content

# out_dir = "output/"
# input_files = [
#     out_dir + "flux-schnell/merged_result_flux-schnell_vllm.jsonl",
#     out_dir + "sdxl/merged_result_sdxl_vllm.jsonl",
# ]

# Direktori output default, bisa tetap digunakan jika diperlukan
out_dir = "output/"
os.makedirs(out_dir, exist_ok=True) # Pastikan direktori output ada
output_file = out_dir + "merged_all_models.jsonl"


def main():
    parser = argparse.ArgumentParser(
        description="Gabungkan dan analisis hasil performa model dari beberapa file JSONL.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input_file', type=str,
                           help='Single JSONL file containing subject detection results')
    parser.add_argument('--input_files', nargs='+', 
                           help='Multiple JSONL files containing subject detection results')
    
    args = parser.parse_args()

    # Convert single file to list for uniform processing
    if args.input_file:
        args.input_files = [args.input_file]
    
    # Validate input files
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            parser.error(f"Input file does not exist: {input_file}")
            
    input_files = args.input_files
    # -----------------------------------------------------------------

    # Define detail categories
    detail_categories = [
        "human_single", "human_double", "human_multi",
        "animal_single", "animal_multi",
        "object_single", "object_multi", "object_compo",
    ]


    # Store all merged records, using dictionary with prompt_t2i as key
    records_by_prompt = {}

    # Process each input file
    for input_file in input_files: # <-- 4. Loop ini sekarang menggunakan input_files dari argumen
        # Extract model name from file path
        # Logika ini masih berfungsi dengan baik untuk path seperti 'output/flux-schnell/...'
        model_name = input_file.split("/")[-2]
        model_name_map = {
            'flux'          : "FLUX.1-dev",
            'seedream'      : "Seedream3.0",
            "qwen-image"    : "Qwen-image",
            "hidream-l1"    : "Hidream-l1",
            "flux-schnell"  : "FLUX.1-schnell",
            "sd3.5"         : "SD3.5",
            "kolors"        : "Kolors1.0",
            "sd3"           : "SD3",
            "sdxl"          : "SDXL",
            "gpt-image-1"   : "GPT-image-1",
            "bagel"         : "Bagel",
            "blip3o"        : "Blip3-o",
            "janus-pro"     : "Janus-pro",
            "showo"         : "Show-o",
        }
        model_name = model_name_map[model_name]
        print(f"Processing file: {input_file}, Model: {model_name}")

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in tqdm(lines, desc=f"Processing {model_name}"):
                    if line.strip():
                        try:
                            record = json.loads(line)

                            prompt_t2i = record.get('prompt_t2i', '')

                            record['model_name'] = model_name

                            if "response" in record:
                                predict_str = extract_boxed_content(record["response"])
                                format_valid = _check_format(predict_str, union_abnormal_labels)

                                re_data = {
                                    "format_valid": 1 if format_valid else 0,
                                    "Type of Deformity": 0,
                                    "L2: Irrational Element Interaction": 0,
                                    "L2: Abnormal Human Anatomy": 0,
                                    "L2: Abnormal Animal Anatomy": 0,
                                    "L2: Abnormal Object Morphology": 0
                                }

                                if record["has_subject"] == 0:
                                    re_data['Type of Deformity'] = 1
                                    if record["class"] == "human":
                                        re_data['L2: Abnormal Human Anatomy'] = 1
                                    elif record["class"] == "animal":
                                        re_data['L2: Abnormal Animal Anatomy'] = 1
                                    elif record["class"] == "object":
                                        re_data['L2: Abnormal Object Morphology'] = 1
                                    else:
                                        print(record["class"])
                                        print("ERROR!!!!!")
                                        exit()
                                elif format_valid:
                                    try:
                                        predict_dict = json.loads(predict_str)

                                        if 'Type of Deformity' in predict_dict:
                                            re_data['Type of Deformity'] = 1

                                            for key in ['L2: Irrational Element Interaction',
                                                       'L2: Abnormal Human Anatomy',
                                                       'L2: Abnormal Animal Anatomy',
                                                       'L2: Abnormal Object Morphology']:
                                                if key in predict_dict['Type of Deformity']:
                                                    re_data[key] = 1
                                    except:
                                        pass

                                record['re'] = re_data

                            if prompt_t2i:
                                if prompt_t2i not in records_by_prompt:
                                    records_by_prompt[prompt_t2i] = []
                                records_by_prompt[prompt_t2i].append(record)
                            else:
                                print(f"Warning: Found record without prompt_t2i field in file {input_file}")

                        except json.JSONDecodeError as e:
                            print(f"Parse error in file {input_file}, line: {line[:100]}... Error: {e}")
        except FileNotFoundError:
            print(f"File not found: {input_file}")

    print("All files processed, starting to write merged file...")

    # Sisa dari skrip Anda tetap sama...
    # (Perhitungan statistik dan pencetakan tabel)

    # Calculate statistics for each model
    model_stats = {}

    for prompt in records_by_prompt.values():
        for record in prompt:
            model_name = record['model_name']
            re_data = record.get('re', {})
            class_type = record.get('class', '')
            detail_class = record.get('detail_class', '')

            if model_name not in model_stats:
                model_stats[model_name] = {
                    "total": 0,
                    "valid_format": 0,
                    "human_valid": 0,
                    "animal_valid": 0,
                    "object_valid": 0,
                    "Type of Deformity": 0,
                    "L2: Irrational Element Interaction": 0,
                    "L2: Abnormal Human Anatomy": 0,
                    "L2: Abnormal Animal Anatomy": 0,
                    "L2: Abnormal Object Morphology": 0
                }

                for category in detail_categories:
                    model_stats[model_name][f"{category}_valid"] = 0
                    model_stats[model_name][f"{category}_deformity"] = 0

            model_stats[model_name]["total"] += 1

            if re_data.get("format_valid", 0) == 1:
                model_stats[model_name]["valid_format"] += 1

                if class_type == "human" or class_type == "person":
                    model_stats[model_name]["human_valid"] += 1
                elif class_type == "animal":
                    model_stats[model_name]["animal_valid"] += 1
                elif class_type == "object":
                    model_stats[model_name]["object_valid"] += 1

                if detail_class in detail_categories:
                    model_stats[model_name][f"{detail_class}_valid"] += 1

                    deformity_detected = False
                    if detail_class.startswith("human_") and re_data.get("L2: Abnormal Human Anatomy", 0) == 1:
                        deformity_detected = True
                    elif detail_class.startswith("animal_") and re_data.get("L2: Abnormal Animal Anatomy", 0) == 1:
                        deformity_detected = True
                    elif detail_class.startswith("object_") and re_data.get("L2: Abnormal Object Morphology", 0) == 1:
                        deformity_detected = True

                    if deformity_detected:
                        model_stats[model_name][f"{detail_class}_deformity"] += 1

                for key in ['Type of Deformity', 'L2: Irrational Element Interaction',
                           'L2: Abnormal Human Anatomy', 'L2: Abnormal Animal Anatomy',
                           'L2: Abnormal Object Morphology']:
                    if re_data.get(key, 0) == 1:
                        model_stats[model_name][key] += 1

    # Print original statistics table
    print("\n" + "=" * 150)
    header_format = "{:<15} & {:<17} & {:<15} & {:<15} & {:<15} & {:<15}"
    row_format = "{:<15} & {:<17.2f} & {:<15.2f} & {:<15.2f} & {:<15.2f} & {:<15.2f} \\\\"

    print(header_format.format(
        "Model", "Interaction Score", "Human Score",
        "Animal Score", "Object Score", "Overall Score"
    ))
    print("-" * 150)

    for model_name, stats in model_stats.items():
        valid_count = stats["valid_format"]
        human_valid = stats["human_valid"]
        animal_valid = stats["animal_valid"]
        object_valid = stats["object_valid"]

        if valid_count > 0:
            interaction_score = 100 * (1 - stats["L2: Irrational Element Interaction"] / valid_count)
            human_score = 100 * (1 - stats["L2: Abnormal Human Anatomy"] / human_valid) if human_valid > 0 else 0
            animal_score = 100 * (1 - stats["L2: Abnormal Animal Anatomy"] / animal_valid) if animal_valid > 0 else 0
            object_score = 100 * (1 - stats["L2: Abnormal Object Morphology"] / object_valid) if object_valid > 0 else 0
            overall_score = 100 * (1 - stats["Type of Deformity"] / valid_count)

            print(row_format.format(
                model_name,
                interaction_score,
                human_score,
                animal_score,
                object_score,
                overall_score
            ))
        else:
            print(f"{model_name}: No valid format responses")

    print("=" * 150)

    # ... (sisa kode untuk mencetak tabel statistik detail tetap sama) ...
    # Print detail category statistics table
    print("\n🔍 Detail Category Score Statistics:")
    print("=" * 200)

    header_parts = ["Model"]
    for category in detail_categories:
        header_parts.append(category[:10])

    header_format = " & ".join(["{:<15}"] + ["{:<12}" for _ in detail_categories])
    print(header_format.format(*header_parts))
    print("-" * 200)

    for model_name, stats in model_stats.items():
        if stats["valid_format"] > 0:
            row_data = [model_name]

            for category in detail_categories:
                valid_key = f"{category}_valid"
                deformity_key = f"{category}_deformity"

                if stats[valid_key] > 0:
                    score = 100 * (1 - stats[deformity_key] / stats[valid_key])
                    row_data.append(f"{score:.2f}")
                else:
                    row_data.append("0.00")

            data_format = " & ".join(["{:<15}"] + ["{:<12}" for _ in detail_categories]) + "\\\\"
            print(data_format.format(*row_data))

    print("=" * 200)

    print("\n📊 Detail Category Detailed Statistics:")
    print("=" * 180)

    print("Sample Count Statistics:")
    header_parts = ["Model", "Total", "Valid"] + [cat[:10] for cat in detail_categories]
    header_format = " & ".join(["{:<15}", "{:<7}", "{:<7}"] + ["{:<10}" for _ in detail_categories])
    print(header_format.format(*header_parts))
    print("-" * 180)

    for model_name, stats in model_stats.items():
        if stats["valid_format"] > 0:
            row_data = [model_name, str(stats["total"]), str(stats["valid_format"])]

            for category in detail_categories:
                valid_key = f"{category}_valid"
                row_data.append(str(stats[valid_key]))

            data_format = " & ".join(["{:<15}", "{:<7}", "{:<7}"] + ["{:<10}" for _ in detail_categories])
            print(data_format.format(*row_data))

    print("\nDeformity Count Statistics:")
    print(header_format.format(*header_parts))
    print("-" * 180)

    for model_name, stats in model_stats.items():
        if stats["valid_format"] > 0:
            row_data = [model_name, str(stats["total"]), str(stats["valid_format"])]

            for category in detail_categories:
                deformity_key = f"{category}_deformity"
                row_data.append(str(stats[deformity_key]))

            data_format = " & ".join(["{:<15}", "{:<7}", "{:<7}"] + ["{:<10}" for _ in detail_categories])
            print(data_format.format(*row_data))

    print("=" * 180)

    print("\n📈 Model Detail Category Performance Summary:")
    print("=" * 100)

    for model_name, stats in model_stats.items():
        if stats["valid_format"] > 0:
            print(f"\n🤖 {model_name}:")
            print("-" * 60)

            for category in detail_categories:
                valid_key = f"{category}_valid"
                deformity_key = f"{category}_deformity"

                valid_count = stats[valid_key]
                deformity_count = stats[deformity_key]

                if valid_count > 0:
                    score = 100 * (1 - deformity_count / valid_count)
                    print(f"  {category:<20}: {valid_count:>3} samples, {deformity_count:>2} deformities, {score:>6.2f} score")
                else:
                    print(f"  {category:<20}: {valid_count:>3} samples, {deformity_count:>2} deformities, {'N/A':>6} score")

    print("=" * 100)

# --- 5. Pindahkan semua logika ke dalam fungsi main() dan panggil di sini ---
if __name__ == "__main__":
    main()