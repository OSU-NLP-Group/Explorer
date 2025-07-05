import json
import argparse

def compute_average_step_sr(jsonl_file):
    total_sr = 0
    total_ele_match = 0
    total_op_f1 = 0

    count = 0

    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            # if 'step_SR' in data:
            total_sr += data['step_SR']
            total_ele_match += data['avg_ele_match']
            total_op_f1 += data['avg_op_f1']
            count += 1

    if count == 0:
        return 0

    print(f"Total SR: {total_sr}")
    print(f"Total ele match: {total_ele_match}")
    print(f"Total op F1: {total_op_f1}")
    
    print(f"Count: {count}")
    average_sr = total_sr / count
    average_ele_match = total_ele_match / count
    average_op_f1 = total_op_f1 / count

    return average_sr, average_ele_match, average_op_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl-file', type=str, required=True, help='Path to the JSONL file')
    args = parser.parse_args()
    
    # Compute the average step_SR
    average_sr, average_ele_match, average_op_f1 = compute_average_step_sr(args.jsonl_file)

    print(f"The average step_SR is: {average_sr:.4f}")
    print(f"The average ele_match is: {average_ele_match:.4f}")
    print(f"The average op_f1 is: {average_op_f1:.4f}")