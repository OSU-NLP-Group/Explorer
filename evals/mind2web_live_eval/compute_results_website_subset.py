import sys, os
import json
import argparse
import pandas as pd
from mind2web_live_eval.experiment_results import read_json_result, calculate_total_score, write_to_json, write_task_result_to_df

def get_result(input_json_path):
    json_result_path = input_json_path + "/json_result"
    out_file_path = input_json_path + "/result"
    task_list = []
    for _, filename in enumerate(os.listdir(json_result_path)):
        id1, id2 = filename.replace('.json', '').split('_')

        # print(id1, id2)
        if id1 != id2:
            continue

        file_path = os.path.join(json_result_path, filename)
        out_json = {}
        task_name, task_status, reference_task_length, evaluate_steps, data_df = write_task_result_to_df(
            file_path)
        out_json["task_id"] = int(filename.split("_")[0])
        out_json["task_name"] = task_name
        out_json["task_status"] = task_status
        if os.path.isfile(file_path):
            try:
                task_step_list = write_to_json(data_df)
                out_json["step_list"] = task_step_list
            except:
                out_json["step_list"] = []
            out_json["evaluation"] = evaluate_steps
            task_list.append(out_json)

    task_list = sorted(task_list, key=lambda x: x['task_id'])

    return task_list

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-result-dir', type=str, required=True)
    parser.add_argument('--unfiltered', action='store_true', help='do not filter based on access denied urls')

    args = parser.parse_args()

    input_results = get_result(args.input_result_dir)
    # input_results_file = os.path.join(args.input_result_dir, 'result', 'out.json')
    # input_results = json.load(open(input_results_file))

    print('len(input_results) = {}'.format(len(input_results)))
    covered_task_ids = [x['task_id'] for x in input_results]
    all_task_ids = set(list(range(104)))
    task_ids_absent = all_task_ids - set(covered_task_ids)
    print('task_ids_absent = {}'.format(task_ids_absent))
    
    task_names = [x['task_name'] for x in input_results]
    # print('task_names = {}'.format(task_names))

    print('len(task_names) = {}'.format(len(task_names)))
    task_names = set(task_names)
    print('unique len(task_names) = {}'.format(len(task_names)))
    # print('unique task_names = {}'.format(task_names))

    # should be 104

    filter_website_file = 'mind2web_live_eval/m2w_live_valid_urls.json'
    valid_urls = json.load(open(filter_website_file))

    filter_results = []

    for task_dict in input_results:
        website_domain = task_dict['evaluation'][0]['reference_answer']

        is_valid_url = False

        if any([website_domain in x for x in valid_urls]):
            is_valid_url = True
        
        if is_valid_url:
            filter_results.append(task_dict)

    print('len(filter_results) = {}'.format(len(filter_results)))
    input_file_path = os.path.join(args.input_result_dir, 'result', 'out_filter.json')
    
    json.dump(filter_results, open(input_file_path, 'w'))

    if args.unfiltered:
        input_results_file = os.path.join(args.input_result_dir, 'result', 'out2.json')
        json.dump(input_results, open(input_results_file, 'w'))
        all_data = read_json_result(input_results_file)
    else:
        all_data = read_json_result(input_file_path)

    df = pd.DataFrame(all_data)
    print(df)
    print(df['task_id'].tolist())

    df["step_score"] = df["task_score"].apply(lambda x: float(x.split("/")[0]))
    try:
        df["efficiency_score"] =  df["steps"] / df["step_score"]
    except:
        df["efficiency_score"] = 0.0
        
    df["task_near_success"] = df["task_score"].apply(lambda x: float(
        x.split("/")[1]) - float(x.split("/")[0]) == 1.0)

    df_evaluate = df[["task_name", "status", "steps", "task_score",
                      "task_score_rate", "step_score", "efficiency_score", "task_near_success"]]

    print('df_evaluate = {}'.format(df_evaluate.shape))

    # logger.info('task_score = {}'.format(df_evaluate["task_score"]))
    # logger.info('task_near_success = {}'.format(df_evaluate["task_near_success"]))

    key_node_completion_rate = calculate_total_score(df_evaluate['task_score'])
    task_success_rate = df_evaluate[df_evaluate["status"]
                                  == "finished"].shape[0] / df_evaluate.shape[0]
    task_near_success_rate = df_evaluate[df_evaluate["task_near_success"]
                             == True].shape[0] / df_evaluate.shape[0]

    average_step_score_rate = df_evaluate["task_score_rate"].mean()
    average_efficiency_score = df_evaluate["efficiency_score"].mean()

    # print(df_evaluate[df_evaluate["status"] == "finished"])
    
    print('average_step_score_rate: {:.4f}'.format(average_step_score_rate))
    print('key_node_completion_rate: {:.4f}'.format(key_node_completion_rate))
    print('task_success_rate: {:.4f}'.format(task_success_rate))
    print('task_near_success_rate: {:.4f}'.format(task_near_success_rate))
    