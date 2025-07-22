import sys, os
import json
import traceback
from collections import Counter

eval_data = []
cnt = 0
path = "/home/pahuja.9/research_nfs/web_traj_gen/gold_6k/m2w_127k_single_r4_t0.01_s10_dedup_fix_verifier_mentionsite/"

traj_dirs = os.listdir(path)
n_actions_list = []
task_desc_list = []

v2_dirs = [
    "10350",
    "10177",
    "10403",
    "10301",
    "10150",
    "10221",
    "10131",
    "10304",
    "10425",
    "10316",
    "10324",
    "10062",
    "10422",
    "10059",
    "10180",
    "10260",
    "1045",
    "10222",
    "10348",
    "10429",
    "10041",
    "10090",
    "10394",
    "10745",
    "10824",
    "10554",
    "10826",
    "10515",
    "10768",
    "10592",
    "10896",
    "10484",
    "10931",
    "10926",
    "10519",
    "10762",
    "10710",
    "10667",
    "10724",
    "10490",
    "10727",
    "11316",
    "10953",
    "10976",
    "1108",
    "11099",
    "11295",
    "11292",
    "11317",
    "11302",
    "11255",
    "11121",
    "11402",
    "11010",
    "11033",
    "11046",
    "11045",
    "11125",
    "1139",
    "10940",
    "11065",
    "11017",
    "11275",
    "11375",
    "11233",
    "11569",
    "11800",
    "11470",
    "11497",
    "11485",
    "11767",
    "11785",
    "11543",
    "1181",
    "11867",
    "1163",
    "11844",
    "11861",
    "11475",
    "11641",
    "12238",
    "11963",
    "11922",
    "12171",
    "11973",
    "12106",
    "11917",
    "12018",
    "12224",
    "11901",
    "12024",
    "12352",
    "12007",
    "12098",
    "12313",
    "12616",
    "12553",
    "12609",
    "12475",
    "12459",
]

traj_dirs = list(set(traj_dirs) - set(v2_dirs))
# print(traj_dirs)

for i, folder in enumerate(traj_dirs):
    if i % 1000 == 0:
        print(i)
        print("cnt", cnt)

    js_file = os.path.join(path, folder, "task_trajectory_data.json")

    # print(js_file)

    if not os.path.exists(js_file):
        continue

    # print(js_file)

    with open(js_file, "r") as f:
        data = json.load(f)
    try:
        if len(data) == 0:
            continue
        is_success = (
            "success" in data["verifier_agent_response"].split("\nStatus: ")[-1]
        )
        n_scroll = sum(
            [
                1
                for action in data["actions"]
                if "new_action_grounded" in action
                and action["new_action_grounded"] is not None
                and action["new_action_grounded"].startswith("scroll")
            ]
        )
        n_actions = sum(
            [
                1
                for action in data["actions"]
                if "new_action_grounded" in action
                and action["new_action_grounded"] is not None
            ]
        )

        if (
            is_success
            and n_scroll < 2
            and "API call failed" not in data["task_summary"]
            and "regex" not in data["task_summary"]
            and n_actions >= 5
            and n_actions <= 7
        ):
            # if is_success:
            eval_data.append(os.path.join(path, folder))
            n_actions_list.append(n_actions)
            task_desc_list.append(data["task_summary"])
            cnt += 1

        if cnt >= 6:
            break

    except:
        traceback.print_exc()
        continue

action_c = Counter(n_actions_list)
print(action_c.most_common())

print(eval_data)
print(task_desc_list)

json.dump(eval_data, open("in_domain_test_v3.json", "w"), indent=1)
