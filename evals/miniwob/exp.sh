conda activate /home/pahuja.9/research_nfs/conda_envs/miniwob

CUDA_VISIBLE_DEVICES=6 bash miniwob/eval-slm.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm.txt 2>&1

CUDA_VISIBLE_DEVICES=5 bash miniwob/eval-slm.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_2.txt 2>&1

CUDA_VISIBLE_DEVICES=6 bash miniwob/eval-slm.sh --parse-two-dicts > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_3.txt 2>&1

CUDA_VISIBLE_DEVICES=6 bash miniwob/eval-slm.sh --parse-two-dicts > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_4.txt 2>&1

CUDA_VISIBLE_DEVICES=6 bash miniwob/eval-slm.sh --parse-two-dicts > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_5.txt 2>&1

# acc tree fixes
CUDA_VISIBLE_DEVICES=3 bash miniwob/eval-slm.sh --parse-two-dicts > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_6.txt 2>&1

# ~/research_nfs/web_traj_gen/m2w_train_v3_visible_ckpts_phi3.5/m2w_train_visible_ckpts_phi3.5/
CUDA_VISIBLE_DEVICES=7 bash miniwob/eval-slm-m2wtrain.sh --parse-two-dicts > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_m2wtrain.txt 2>&1

bash miniwob/eval-gpt-3.5-turbo.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_gpt35_complete_2.txt 2>&1

# test type action

python -u -m miniwob.test_type_action

python -u -m miniwob.main --env enter-text --llm chatgpt --num-episodes 4 --erci 1 --irci 3 --sgrounding > ~/research_nfs/web_traj_gen/miniwob_logs/log_gpt3.5_entertext.txt 2>&1

# GPT 3.5, count shape
python -u -m miniwob.main --env count-shape --llm chatgpt --num-episodes 4 --erci 1 --irci 3 --sgrounding > ~/research_nfs/web_traj_gen/miniwob_logs/log_gpt3.5_countshape.txt 2>&1

# test on enter-text
CUDA_VISIBLE_DEVICES=6 python -u -m miniwob.main_slm --env enter-text --llm chatgpt --num-episodes 4 --erci 1 --irci 3 --sgrounding --ckpt-path ~/research_nfs/web_traj_gen/ckpts/phi3.5_autogen_42_1_m2wtrain_sample_10epoch_10k > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_entertext.txt 2>&1

# test on grid-coordinate
CUDA_VISIBLE_DEVICES=1 python -u -m miniwob.main_slm --env grid-coordinate --llm chatgpt --num-episodes 4 --erci 1 --irci 3 --sgrounding --ckpt-path ~/research_nfs/web_traj_gen/ckpts/phi3.5_autogen_42_1_m2wtrain_sample_10epoch_10k > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_grid_coord.txt 2>&1

# click-color
CUDA_VISIBLE_DEVICES=6 python -u -m miniwob.main_slm --env click-color --llm chatgpt --num-episodes 4 --erci 1 --irci 3 --sgrounding --ckpt-path ~/research_nfs/web_traj_gen/ckpts/phi3.5_autogen_42_1_m2wtrain_sample_10epoch_10k > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_click_color.txt 2>&1

# click-link
CUDA_VISIBLE_DEVICES=6 python -u -m miniwob.main_slm --env click-link --llm chatgpt --num-episodes 4 --erci 1 --irci 3 --sgrounding --ckpt-path ~/research_nfs/web_traj_gen/ckpts/phi3.5_autogen_42_1_m2wtrain_sample_10epoch_10k > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_click_link.txt 2>&1

# login-user
CUDA_VISIBLE_DEVICES=6 python -u -m miniwob.main_slm --env login-user --llm chatgpt --num-episodes 4 --erci 1 --irci 3 --sgrounding --ckpt-path ~/research_nfs/web_traj_gen/ckpts/phi3.5_autogen_42_1_m2wtrain_sample_10epoch_10k > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_login_user.txt 2>&1

time CUDA_VISIBLE_DEVICES=7 bash miniwob/eval-slm-dynamic-multiple.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=5 bash miniwob/eval-slm-dynamic-single.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_single.txt 2>&1 &
# time CUDA_VISIBLE_DEVICES=6 bash miniwob/eval-slm-fixed-multiple.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_fixed_multiple.txt 2>&1 &
# time CUDA_VISIBLE_DEVICES=7 bash miniwob/eval-slm-fixed-single.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_fixed_single.txt 2>&1 &

# expts with old script
time CUDA_VISIBLE_DEVICES=1 bash miniwob/eval-slm-dynamic-single-oldjs.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_single_oldjs.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=7 bash miniwob/eval-slm-dynamic-single-oldjs-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_single_oldjs_2.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=6 bash miniwob/eval-slm-dynamic-single-oldjs-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_single_oldjs_3.txt 2>&1 &

time CUDA_VISIBLE_DEVICES=0 bash miniwob/eval-slm-dynamic-multiple-2stage.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_2stage.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=7 bash miniwob/eval-slm-dynamic-multiple-2stage-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_2stage_2.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=0 bash miniwob/eval-slm-dynamic-multiple-2stage-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_2stage_3.txt 2>&1 &

# additional runs for dynamic multiple
time CUDA_VISIBLE_DEVICES=1 bash miniwob/eval-slm-dynamic-multiple-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_2.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=7 bash miniwob/eval-slm-dynamic-multiple-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_3.txt 2>&1 &

time CUDA_VISIBLE_DEVICES=1 bash miniwob/eval-slm-dynamic-multiple-4.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_4.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-5.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_5.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=0 bash miniwob/eval-slm-dynamic-multiple-6.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_6.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=5 bash miniwob/eval-slm-dynamic-multiple-7.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_7.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=6 bash miniwob/eval-slm-dynamic-multiple-8.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_8.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=7 bash miniwob/eval-slm-dynamic-multiple-9.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_9.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=7 bash miniwob/eval-slm-dynamic-multiple-10.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_10.txt 2>&1 &

python -m miniwob.compute_overall_metric ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_2.txt
python -m miniwob.compare_model_errors --log-file-1 ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple.txt --log-file-2 ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_2stage.txt

time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-mapselect.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_mapselect.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=0 bash miniwob/eval-slm-dynamic-multiple-mapselect-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_mapselect_2.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=7 bash miniwob/eval-slm-dynamic-multiple-mapselect-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_mapselect_3.txt 2>&1 &

python -u -m miniwob.main_slm_gym --env choose-list --llm chatgpt --num-episodes 4 --erci 1 --irci 3 --sgrounding --ckpt-path ~/research_nfs/web_traj_gen/ckpts/phi3.5_autogen_42_1_m2wtrain_sample_10epoch_10k --use-dynamic-seed

time CUDA_VISIBLE_DEVICES=6 bash miniwob/eval-slm-dynamic-multiple-gym.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_gym.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=7 bash miniwob/eval-slm-dynamic-multiple-gym-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_gym_2.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=7 bash miniwob/eval-slm-dynamic-multiple-gym-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_gym_3.txt 2>&1 &

# screenhot crop, fix type, color
time CUDA_VISIBLE_DEVICES=6 bash miniwob/eval-slm-dynamic-multiple-gym-v2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_gym_v2.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=6 bash miniwob/eval-slm-dynamic-multiple-gym-v2-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_gym_v2_2.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=6 bash miniwob/eval-slm-dynamic-multiple-gym-v2-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_gym_v2_3.txt 2>&1 &

# screenhot crop, fix type, color only for color elements
time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-gym-v3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_gym_v3.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=6 bash miniwob/eval-slm-dynamic-multiple-gym-v3-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_gym_v3_2.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=5 bash miniwob/eval-slm-dynamic-multiple-gym-v3-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_gym_v3_3.txt 2>&1 &

# s10+m2w train model
time CUDA_VISIBLE_DEVICES=1 bash miniwob/eval-slm-dynamic-multiple-s10.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_s10.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=5 bash miniwob/eval-slm-dynamic-multiple-s10-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_s10_2.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=5 bash miniwob/eval-slm-dynamic-multiple-s10-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_s10_3.txt 2>&1 &

# text fix, body, t tag fix, xpath fix for input elements
time CUDA_VISIBLE_DEVICES=3 bash miniwob/eval-slm-dynamic-multiple-gym-v4.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_gym_v4.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=3 bash miniwob/eval-slm-dynamic-multiple-gym-v4-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_gym_v4_2.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=6 bash miniwob/eval-slm-dynamic-multiple-gym-v4-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_gym_v4_3.txt 2>&1 &

# remove overlap in SOM
time CUDA_VISIBLE_DEVICES=3 bash miniwob/eval-slm-dynamic-multiple-gym-v5.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_gym_v5.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=3 bash miniwob/eval-slm-dynamic-multiple-gym-v5-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_gym_v5_2.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=3 bash miniwob/eval-slm-dynamic-multiple-gym-v5-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_gym_v5_3.txt 2>&1 &

# dynamic multiple with class and id added
time CUDA_VISIBLE_DEVICES=3 bash miniwob/eval-slm-dynamic-multiple-v2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v2.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=5 bash miniwob/eval-slm-dynamic-multiple-v2-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v2_2.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-v2-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v2_3.txt 2>&1 &

# dynamic multiple with class added
time CUDA_VISIBLE_DEVICES=3 bash miniwob/eval-slm-dynamic-multiple-v3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v3.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=0 bash miniwob/eval-slm-dynamic-multiple-v3-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v3_2.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-v3-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v3_3.txt 2>&1 &

time CUDA_VISIBLE_DEVICES=3 bash miniwob/eval-slm-toy.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_1.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=3 bash miniwob/eval-slm-toy-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_2.txt 2>&1 &

# omit task desc element in acc tree
CUDA_VISIBLE_DEVICES=7 bash miniwob/eval-slm-dynamic-multiple-omitdesc.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_omitdesc.txt 2>&1
CUDA_VISIBLE_DEVICES=4 bash miniwob/eval-slm-dynamic-multiple-omitdesc-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_omitdesc_2.txt 2>&1
CUDA_VISIBLE_DEVICES=3 bash miniwob/eval-slm-dynamic-multiple-omitdesc-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_omitdesc_3.txt 2>&1

# omit empty div in acc tree and SOM
CUDA_VISIBLE_DEVICES=4 bash miniwob/eval-slm-dynamic-multiple-omitdiv.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_omitdiv.txt 2>&1
CUDA_VISIBLE_DEVICES=4 bash miniwob/eval-slm-dynamic-multiple-omitdiv-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_omitdiv_2.txt 2>&1
CUDA_VISIBLE_DEVICES=4 bash miniwob/eval-slm-dynamic-multiple-omitdiv-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_omitdiv_3.txt 2>&1

# omit empty div and desc in acc tree and SOM
CUDA_VISIBLE_DEVICES=4 bash miniwob/eval-slm-dynamic-multiple-omitdivdesc.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_omitdivdesc.txt 2>&1
CUDA_VISIBLE_DEVICES=5 bash miniwob/eval-slm-dynamic-multiple-omitdivdesc-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_omitdivdesc_2.txt 2>&1
CUDA_VISIBLE_DEVICES=4 bash miniwob/eval-slm-dynamic-multiple-omitdivdesc-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_omitdivdesc_3.txt 2>&1

# remove elements with empty text from acc tree
time CUDA_VISIBLE_DEVICES=7 bash miniwob/eval-slm-dynamic-multiple-gym-v6.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_gym_v6.txt 2>&1
time CUDA_VISIBLE_DEVICES=4 bash miniwob/eval-slm-dynamic-multiple-gym-v6-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_gym_v6_2.txt 2>&1
time CUDA_VISIBLE_DEVICES=6 bash miniwob/eval-slm-dynamic-multiple-gym-v6-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_gym_v6_3.txt 2>&1

# original, temp=0.1
CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-t-0.1.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_0.1.txt 2>&1 && CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-t-0.1-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_0.1_2.txt 2>&1 && CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-t-0.1-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_0.1_3.txt 2>&1

# original, temp=0.7
CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-t-0.7.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_0.7.txt 2>&1 && CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-t-0.7-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_0.7_2.txt 2>&1 && CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-t-0.7-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_0.7_3.txt 2>&1

# use whole dict for last action
time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-v4.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v4.txt 2>&1 &
time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-v4-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v4_2.txt 2>&1
time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-v4-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v4_3.txt 2>&1

# general fixes
time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-v5.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v5.txt 2>&1
time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-v5-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v5_2.txt 2>&1
time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-v5-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v5_3.txt 2>&1
time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-v5-4.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v5_4.txt 2>&1
time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-v5-5.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v5_5.txt 2>&1

# test single task
CUDA_VISIBLE_DEVICES=0 python -u -m miniwob.main_slm --env click-shades --llm chatgpt --num-episodes 4 --erci 1 --irci 3 --sgrounding --ckpt-path ~/research_nfs/web_traj_gen/ckpts/phi3.5_autogen_42_1_m2wtrain_sample_10epoch_10k --use-dynamic-seed  > tmp.txt 2>&1

# test email
CUDA_VISIBLE_DEVICES=0 python -u -m miniwob.main_slm --env email-inbox --llm chatgpt --num-episodes 4 --erci 1 --irci 3 --sgrounding --ckpt-path ~/research_nfs/web_traj_gen/ckpts/phi3.5_autogen_42_1_m2wtrain_sample_10epoch_10k --use-dynamic-seed --add-class > tmp.txt 2>&1

# dones fix
time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-v6.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v6.txt 2>&1
time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-v6-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v6_2.txt 2>&1
time CUDA_VISIBLE_DEVICES=1 bash miniwob/eval-slm-dynamic-multiple-v6-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v6_3.txt 2>&1

python -m miniwob.compute_max_aggregate_result --log-file-1 ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v5.txt --log-file-2 ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v5_2.txt --log-file-3 ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v5_3.txt

python -m miniwob.compute_max_aggregate_result --log-file-1 ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v6.txt --log-file-2 ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v6_2.txt --log-file-3 ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v6_3.txt

# add class
# dones fix
time CUDA_VISIBLE_DEVICES=6 bash miniwob/eval-slm-dynamic-multiple-v7.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v7.txt 2>&1
time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-v7-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v7_2.txt 2>&1
time CUDA_VISIBLE_DEVICES=1 bash miniwob/eval-slm-dynamic-multiple-v7-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v7_3.txt 2>&1

# add class for subset
time CUDA_VISIBLE_DEVICES=5 bash miniwob/eval-slm-dynamic-multiple-v8.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v8.txt 2>&1
time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-v8-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v8_2.txt 2>&1
time CUDA_VISIBLE_DEVICES=1 bash miniwob/eval-slm-dynamic-multiple-v8-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v8_3.txt 2>&1

python -m miniwob.compute_max_aggregate_result --log-file-1 ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v8.txt --log-file-2 ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v8_2.txt --log-file-3 ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v8_3.txt

# add class id plus class for subset
time CUDA_VISIBLE_DEVICES=5 bash miniwob/eval-slm-dynamic-multiple-v9.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v9.txt 2>&1
time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-v9-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v9_2.txt 2>&1
time CUDA_VISIBLE_DEVICES=1 bash miniwob/eval-slm-dynamic-multiple-v9-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v9_3.txt 2>&1
python -m miniwob.compute_max_aggregate_result --log-file-1 ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v9.txt --log-file-2 ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v9_2.txt --log-file-3 ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v9_3.txt

# redo with actual ckpt
time CUDA_VISIBLE_DEVICES=5 bash miniwob/eval-slm-dynamic-multiple-v10.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v10.txt 2>&1
time CUDA_VISIBLE_DEVICES=2 bash miniwob/eval-slm-dynamic-multiple-v10-2.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v10_2.txt 2>&1
time CUDA_VISIBLE_DEVICES=1 bash miniwob/eval-slm-dynamic-multiple-v10-3.sh > ~/research_nfs/web_traj_gen/miniwob_logs/log_slm_dynamic_multiple_v10_3.txt 2>&1
