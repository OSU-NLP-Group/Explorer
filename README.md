# Explorer

<p align="center">
  <a href="https://aclanthology.org/events/acl-2025/"><img src="https://img.shields.io/badge/ACL_Findings-2025-purple"></a>
  <a href="https://aclanthology.org/2025.findings-acl.326.pdf"><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></a>
  <a href="https://github.com/OSU-NLP-Group/Explorer/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/MIT-License-blue">
  </a>
</p>

This is the official codebase for **Explorer: Scaling Exploration-driven Web Trajectory Synthesis for Multimodal Web Agents** [**ACL 2025 (Findings)**]. This project is a collaboration between The Ohio State University and Microsoft Research.

- [🏠 Website](https://osu-nlp-group.github.io/Explorer/)
- [📖 Paper](https://aclanthology.org/2025.findings-acl.326.pdf)

Stay tuned for additional code releases and modules for this project.

## 🚀 Trajectory Synthesis
```
Xvfb :99 -screen 0 1920x1280x16 &
export DISPLAY=:99
export OPENAI_API_KEY=xxxxxxxxxxxx

python -m traj_gen.main \
  --model-dir MODEL_DIR \
  --init-url INIT_URL \
  --max-steps MAX_STEPS
```

## 📦 Training

### Mind2Web-Live
```
cd train/
torchrun --nproc_per_node=4 train_qwen2vl.py \
  --use_flash_attention --bf16 \
  --train_dir <PATH_TO_SAVED_HUGGINGFACE_TRAJ_DATASET> \
  --train_data_dir <PATH_ROOT_TO_RAW_TRAJS> \
  --output_dir <OUTPUT_DIR> \
  --num_train_epochs 10 \
  --batch_size 64 \
  --use-google-search
```

### Multimodal-Mind2Web
```
cd train/
torchrun --nproc_per_node=4 train_qwen2vl.py \
  --use_flash_attention --bf16 \
  --train_dir <PATH_TO_SAVED_HUGGINGFACE_TRAJ_DATASET> \
  --train_dir_order <EMPTY_DIR_TO_SAVE_ORDERED_HUGGINGFACE_TRAJ_DATASET> \
  --train_data_dir <PATH_ROOT_TO_RAW_TRAJS> \
  --output_dir <OUTPUT_DIR> \
  --num_train_epochs 2 \
  --batch_size 64 \
  --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
  --use-nogoto-gs-format \
  --order_all_steps \
  --learning_rate 1e-5
```

## 🧪 Evaluation

### Mind2Web-Live

**Step 1:** Installation
```
conda create --name myenv python=3.12.5
pip install -r evals/mind2web_live_eval/requirements.txt
```

**Step 2:** Start x server and set the DISPLAY environment variable
```
Xvfb :99 -screen 0 1920x1280x16 &
export DISPLAY=:99
export OPENAI_API_KEY=xxxxxxxxxxxx
```

**Step 3:** Run the evaluation script:
```
python -m evals.mind2web_live_eval.evaluate_model    --index -1     --planning_text_model {qwen2-vl-7b|phi-3.5v}     --toml-path evals/mind2web_live_eval/configs/setting.toml     --use-flash-attention --ckpt-path CKPT_PATH --temp 0.01 --log-dir LOG_DIR --viewport-width 1280
```

### Multimodal-Mind2Web

To evaluate the performance of the trained model on the Multimodal-Mind2Web benchmark:

**Step 1:** Installation
```
conda create --name myenv python=3.12.5
pip install -r evals/mind2web_orig_eval/requirements.txt
```

**Step 2:** Download the DeBERTa candidate generation scores from the following link:

[🔗 DeBERTa Score File](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/deng_595_buckeyemail_osu_edu/EZllMua3lABAhXQnCN7-pr4BIP4YV8xPfbgyP5FXT18wag?e=yXkK8k)

**Step 3:** Run the evaluation script:

```
cd evals
python -m mind2web_orig_eval.eval \
  --use-flash-attention \
  --ckpt-path <CKPT_PATH> \
  --log-dir <LOG_DIR> \
  --score-file <PATH_TO_DEBERTA_FILE> \
  --split {test_domain|test_task|test_website} \
  --model {qwen-7b|phi-3.5}
```

### In-domain evaluation

**Step 1:** Installation
```
conda create --name myenv python=3.12.5
pip install -r evals/in_domain_eval/requirements.txt
```

**Step 2:** Set necessary env variables (`OPENAI_API_KEY` for evaluating API-based models)
```
export OPENAI_API_KEY=xxxxxxxxxxxx
```

**Step 3:** Run the evaluation script:

```
python -u -m evals.in_domain_eval.eval --input-file in_domain_test.json --ckpt-path <CKPT_PATH> --use-flash-attention --log-dir <LOG_DIR> --use-spiral
```

Structure of `in_domain_test.json`:
```
[
 <path to traj dir 1>,
 <path to traj dir 2>,
 ...
 <path to traj dir n>,
]
```

### MiniWoB++ 

**Step 1:** Installation
```
conda create --name myenv python=3.12.5
pip install -r evals/miniwob/requirements.txt
```

**Step 2:** Run the evaluation script:

```
bash evals/miniwob/eval-explorer.sh
```

## Citation

If you find this work useful, please consider starring our repo and citing our paper: 

```
@inproceedings{pahuja-etal-2025-explorer,
    title = "Explorer: Scaling Exploration-driven Web Trajectory Synthesis for Multimodal Web Agents",
    author = "Pahuja, Vardaan  and
      Lu, Yadong  and
      Rosset, Corby  and
      Gou, Boyu  and
      Mitra, Arindam  and
      Whitehead, Spencer  and
      Su, Yu  and
      Awadallah, Ahmed Hassan",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.326/",
    pages = "6300--6323",
    ISBN = "979-8-89176-256-5",
}
```
