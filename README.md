# Explorer

This is the official codebase for **Explorer: Scaling Exploration-driven Web Trajectory Synthesis for Multimodal Web Agents**, accepted to **ACL 2025 (Findings)**. This project is a collaboration between The Ohio State University and Microsoft Research.

- [🏠Website](https://osu-nlp-group.github.io/Explorer/)
- [📖Paper](https://arxiv.org/pdf/2502.11357)

Stay tuned for additional code releases and modules for this project.

## 🧪 Evaluation

### Multimodal-Mind2Web

To evaluate the performance of the trained model on the Multimodal-Mind2Web benchmark:

**Step 1:** Download the DeBERTa candidate generation scores from the following link:

[🔗 DeBERTa Score File](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/deng_595_buckeyemail_osu_edu/EZllMua3lABAhXQnCN7-pr4BIP4YV8xPfbgyP5FXT18wag?e=yXkK8k)

**Step 2:** Run the evaluation script:

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

## Citation

If you find this work useful, please consider starring our repo and citing our paper: 

```
@article{pahuja2025explorer,
  title={Explorer: Scaling Exploration-driven Web Trajectory Synthesis for Multimodal Web Agents},
  author={Pahuja, Vardaan and Lu, Yadong and Rosset, Corby and Gou, Boyu and Mitra, Arindam and Whitehead, Spencer and Su, Yu and Awadallah, Ahmed},
  journal={arXiv preprint arXiv:2502.11357},
  year={2025}
}
```