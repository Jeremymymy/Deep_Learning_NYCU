# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Coursework for the NYCU Deep Learning course (student ID `311351009`). Six **independent, self-contained labs** (`Lab1`–`Lab6`), each implementing a different model family from a course spec. There is **no shared code** between labs — each lab has its own `dataloader.py`, model definitions, and entry point. The `README.md` inside each lab is a Traditional-Chinese write-up explaining the technique and experimental findings; read it first to understand a lab's intent.

There is **no build system, no test framework, no requirements file, and no linter config.** Labs are run directly with `python`. CUDA is assumed throughout (`.to('cuda')`, `torch.cuda.FloatTensor(...)`); none of the scripts gracefully fall back to CPU except Lab6.

## Critical gotcha: hardcoded absolute paths

Several scripts contain absolute paths to the original training server (`/home/pp037/DL/...`) that **will not exist anywhere else** and must be edited before running:

- `Lab3/dataloader.py` — `train_df`/`valid_df` read `/home/pp037/DL/Lab3/{train,valid}.csv`
- `Lab6/dataloader.py` — `root`, `objects.json`, and the `iclevr/` image dir
- `Lab6/evaluator.py:40` — loads `/home/pp037/DL/Lab6/checkpoint.pth` (the pretrained ResN18 evaluator)
- `Lab6/main.py` — `--model_path` / `--data_path` argparse defaults

Lab2 is the opposite trap: `read_bci_data()` loads `S4b_train.npz` etc. by **bare relative name**, but the files live in `Lab2/data/`. Run it from `Lab2/data/` or move the files, or it will fail to find them.

## How to run each lab

```bash
# Lab1 — NumPy-only NN (no torch). Self-contained, generates its own data.
python Lab1/Lab1_311351009_code.py

# Lab2 — EEGNet & DeepConvNet. Single script trains all activation-function variants in sequence.
#   Run from a dir where the .npz data files are visible (see gotcha above).
python Lab2_311351009_code.py

# Lab3 — ResNet18/50/152 for leukemia classification.
#   main.py trains all three sequentially, saving model_18/50/152.pkl. Edit CSV paths first.
python Lab3/main.py        # train
python Lab3/test.py        # evaluate with TTA + confusion matrix

# Lab4 — Conditional VAE video prediction. argparse-driven; --DR and --save_root are REQUIRED.
python Lab4/Trainer.py --DR <dataset_root> --save_root <out_dir> [--fast_train] [--kl_anneal_type Cyclical|Monotonic]
python Lab4/Tester.py  --DR <dataset_root> --save_root <out_dir> --ckpt_path <file.ckpt> --test [--make_gif]

# Lab5 — Reinforcement learning. Each file is a standalone agent.
python Lab5/dqn.py            # DQN on LunarLander-v2
python Lab5/ddpg.py           # DDPG on LunarLanderContinuous-v2
python Lab5/dqn_breakout.py   # DQN on Atari Breakout (uses atari_wrappers.py)
#   add --test_only [--render] to evaluate a saved model instead of training

# Lab6 — Conditional DDPM (diffusers). Edit hardcoded paths first.
python Lab6/main.py --data_path <root> --model_path <ckpt> [--epoch N --batch_size B]
```

There is no "run a single test" — verification in each lab is the script's own train/eval loop printing accuracy or reward.

## Shared conventions

Repeated patterns across labs (most prominent in Lab2/Lab3) — match these when editing:

- **`class cfg:`** holds hyperparameters (`seed`, `batch_size`, `epochs`, `T0`, `lr`, `min_lr`) as class attributes.
- **`setSeed(seed=cfg.seed)`** seeds `random`/`numpy`/`torch` and sets `cudnn.deterministic = True`; called at module load. Seed is usually `1002` (Lab5 uses its own).
- **Optimizer/scheduler:** `AdamW` + `CosineAnnealingWarmRestarts` is the default pairing in Lab2/Lab3; Lab4/Lab6 use plain `Adam`.
- **Checkpoints:** Lab2/Lab3 save `state_dict` as `*.pkl`; Lab4 saves `.ckpt` dicts (includes `lr`, optimizer state) under `--save_root`; Lab5/Lab6 use `.pth`/`.pkl`.
- **Lab4 argparse contract:** `Trainer.py` and `Tester.py` share a nearly identical argument set (`--F_dim`, `--L_dim`, `--N_dim`, `--D_out_dim`, teacher-forcing `--tfr*`, KL-annealing `--kl_anneal_*`). When changing a model dimension, update **both** files or train/test will mismatch.

## Lab-specific structure notes

- **Lab1** is the only pure-NumPy lab: a `Layer` class (forward/backward/update) composed inside a `Network` class. No PyTorch.
- **Lab3** keeps each ResNet depth in its own file (`ResNet18.py`, `ResNet50.py`, `ResNet152.py`) — `BasicBlock` for 18, `Bottleneck` for 50/152. `test.py` adds test-time augmentation via the `ttach` library.
- **Lab4** model components live in `modules/` (`RGB_Encoder`, `Label_Encoder`, `Gaussian_Predictor`, `Decoder_Fusion`, `Generator`); the `VAE_Model` in `Trainer.py` wires them together.
- **Lab6** subclasses `diffusers.UNet2DModel` (`MyConditionedUNet`) to inject a multi-label embedding added to the time embedding, with a `DDPMScheduler` using the `squaredcos_cap_v2` (cosine) beta schedule.

## Dependencies

No manifest exists; install as needed. Beyond `torch`/`torchvision`/`numpy`/`matplotlib`/`scikit-learn`/`pandas`/`Pillow`/`tqdm`, the less-obvious ones are:

- `cosine_annealing_warmup` (Lab2/Lab3) — the `CosineAnnealingWarmupRestarts` class (distinct from torch's built-in `CosineAnnealingWarmRestarts`, which is also used)
- `albumentations`, `ttach`, `opencv-python` (Lab3)
- `gym` + Atari ROMs, `atari_wrappers` (Lab5)
- `diffusers`, `imageio` (Lab4/Lab6)
