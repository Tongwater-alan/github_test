# poison-tester-ui (MVP scaffold)

A Gradio-based UI to test poisoning/backdoor attacks and defenses using ART (adversarial-robustness-toolbox==1.20.1).

## What works in this scaffold
- NPZ dataset loader (supports two schemas)
- Image classification pipeline (train/val/test)
- TorchScript model loading + baseline evaluation
- Hidden Trigger Backdoor *fallback* attack:
  - poison a fraction of training samples by applying trigger + relabel to target
  - create triggered test set for ASR measurement
- Simple trainer (Quick/Normal modes)
- Run output folder under `./run/<timestamp>/`:
  - report.json, report.md
  - plots (metrics bar + placeholder ROC/reject plots)
  - artifacts (poison indices, trigger spec)

## What is stubbed (interfaces ready)
- Clean-label backdoor attack
- Feature collision attack
- Spectral Signature Defense
- Activation Defence

You can later implement these using ART 1.20.1 classes:
- art.attacks.poisoning.HiddenTriggerBackdoor
- art.attacks.poisoning.PoisoningAttackCleanLabelBackdoor
- art.attacks.poisoning.FeatureCollisionAttack
- art.defences.detector.poison.SpectralSignatureDefense
- art.defences.detector.poison.ActivationDefence

## Dataset NPZ schemas
Schema-1:
- x_train, y_train, x_val, y_val, x_test, y_test

Schema-2:
- x, y, train_idx, val_idx, test_idx

x can be NHWC or NCHW. y can be (N,) integer labels or one-hot (N,K).

## Run
```bash
pip install -r requirements.txt
python app.py
```

Open the Gradio link in your browser.