# Stub: will be used for Clean-label Backdoor / Feature Collision attacks.
# Expected zip content:
# - model.py with build_model(num_classes)->nn.Module
# - weights.pth with state_dict
#
# You can implement:
# - unzip to temp folder in run_dir
# - dynamic import model.py
# - build model, load weights
# - validate forward/backward