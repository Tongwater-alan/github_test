import inspect
import art

targets = [
    ("HiddenTriggerBackdoor", "art.attacks.poisoning"),
    ("PoisoningAttackCleanLabelBackdoor", "art.attacks.poisoning"),
    ("FeatureCollisionAttack", "art.attacks.poisoning"),
    ("SpectralSignatureDefense", "art.defences.detector.poison"),
    ("ActivationDefence", "art.defences.detector.poison"),
    ("KerasClassifier", "art.estimators.classification"),
    ("TensorFlowV2Classifier", "art.estimators.classification"),
]

def try_import(name, mod_prefix):
    try:
        m = __import__(mod_prefix, fromlist=[name])
        obj = getattr(m, name, None)
        return obj
    except Exception as e:
        return e

for name, mod in targets:
    obj = try_import(name, mod)
    print("="*80)
    print(f"{mod}.{name}")
    if isinstance(obj, Exception):
        print("IMPORT ERROR:", repr(obj))
        continue
    try:
        print("FILE:", inspect.getsourcefile(obj))
    except Exception as e:
        print("FILE ERROR:", repr(e))
    try:
        sig = inspect.signature(obj.__init__)
        print("INIT SIGNATURE:", sig)
    except Exception as e:
        print("SIG ERROR:", repr(e))
    # show key methods
    for meth in ["poison", "generate", "detect_poison", "evaluate_defence", "fit", "predict", "get_activations"]:
        if hasattr(obj, meth):
            try:
                print(f"METHOD {meth}:", inspect.signature(getattr(obj, meth)))
            except Exception:
                print(f"METHOD {meth}: (signature unavailable)")