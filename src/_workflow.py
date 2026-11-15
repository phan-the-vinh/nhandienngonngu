"""Safe train/evaluate routing, with explicit unsupported compatibility states."""
import argparse
import importlib
import json
from src import ROOT

ROUTES={
    "train":{"digits":"training.digits.train_digits","dynamic":"training.dynamic.train_dynamic"},
    "evaluate":{"digits":"training.digits.evaluate_digits","dynamic":"training.dynamic.evaluate_dynamic"},
}

def main(kind,argv=None):
    parser=argparse.ArgumentParser(description=f"{kind} facade; existing implementations retain their confirmation gates")
    parser.add_argument("--module",choices=("letters","digits","dynamic"),required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only",action="store_true",help="Inspect routing only; no config/dataset execution")
    mode.add_argument("--execute",action="store_true",help="USER ONLY: delegate to existing implementation and its confirmation")
    args,forwarded=parser.parse_known_args(argv)
    route=ROUTES[kind].get(args.module)
    blocker=None
    if route is None:
        blocker="Letters training/evaluation is not standardized; no production-compatible entrypoint selected"
    if kind=="evaluate" and args.module=="digits":
        blocker="Existing evaluator supports only two-step scaler/MLP; digits_final has a three-step orientation pipeline"
    info={"module":args.module,"operation":kind,"implementation":route,"blocker":blocker,
          "execution_performed":False,"scope":"ROUTING_ONLY_NOT_DATA_OR_EVALUATION_PREFLIGHT"}
    if args.check_only:
        if forwarded:
            parser.error("--check-only inspects default routing only; use underlying tool for its detailed preflight")
        print(json.dumps(info))
        return 2 if blocker else 0
    if blocker:
        print(json.dumps(info))
        return 2
    if kind=="evaluate" and args.module=="dynamic" and not any(
            value=="--experiment-dir" or value.startswith("--experiment-dir=") for value in forwarded):
        print("ERROR: specify --experiment-dir explicitly; the existing default already has metrics.json")
        return 2
    implementation=importlib.import_module(route)
    if args.module=="dynamic":
        # Existing dynamic programs require their explicit operation switch.
        forwarded=[("--train" if kind=="train" else "--evaluate"),*forwarded]
    return implementation.main(forwarded)
