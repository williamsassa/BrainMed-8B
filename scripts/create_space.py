#!/usr/bin/env python
"""
Create (or update) the Hugging Face Space that serves the model.

    HF_TOKEN=... python scripts/create_space.py --space_id BrainHealthAI/BrainMed-8B-demo

The Space is created **private** by default. A demo of a medical model is an outward-facing
artifact: make it public deliberately, once the disclaimer and the hardware are what you want
people to see, not as a side effect of a deploy script.
"""
import argparse
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space_id", required=True, help="e.g. BrainHealthAI/BrainMed-8B-demo")
    ap.add_argument("--model_id", default="BrainHealthAI/BrainMed-8B")
    ap.add_argument("--folder", default=os.path.join(REPO_ROOT, "space"))
    ap.add_argument("--hardware", default=None,
                    help="e.g. a10g-small, t4-small. Omit to keep the current/free CPU tier "
                         "(on which an 8B model is unusably slow)")
    ap.add_argument("--public", action="store_true", help="create it public instead of private")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("export HF_TOKEN first")

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    try:
        api.create_repo(args.space_id, repo_type="space", space_sdk="gradio",
                        private=not args.public, exist_ok=True)
    except Exception as e:                                          # noqa: BLE001
        if "402" in str(e) or "Payment Required" in str(e):
            owner = args.space_id.split("/")[0]
            raise SystemExit(
                f"Hugging Face refused to create the Space under '{owner}'.\n\n"
                f"  Gradio Spaces on the free CPU tier now require a Team or Enterprise plan\n"
                f"  for *organizations*. Personal accounts are unaffected.\n\n"
                f"  Options:\n"
                f"    1. create it under your personal account instead:\n"
                f"         --space_id {api.whoami()['name']}/{args.space_id.split('/')[-1]}\n"
                f"    2. subscribe the organization: https://huggingface.co/enterprise\n"
                f"    3. skip the Space entirely and serve the demo from a GPU box you already\n"
                f"       rent:  GRADIO_SHARE=1 MODEL_ID=<local-path> python space/app.py\n\n"
                f"  Note that whichever you pick, an 8B model still needs a GPU: the free CPU\n"
                f"  tier has ~16 GB of RAM and the weights alone are 16 GB.\n\n"
                f"  original error: {e}") from e
        raise
    print(f"[space] {args.space_id} (private={not args.public})")

    api.upload_folder(folder_path=args.folder, repo_id=args.space_id, repo_type="space",
                      commit_message="deploy BrainMed demo")
    print(f"[space] uploaded {args.folder}")

    # the model repo is private: the Space needs a token of its own to pull the weights
    api.add_space_secret(args.space_id, "HF_TOKEN", token,
                         description="read access to the private model repo")
    api.add_space_variable(args.space_id, "MODEL_ID", args.model_id)
    print(f"[space] set MODEL_ID={args.model_id} and HF_TOKEN secret")

    if args.hardware:
        api.request_space_hardware(args.space_id, args.hardware)
        print(f"[space] requested hardware: {args.hardware} (billed while running)")
    else:
        print("[space] no hardware requested - on the free CPU tier an 8B model will not be "
              "usable; set one with --hardware a10g-small")

    print(f"\nhttps://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
