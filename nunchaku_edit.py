# nunchaku_edit.py
# Drive "Nunchaku Qwen Edit Lightning 2509 - One Image Edit.json" (API format)
# Requires: comfy_client.ComfyClient (HTTP+WS) and ComfyUI running on 127.0.0.1:8188

import argparse, json
from typing import Dict, Any
from comfy_client import ComfyClient

# Known node IDs from your JSON:
ID_LOADIMAGE = "109"
ID_PROMPT_POS = "113"
ID_PROMPT_NEG = "114"
ID_KSAMPLER  = "3"
ID_SAVEIMG   = "79"

def load_workflow(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _set(d: Dict[str, Any], path: list, value) -> bool:
    cur = d
    try:
        for k in path[:-1]:
            cur = cur[k]
        cur[path[-1]] = value
        return True
    except Exception:
        return False

def set_input_image_api(wf: Dict[str, Any], image_name: str) -> bool:
    # Prefer exact ID -> inputs.image
    if ID_LOADIMAGE in wf and _set(wf, [ID_LOADIMAGE, "inputs", "image"], image_name):
        return True
    # Fallback: search by class_type
    for nid, node in wf.items():
        if isinstance(node, dict) and node.get("class_type") == "LoadImage":
            node.setdefault("inputs", {})["image"] = image_name
            return True
    return False

def set_qwen_prompts_api(wf: Dict[str, Any], positive: str=None, negative: str=None) -> int:
    count = 0
    if positive is not None and ID_PROMPT_POS in wf and _set(wf, [ID_PROMPT_POS, "inputs", "prompt"], positive):
        count += 1
    if negative is not None and ID_PROMPT_NEG in wf and _set(wf, [ID_PROMPT_NEG, "inputs", "prompt"], negative):
        count += 1
    # Fallbacks by class_type if ids missing
    if count < (1 if positive else 0) + (1 if negative else 0):
        pos_set = False
        neg_set = False
        for nid, node in wf.items():
            if isinstance(node, dict) and node.get("class_type") == "TextEncodeQwenImageEditPlus":
                if positive is not None and not pos_set:
                    node.setdefault("inputs", {})["prompt"] = positive
                    pos_set = True; count += 1
                elif negative is not None and not neg_set:
                    node.setdefault("inputs", {})["prompt"] = negative
                    neg_set = True; count += 1
    return count

def set_seed_api(wf: Dict[str, Any], seed: int) -> bool:
    if ID_KSAMPLER in wf and _set(wf, [ID_KSAMPLER, "inputs", "seed"], seed):
        return True
    for nid, node in wf.items():
        if isinstance(node, dict) and node.get("class_type") == "KSampler":
            node.setdefault("inputs", {})["seed"] = seed
            return True
    return False

def set_prefix_api(wf: Dict[str, Any], prefix: str) -> bool:
    if ID_SAVEIMG in wf and _set(wf, [ID_SAVEIMG, "inputs", "filename_prefix"], prefix):
        return True
    for nid, node in wf.items():
        if isinstance(node, dict) and node.get("class_type") == "SaveImage":
            node.setdefault("inputs", {})["filename_prefix"] = prefix
            return True
    return False

def main():
    ap = argparse.ArgumentParser(description="Drive Nunchaku Qwen Edit Lightning (API JSON)")
    ap.add_argument("--workflow", default="Nunchaku Qwen Edit Lightning 2509 - One Image Edit.json",
                    help="Path to API-format workflow JSON")
    ap.add_argument("--input_image", default="ref_base.png",
                    help="Path relative to ComfyUI/input/, e.g. 'ref_base.png' or 'refs/ref_base.png'")
    ap.add_argument("--positive", required=True, help="Edit instruction")
    ap.add_argument("--negative", default="", help="Negative instruction")
    ap.add_argument("--prefix", default="nunchaku_edit", help="SaveImage filename prefix")
    ap.add_argument("--seed", type=int, default=None, help="Optional fixed seed")
    ap.add_argument("--dump_history", action="store_true")
    args = ap.parse_args()

    client = ComfyClient()
    print("[*] Pinging ComfyUI ...", "OK" if client.ping() else "FAILED")
    if not client.ping():
        raise SystemExit("ComfyUI not reachable at http://127.0.0.1:8188")

    wf = load_workflow(args.workflow)

    # Patch fields
    if not set_input_image_api(wf, args.input_image):
        print("[!] Warning: could not set LoadImage -> inputs.image")
    set_qwen_prompts_api(wf, positive=args.positive, negative=args.negative or "")
    if not set_prefix_api(wf, args.prefix):
        print("[!] Warning: could not set SaveImage -> inputs.filename_prefix")
    if args.seed is not None and not set_seed_api(wf, args.seed):
        print("[!] Warning: could not set KSampler -> inputs.seed")

    # Submit → wait → download
    prompt_id = client.submit_prompt(wf)
    print(f"[*] Submitted. prompt_id={prompt_id}")
    history = client.wait_for_completion(prompt_id)
    if args.dump_history:
        try:
            with open("last_history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except Exception:
            pass
    print("[*] Complete. Downloading outputs...")
    for p in client.download_outputs(history):
        print("  ->", p)

if __name__ == "__main__":
    main()
