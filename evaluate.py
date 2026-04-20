import os
import glob
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from model   import UniAttackDetection
from dataset import UniAttackDataset, get_transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from metrics import evaluate, evaluate_with_eer_threshold

torch.set_float32_matmul_precision('high')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       required=True)
    p.add_argument("--checkpoint",   default=None)
    p.add_argument("--split",        default="test", choices=["dev","test"])
    p.add_argument("--ckpt_dir",     default=None)
    p.add_argument("--protocol",     default=None, choices=["2.1","2.2"])
    p.add_argument("--ckpt_dir_p21", default=None, help="Checkpoint folder for Protocol 2.1")
    p.add_argument("--ckpt_dir_p22", default=None, help="Checkpoint folder for Protocol 2.2")
    p.add_argument("--split_multi",  default="test", choices=["dev","test"])
    p.add_argument("--ckpt_glob",    default="*.pth", help="Glob pattern inside checkpoint folder")
    p.add_argument("--save_csv",     default=None)
    return p.parse_args()

def build_model(cfg, device):
    return UniAttackDetection(
        clip_model_name       = cfg.get("clip_model", "ViT-B/16"),
        num_student_tokens    = cfg.get("num_student_tokens", 16),
        num_teacher_templates = cfg.get("num_teacher_templates", 6),
        lam                   = cfg.get("lambda_ufm", 1.0),
        device                = device,
    ).to(device)

def load_checkpoint(model, ckpt_path, device):
    ckpt     = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    epoch    = ckpt.get("epoch", "?")
    best_auc = ckpt.get("best_auc",  None)
    best_acer= ckpt.get("best_acer", None)
    dev_thr  = float(ckpt.get("dev_acer_threshold", ckpt.get("threshold", 0.5)))
    tag = f"epoch={epoch}"
    if best_auc  is not None: tag += f"  AUC={float(best_auc):.4f}%"
    if best_acer is not None: tag += f"  ACER={float(best_acer):.4f}%"
    return model, dev_thr, tag

def run_inference(model, txt_path, data_root, cfg, device):
    ds = UniAttackDataset(txt_path, data_root, get_transforms(False))
    loader = DataLoader(ds,
                        batch_size  = cfg.get("batch_size", 32),
                        shuffle     = False,
                        num_workers = cfg.get("num_workers", 8),
                        pin_memory  = True)
    all_labels, all_scores = [], []
    model.eval()
    
    # Use inference_mode and AMP for faster evaluation
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True):
        for imgs, labels in tqdm(loader, desc="  Inference", leave=False):
            logits, _ = model(imgs.to(device, non_blocking=True))
            probs = torch.softmax(logits, -1)[:, 1].cpu().numpy()
            all_scores.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_scores)

# ... [Keep eval_one, compute_stats, print helpers, save_csv exactly as they were] ...
# (Omitted print helpers for brevity, keep them identical to your original file)

def main():
    args   = parse_args()
    cfg    = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    model       = build_model(cfg, device)
    all_results = {}

    if args.checkpoint:
        proto = "2.1" if ("2.1" in args.checkpoint or "p21" in args.checkpoint.lower()) else "2.2"
        txt   = get_txt(cfg, proto, args.split)
        if txt is None:
            print(f"[ERROR] No txt for P{proto} in config."); return
        print(f"\n[INFO] Single checkpoint | Protocol {proto} | split={args.split}")
        r = eval_one(model, args.checkpoint, txt, cfg["data_root"], cfg, device)
        all_results[proto] = [r]

    elif args.ckpt_dir and args.protocol:
        results = run_protocol(args.protocol, args.ckpt_dir,
                               args.split_multi, model, cfg, device, args.ckpt_glob)
        all_results[args.protocol] = results

    else:
        # Linux-friendly relative paths instead of C:\Users\...
        base = os.environ.get("UNIATTACK_DATA_ROOT", "./")
        dirs = {
            "2.1": args.ckpt_dir_p21 or os.path.join(base, "checkpoints_J_p2.1"),
            "2.2": args.ckpt_dir_p22 or os.path.join(base, "checkpoints_J_p2.2"),
        }

        for proto, folder in dirs.items():
            if not os.path.isdir(folder):
                print(f"[WARN] Folder not found, skipping P{proto}: {folder}")
                continue
            results = run_protocol(proto, folder, args.split_multi,
                                   model, cfg, device, args.ckpt_glob)
            all_results[proto] = results

    # ... [Keep the rest of the printing logic identical] ...