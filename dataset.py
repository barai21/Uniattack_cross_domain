import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

def get_transforms(train: bool = True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275,  0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275,  0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
        ])

class UniAttackDataset(Dataset):
    def __init__(self, txt_file: str, data_root: str, transform=None, label_col: int = 1, path_col: int = 0):
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        self.missing_warnings = 0  # Counter to prevent terminal flooding

        with open(txt_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if not lines:
            print(f"[ERROR] Text file is empty: {txt_file}")
            return

        # --- 1. Auto-Detect the correct path format ---
        first_rel = lines[0].split()[path_col].replace("\\", "/")
        
        # Option A: Direct join
        path_opt_a = os.path.join(data_root, first_rel)
        # Option B: Strip "UniAttackData_P/" prefix
        path_opt_b = os.path.join(data_root, first_rel.replace("UniAttackData/", "", 1))

        strip_prefix = False
        if os.path.exists(path_opt_a):
            strip_prefix = False
        elif os.path.exists(path_opt_b):
            strip_prefix = True
        else:
            print(f"\n[CRITICAL ERROR] Cannot find images on disk!")
            print(f"  I looked in Option A: {path_opt_a}")
            print(f"  I looked in Option B: {path_opt_b}")
            print(f"  Please check the 'data_root' in your config.yaml.\n")
            strip_prefix = True # Fallback

        # --- 2. Load all samples ---
        for line in lines:
            parts = line.split()
            rel_path = parts[path_col].replace("\\", "/")
            label    = int(parts[label_col])
            binary_label = 0 if label == 0 else 1

            if strip_prefix and rel_path.startswith("UniAttackData/"):
                rel_path = rel_path.replace("UniAttackData/", "", 1)
            
            if rel_path.startswith("/"):
                rel_path = rel_path[1:]

            abs_path = os.path.join(data_root, rel_path)
            self.samples.append((abs_path, binary_label))

        print(f"[Dataset] Loaded {len(self.samples)} samples from {txt_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            # Return a black image so training doesn't crash
            img = Image.new("RGB", (224, 224), (0, 0, 0))
            
            # Limit warnings to 5 so we don't flood the terminal
            if self.missing_warnings < 5:
                print(f"\n[Warning] Could not load image: {path}")
                self.missing_warnings += 1
                if self.missing_warnings == 5:
                    print("[Warning] Suppressing further missing file warnings to keep terminal clean...")
                    
        if self.transform:
            img = self.transform(img)
        return img, label

def build_dataloaders(train_txt: str, dev_txt: str, test_txt: str, data_root: str, batch_size: int = 32, num_workers: int = 4):
    train_ds = UniAttackDataset(train_txt, data_root, get_transforms(True))
    dev_ds   = UniAttackDataset(dev_txt,   data_root, get_transforms(False))
    test_ds  = UniAttackDataset(test_txt,  data_root, get_transforms(False))

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True if num_workers > 0 else False,
        "prefetch_factor": 4 if num_workers > 0 else None
    }

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    dev_loader   = DataLoader(dev_ds, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    
    return train_loader, dev_loader, test_loader