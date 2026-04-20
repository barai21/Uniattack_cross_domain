import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import HuggingFace CLIP
from transformers import CLIPProcessor, CLIPModel

# ==========================================
# 1. Configuration & Paths
# ==========================================
class Config:
    DATA_ROOT = "/home/nayana/Data/UniAttackData_P/UniAttackData_P"
    
    PATHS = {
        'P2.1': {
            'train': "/home/nayana/Data/UniAttackData@p2.1_image_train_cropped.txt",
            'dev':   "/home/nayana/Data/UniAttackData@p2.1_image_dev_cropped.txt",
            'test':  "/home/nayana/Data/UniAttackData@p2.1_image_test_cropped.txt"
        },
        'P2.2': {
            'train': "/home/nayana/Data/UniAttackData@p2.2_image_train_cropped.txt",
            'dev':   "/home/nayana/Data/UniAttackData@p2.2_image_dev_cropped.txt",
            'test':  "/home/nayana/Data/UniAttackData@p2.2_image_test_cropped.txt"
        }
    }

    BATCH_SIZE = 128
    EPOCHS = 50
    LR = 1e-5  # Lower learning rate for fine-tuning the ViT backbone
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Data Augmentation & Dataset
# ==========================================
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
    def __init__(self, txt_path, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.samples =[]
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                img_rel_path = parts[0]
                detailed_label = int(parts[1])
                
                if img_rel_path.startswith("UniAttackData_P/"):
                    img_rel_path = img_rel_path.replace("UniAttackData_P/", "", 1)
                
                full_path = os.path.join(self.data_root, img_rel_path)
                # 1 = Real Face, 0 = Spoof Face
                binary_label = 1 if detailed_label == 0 else 0
                self.samples.append((full_path, binary_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ==========================================
# 3. Recos Similarity & Custom Loss Function
# ==========================================
def recos_sim(u, v, dim=-1, eps=1e-8):
    """
    Rearrangement-based cosine similarity (recos) — arXiv:2602.05266.
    """
    dot = (u * v).sum(dim=dim)
    u_sorted = u.sort(dim=dim, descending=True).values
    v_sorted = v.sort(dim=dim, descending=True).values
    rearranged_max = (u_sorted * v_sorted).sum(dim=dim)
    return dot / (rearranged_max.abs() + eps)

class BCERecosLoss(nn.Module):
    def __init__(self, logit_scale, lambda_cos=1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.lambda_cos = lambda_cos
        self.logit_scale = logit_scale

    def forward(self, image_features, text_features, labels):
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 1. Binary Cross Entropy Loss using Recos
        # Broadcast to compute pairwise recos: (B, 1, D) and (1, 2, D) -> (B, 2)
        u = image_features.unsqueeze(1)
        v = text_features.unsqueeze(0)
        recos_matrix = recos_sim(u, v, dim=-1)
        
        logits = self.logit_scale.exp() * recos_matrix
        probs = torch.softmax(logits, dim=1)[:, 1] # Probability of class 1 (Real Face)
        bce_loss = self.bce(probs, labels.float())
        
        # 2. Recos Similarity Loss (Maximize similarity with the true class text prompt)
        target_text_features = text_features[labels]
        recos_val = recos_sim(image_features, target_text_features, dim=-1)
        recos_loss = (1.0 - recos_val).mean()
        
        return bce_loss + self.lambda_cos * recos_loss

# ==========================================
# 4. Metrics & ROC Calculation
# ==========================================
def calculate_metrics(labels, scores, threshold=None):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    # Calculate EER Threshold
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    
    # Use provided threshold (e.g., from Dev set) or fallback to current EER threshold
    eval_thresh = threshold if threshold is not None else eer_threshold
    preds = (np.array(scores) >= eval_thresh).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    
    apcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    bpcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    acer = (apcer + bpcer) / 2.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    return acc, roc_auc, eer, apcer, bpcer, acer, eval_thresh

def save_roc_curve(y_true, y_scores, classifier_name, protocol_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1],[0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {classifier_name} ({protocol_name})')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'ROC_{classifier_name}_{protocol_name}.png')
    plt.close()

# ==========================================
# 5. Safe Feature Extraction Helpers
# ==========================================
def get_text_features_safe(model, inputs):
    out = model.get_text_features(**inputs)
    if isinstance(out, torch.Tensor): return out
    if hasattr(out, 'text_embeds') and out.text_embeds is not None: return out.text_embeds
    text_outputs = model.text_model(**inputs)
    return model.text_projection(text_outputs[1])

def get_image_features_safe(model, images):
    out = model.get_image_features(pixel_values=images)
    if isinstance(out, torch.Tensor): return out
    if hasattr(out, 'image_embeds') and out.image_embeds is not None: return out.image_embeds
    vision_outputs = model.vision_model(pixel_values=images)
    return model.visual_projection(vision_outputs[1])

# ==========================================
# 6. Evaluation & Feature Extraction
# ==========================================
def evaluate_clip(model, dataloader, text_features, criterion=None, threshold=None):
    """Evaluates the CLIP model using Recos Similarity with text prompts."""
    model.eval()
    running_loss = 0.0
    all_labels =[]
    all_scores =[]
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            image_features = get_image_features_safe(model, images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            if criterion is not None:
                loss = criterion(image_features, text_features, labels)
                running_loss += loss.item() * images.size(0)
            
            logit_scale = model.logit_scale.exp()
            
            # Use Recos for logits instead of standard dot product
            u = image_features.unsqueeze(1) # (B, 1, D)
            v = text_features.unsqueeze(0)  # (1, 2, D)
            recos_matrix = recos_sim(u, v, dim=-1) # (B, 2)
            
            logits = logit_scale * recos_matrix
            probs = logits.softmax(dim=1)[:, 1]
            
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset) if criterion else 0.0
    acc, roc_auc, eer, apcer, bpcer, acer, calc_thresh = calculate_metrics(all_labels, all_scores, threshold)
    return epoch_loss, acc, roc_auc, eer, apcer, bpcer, acer, calc_thresh, all_labels, all_scores

def extract_features(model, dataloader):
    """Extracts normalized features from the CLIP backbone."""
    model.eval()
    features =[]
    labels_list =[]
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Features", leave=False):
            images = images.to(Config.DEVICE)
            image_features = get_image_features_safe(model, images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            features.append(image_features.cpu().numpy())
            labels_list.extend(labels.numpy())
            
    return np.vstack(features), np.array(labels_list)

# ==========================================
# 7. Training & Testing for a Single Protocol
# ==========================================
def run_protocol(protocol_name):
    print(f"\n{'='*50}")
    print(f"--- Running Face Attack Detection on {protocol_name} ---")
    print(f"{'='*50}")
    
    paths = Config.PATHS[protocol_name]

    train_dataset = UniAttackDataset(paths['train'], Config.DATA_ROOT, transform=get_transforms(train=True))
    dev_dataset   = UniAttackDataset(paths['dev'], Config.DATA_ROOT, transform=get_transforms(train=False))
    test_dataset  = UniAttackDataset(paths['test'], Config.DATA_ROOT, transform=get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    dev_loader   = DataLoader(dev_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Load CLIP Model & Processor
    print("Loading CLIP Model (ViT-B/16)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(Config.DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    # Freeze Text Encoder, Unfreeze Vision Encoder for Fine-Tuning
    for param in model.text_model.parameters(): param.requires_grad = False
    for param in model.text_projection.parameters(): param.requires_grad = False
    for param in model.vision_model.parameters(): param.requires_grad = True
    for param in model.visual_projection.parameters(): param.requires_grad = True

    # ------------------------------------------
    # A. Precompute Text Embeddings
    # ------------------------------------------
    templates =[
        "This photo contains {}.", "There is a {} in this photo.", "{} is in this photo.",
        "A photo of a {}.", "This is an example of a {}.", "This is how a {} looks like.",
        "This is an image of {}.", "The picture is a {}."
    ]
    classes =["spoof face", "real face"] 
    
    text_prompts =[t.format(cls) for cls in classes for t in templates]
    inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(Config.DEVICE)
    
    with torch.no_grad():
        text_features = get_text_features_safe(model, inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(2, 8, -1).mean(dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Zero-Shot Baseline Evaluation
    print(f"\n--- Running Zero-Shot Baseline ({protocol_name}) ---")
    _, _, _, _, _, _, _, dev_thresh_zs, _, _ = evaluate_clip(model, dev_loader, text_features)
    _, test_acc_zs, test_auc_zs, test_eer_zs, test_apcer_zs, test_bpcer_zs, test_acer_zs, _, test_labels_zs, test_scores_zs = evaluate_clip(model, test_loader, text_features, threshold=dev_thresh_zs)
    save_roc_curve(test_labels_zs, test_scores_zs, "ZeroShot", protocol_name)

    # ------------------------------------------
    # B. Fine-Tune CLIP Vision Encoder
    # ------------------------------------------
    print(f"\n--- Fine-Tuning CLIP Vision Encoder ({protocol_name}) ---")
    criterion = BCERecosLoss(logit_scale=model.logit_scale, lambda_cos=1.0)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_dev_acer = float('inf')
    best_dev_threshold = 0.5
    model_save_path = f"best_clip_finetuned_{protocol_name}.pth"

    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
        for images, labels in pbar:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            image_features = get_image_features_safe(model, images)
            
            loss = criterion(image_features, text_features, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss = running_loss / len(train_dataset)
        scheduler.step()
        
        # Evaluate on Dev set to get EER threshold
        dev_loss, dev_acc, dev_auc, dev_eer, dev_apcer, dev_bpcer, dev_acer, dev_thresh, _, _ = evaluate_clip(model, dev_loader, text_features, criterion)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f}")
        print(f"Dev Metrics -> ACC: {dev_acc*100:.2f}% | AUC: {dev_auc*100:.2f}% | EER: {dev_eer*100:.2f}%")
        
        if dev_acer < best_dev_acer:
            best_dev_acer = dev_acer
            best_dev_threshold = dev_thresh
            torch.save(model.state_dict(), model_save_path)

    # Final Testing for Fine-Tuned CLIP (Using Dev EER Threshold)
    model.load_state_dict(torch.load(model_save_path))
    _, test_acc_ft, test_auc_ft, test_eer_ft, test_apcer_ft, test_bpcer_ft, test_acer_ft, _, test_labels_ft, test_scores_ft = evaluate_clip(model, test_loader, text_features, criterion, threshold=best_dev_threshold)
    save_roc_curve(test_labels_ft, test_scores_ft, "FineTunedCLIP", protocol_name)

    # ------------------------------------------
    # C. Train SVM & RF Classifiers on Fine-Tuned Features
    # ------------------------------------------
    print(f"\n--- Extracting Features for SVM & RF ({protocol_name}) ---")
    train_dataset_feat = UniAttackDataset(paths['train'], Config.DATA_ROOT, transform=get_transforms(train=False))
    train_loader_feat = DataLoader(train_dataset_feat, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    X_train, y_train = extract_features(model, train_loader_feat)
    X_dev, y_dev     = extract_features(model, dev_loader)
    X_test, y_test   = extract_features(model, test_loader)

    # --- SVM ---
    print(f"--- Training SVM ({protocol_name}) ---")
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    dev_scores_svm = svm.predict_proba(X_dev)[:, 1]
    _, _, _, _, _, _, dev_thresh_svm = calculate_metrics(y_dev, dev_scores_svm) # Get EER threshold
    
    test_scores_svm = svm.predict_proba(X_test)[:, 1]
    test_acc_svm, test_auc_svm, test_eer_svm, test_apcer_svm, test_bpcer_svm, test_acer_svm, _ = calculate_metrics(y_test, test_scores_svm, threshold=dev_thresh_svm)
    save_roc_curve(y_test, test_scores_svm, "SVM", protocol_name)

    # --- Random Forest ---
    print(f"--- Training Random Forest ({protocol_name}) ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    dev_scores_rf = rf.predict_proba(X_dev)[:, 1]
    _, _, _, _, _, _, dev_thresh_rf = calculate_metrics(y_dev, dev_scores_rf) # Get EER threshold
    
    test_scores_rf = rf.predict_proba(X_test)[:, 1]
    test_acc_rf, test_auc_rf, test_eer_rf, test_apcer_rf, test_bpcer_rf, test_acer_rf, _ = calculate_metrics(y_test, test_scores_rf, threshold=dev_thresh_rf)
    save_roc_curve(y_test, test_scores_rf, "RF", protocol_name)

    return {
        'ZeroShot': {
            'acc': test_acc_zs * 100, 'auc': test_auc_zs * 100, 'apcer': test_apcer_zs * 100,
            'bpcer': test_bpcer_zs * 100, 'eer': test_eer_zs * 100, 'acer': test_acer_zs * 100
        },
        'FineTunedCLIP': {
            'acc': test_acc_ft * 100, 'auc': test_auc_ft * 100, 'apcer': test_apcer_ft * 100,
            'bpcer': test_bpcer_ft * 100, 'eer': test_eer_ft * 100, 'acer': test_acer_ft * 100
        },
        'SVM': {
            'acc': test_acc_svm * 100, 'auc': test_auc_svm * 100, 'apcer': test_apcer_svm * 100,
            'bpcer': test_bpcer_svm * 100, 'eer': test_eer_svm * 100, 'acer': test_acer_svm * 100
        },
        'RF': {
            'acc': test_acc_rf * 100, 'auc': test_auc_rf * 100, 'apcer': test_apcer_rf * 100,
            'bpcer': test_bpcer_rf * 100, 'eer': test_eer_rf * 100, 'acer': test_acer_rf * 100
        }
    }

# ==========================================
# 8. Main Execution & Final Tables
# ==========================================
def main():
    results = {}
    
    for protocol in['P2.1', 'P2.2']:
        results[protocol] = run_protocol(protocol)
        
    metrics =['acc', 'auc', 'apcer', 'bpcer', 'eer', 'acer']
    classifiers =['ZeroShot', 'FineTunedCLIP', 'SVM', 'RF']
    
    for clf in classifiers:
        summary = {}
        for m in metrics:
            vals = [results['P2.1'][clf][m], results['P2.2'][clf][m]]
            summary[m] = {
                'mean': np.mean(vals),
                'std': np.std(vals)
            }
            
        print("\n\n" + "="*80)
        print(f"{f'FINAL RESULTS: PROTOCOL 2 - Classifier: {clf}':^80}")
        print("="*80)
        print(f"{'Metric':<15} | {'Protocol 2.1 (%)':<18} | {'Protocol 2.2 (%)':<18} | {'Protocol 2 (Mean ± Std) (%)':<25}")
        print("-" * 80)
        
        for m in metrics:
            p21_val = f"{results['P2.1'][clf][m]:.2f}"
            p22_val = f"{results['P2.2'][clf][m]:.2f}"
            mean_std_val = f"{summary[m]['mean']:.2f} ± {summary[m]['std']:.2f}"
            
            print(f"{m.upper():<15} | {p21_val:<18} | {p22_val:<18} | {mean_std_val:<25}")
        print("="*80)

if __name__ == '__main__':
    main()