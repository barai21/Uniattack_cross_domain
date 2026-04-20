import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. Configuration & Paths
# ==========================================
class Config:
    DATA_ROOT = "/home/nayana/Data/UniAttackData_P/UniAttackData_P"
    
    # Dictionary to hold paths for both protocols
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

    # Hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 50
    LR = 1e-4
    IMG_SIZE = 224  # ViT-B/16 natively uses 224x224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Dataset Definition
# ==========================================
class UniAttackDataset(Dataset):
    def __init__(self, txt_path, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        
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
                binary_label = 1 if detailed_label == 0 else 0
                self.samples.append((full_path, binary_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ==========================================
# 3. Metrics & ROC Calculation
# ==========================================
def calculate_metrics(labels, scores, threshold=None):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    
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
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
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
# 4. Evaluation & Feature Extraction
# ==========================================
def evaluate(model, dataloader, criterion, threshold=None):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    acc, roc_auc, eer, apcer, bpcer, acer, calc_thresh = calculate_metrics(all_labels, all_scores, threshold)
    
    return epoch_loss, acc, roc_auc, eer, apcer, bpcer, acer, calc_thresh, all_labels, all_scores

def extract_features(model, dataloader):
    """Extracts features from the frozen ViT backbone."""
    model.eval()
    features = []
    labels_list = []
    
    # Temporarily replace the final classification head in ViT with Identity
    original_fc = model.heads.head
    model.heads.head = nn.Identity()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Features", leave=False):
            images = images.to(Config.DEVICE)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels_list.extend(labels.numpy())
            
    model.heads.head = original_fc # Restore the classification layer
    return np.vstack(features), np.array(labels_list)

# ==========================================
# 5. Training & Testing for a Single Protocol
# ==========================================
def run_protocol(protocol_name):
    print(f"\n{'='*50}")
    print(f"--- Running Face Attack Detection on {protocol_name} ---")
    print(f"{'='*50}")
    
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    paths = Config.PATHS[protocol_name]

    train_dataset = UniAttackDataset(paths['train'], Config.DATA_ROOT, transform=train_transform)
    dev_dataset   = UniAttackDataset(paths['dev'], Config.DATA_ROOT, transform=test_transform)
    test_dataset  = UniAttackDataset(paths['test'], Config.DATA_ROOT, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    dev_loader   = DataLoader(dev_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model Setup (ViT-B/16)
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    
    # FREEZE ALL LAYERS
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final fully connected layer in ViT (model.heads.head)
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, 2)
    model = model.to(Config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    # Pass ONLY the trainable parameters (the new classification layer) to the optimizer
    optimizer = optim.Adam(model.heads.head.parameters(), lr=Config.LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_dev_acer = float('inf')
    best_dev_threshold = 0.5
    model_save_path = f"best_vit_b_16_{protocol_name}.pth"

    # ------------------------------------------
    # A. Train nn.Linear Classifier
    # ------------------------------------------
    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
        for images, labels in pbar:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss = running_loss / len(train_dataset)
        scheduler.step()
        
        dev_loss, dev_acc, dev_auc, dev_eer, dev_apcer, dev_bpcer, dev_acer, dev_thresh, _, _ = evaluate(model, dev_loader, criterion)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f}")
        print(f"Dev Metrics -> ACC: {dev_acc*100:.2f}% | AUC: {dev_auc*100:.2f}% | EER: {dev_eer*100:.2f}%")
        
        if dev_acer < best_dev_acer:
            best_dev_acer = dev_acer
            best_dev_threshold = dev_thresh
            torch.save(model.state_dict(), model_save_path)

    # Final Testing for nn.Linear
    model.load_state_dict(torch.load(model_save_path))
    _, test_acc_lin, test_auc_lin, test_eer_lin, test_apcer_lin, test_bpcer_lin, test_acer_lin, _, test_labels_lin, test_scores_lin = evaluate(model, test_loader, criterion, threshold=best_dev_threshold)
    
    save_roc_curve(test_labels_lin, test_scores_lin, "Linear", protocol_name)

    # ------------------------------------------
    # B. Train SVM & RF Classifiers
    # ------------------------------------------
    print(f"\n--- Extracting Features for SVM & RF ({protocol_name}) ---")
    # Use test_transform for clean feature extraction without random augmentations
    train_dataset_feat = UniAttackDataset(paths['train'], Config.DATA_ROOT, transform=test_transform)
    train_loader_feat = DataLoader(train_dataset_feat, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    X_train, y_train = extract_features(model, train_loader_feat)
    X_dev, y_dev     = extract_features(model, dev_loader)
    X_test, y_test   = extract_features(model, test_loader)

    # --- SVM ---
    print(f"--- Training SVM ({protocol_name}) ---")
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    dev_scores_svm = svm.predict_proba(X_dev)[:, 1]
    _, _, _, _, _, _, dev_thresh_svm = calculate_metrics(y_dev, dev_scores_svm)
    
    test_scores_svm = svm.predict_proba(X_test)[:, 1]
    test_acc_svm, test_auc_svm, test_eer_svm, test_apcer_svm, test_bpcer_svm, test_acer_svm, _ = calculate_metrics(y_test, test_scores_svm, threshold=dev_thresh_svm)
    save_roc_curve(y_test, test_scores_svm, "SVM", protocol_name)

    # --- Random Forest ---
    print(f"--- Training Random Forest ({protocol_name}) ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    dev_scores_rf = rf.predict_proba(X_dev)[:, 1]
    _, _, _, _, _, _, dev_thresh_rf = calculate_metrics(y_dev, dev_scores_rf)
    
    test_scores_rf = rf.predict_proba(X_test)[:, 1]
    test_acc_rf, test_auc_rf, test_eer_rf, test_apcer_rf, test_bpcer_rf, test_acer_rf, _ = calculate_metrics(y_test, test_scores_rf, threshold=dev_thresh_rf)
    save_roc_curve(y_test, test_scores_rf, "RF", protocol_name)

    # Return nested dictionary for all classifiers
    return {
        'Linear': {
            'acc': test_acc_lin * 100, 'auc': test_auc_lin * 100, 'apcer': test_apcer_lin * 100,
            'bpcer': test_bpcer_lin * 100, 'eer': test_eer_lin * 100, 'acer': test_acer_lin * 100
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
# 6. Main Execution & Final Tables
# ==========================================
def main():
    results = {}
    
    for protocol in ['P2.1', 'P2.2']:
        results[protocol] = run_protocol(protocol)
        
    metrics = ['acc', 'auc', 'apcer', 'bpcer', 'eer', 'acer']
    classifiers = ['Linear', 'SVM', 'RF']
    
    for clf in classifiers:
        summary = {}
        for m in metrics:
            vals = [results['P2.1'][clf][m], results['P2.2'][clf][m]]
            summary[m] = {
                'mean': np.mean(vals),
                'std': np.std(vals)
            }
            
        print("\n\n" + "="*80)
        print(f"{f'FINAL RESULTS: PROTOCOL 2 - Classifier: {clf} (ViT-B/16)':^80}")
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