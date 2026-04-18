import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score, precision_recall_curve, auc

class ECGTrainer:
    def __init__(self, model, train_loader, val_loader, device, save_dir, criterion=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 允许外部传入自定义损失函数 (如 FocalLoss)
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()
            
    def fit(self, epochs=100, lr=1e-4, weight_decay=1e-4, patience=10):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        history = {'train_loss': [], 'val_f1':[], 'val_auprc':[]}
        best_f1 = 0.0
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            val_metrics = self.evaluate()

            history['train_loss'].append(train_loss / len(self.train_loader))
            history['val_f1'].append(val_metrics['f1'])
            history['val_auprc'].append(val_metrics['auprc'])

            scheduler.step(val_metrics['f1'])

            print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {history['train_loss'][-1]:.4f} - Val F1: {val_metrics['f1']:.4f} - Val AUPRC: {val_metrics['auprc']:.4f}")

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break

        # 训练结束后重新加载最优权重，确保后续 evaluate() 使用的是最佳模型
        best_path = os.path.join(self.save_dir, 'best_model.pth')
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, weights_only=True))

        return history
    
    def evaluate(self, loader=None):
        self.model.eval()
        if loader is None: loader = self.val_loader
        
        all_preds, all_probs, all_targets = [], [],[]
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(y_batch.numpy())
                
        f1 = f1_score(all_targets, all_preds, average='macro')
        precision, recall, _ = precision_recall_curve(
            pd.get_dummies(all_targets).values.ravel(), 
            np.array(all_probs).ravel()
        )
        auprc = auc(recall, precision)
        
        return {'f1': float(f1), 'auprc': float(auprc), 'preds': all_preds, 'targets': all_targets}