import torch
import torch.nn as nn
from torchmetrics.classification import AUROC
from tqdm import tqdm
import numpy as np 
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, early_patience=10, early_stop=True, cutmix=False, cutmix_prob=0.3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.metric = AUROC(task='multiclass', num_classes=4)
        self.current_lr = optimizer.param_groups[0]['lr']
        self.scheduler = scheduler
        # early stopping parameters
        self.early_patience = early_patience
        self.early_stop = early_stop
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        # cutmix parameters
        self.cutmix = cutmix
        self.cutmix_prob = cutmix_prob

    def train_epoch(self, epoch):
        self.model.train()
        accu_loss = 0.0
        running_avg_loss = 0.0
        

        with tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1} [Training..]') as pbar:
            self.metric.reset()
            for batch_idx, (images, labels)in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                if self.cutmix:
                    r = np.random.rand()
                    if r < self.cutmix_prob:
                        images, labels, shuffled_labels, lam = self.cutmix(images, labels)
                        outputs = self.model(images)
                        loss = self.loss_fn(outputs, labels) * lam + self.loss_fn(outputs, shuffled_labels) * (1-lam)
                 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_proba = F.softmax(outputs, dim=1)
                self.metric.update(pred_proba, labels)

                accu_loss += loss.item()
                running_avg_loss = accu_loss / (batch_idx + 1)

                pbar.update(1)
                if batch_idx % 20 == 0 or batch_idx == len(self.train_loader) - 1:
                    pbar.set_postfix(loss=running_avg_loss, auroc=self.metric.compute().item())

        if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
            self.current_lr = self.scheduler.get_last_lr()[0]

        return running_avg_loss, self.metric.compute().item()
    
    def val_epoch(self, epoch):
        self.model.eval()

        accu_loss = 0.0
        running_avg_loss = 0.0

        with tqdm(total=len(self.val_loader), desc=f'Epoch {epoch+1} [Validation..]') as pbar:
            self.metric.reset()
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                pred_proba = F.softmax(outputs, dim=1)
                self.metric.update(pred_proba, labels)

                accu_loss += loss.item()
                running_avg_loss = accu_loss / (batch_idx + 1)

                pbar.update(1)
                if batch_idx % 20 == 0 or batch_idx == len(self.val_loader) - 1:
                    pbar.set_postfix(loss=running_avg_loss, auroc=self.metric.compute().item())

        self.scheduler.step()
        self.current_lr = self.scheduler.get_last_lr()[0]
        return running_avg_loss, self.metric.compute().item()
    
    def fit(self, epochs):
        history = {'train_loss': [], 'train_auroc': [], 'val_loss': [], 'val_auroc': [], 'lr': []}

        for epoch in range(epochs):
            train_loss, train_auroc = self.train_epoch(epoch)
            val_loss, val_auroc = self.val_epoch(epoch)

            history['train_loss'].append(train_loss)
            history['train_auroc'].append(train_auroc)
            history['val_loss'].append(val_loss)
            history['val_auroc'].append(val_auroc)
            history['lr'].append(self.current_lr)

            if self.early_stop:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.early_patience:
                        print(f'Early stopping at epoch {epoch+1}')
                        break

        return history

    # random bbox function
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]

        # center coordinatees of the bbox
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # cuting ratio 
        cut_ratio = np.sqrt(1.0-lam)
        cut_w = np.array(W * cut_ratio).astype(np.int32)
        cut_h = np.array(H * cut_ratio).astype(np.int32)

        # generating cuting bbox
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    # cutmix function
    def cutmix(self, images, labels):
        indices = torch.randperm(images.size(0))
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]
        
        lam = np.random.beta(1.0,1.0)

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)

        # fetch the images from the shuffled imagees on the orinal images
        images[:, :, bbx1:bbx2, bby1:bby2] = shuffled_images[:, :, bbx1:bbx2, bby1:bby2]
        lam = 1- ((bbx2-bbx1)*(bby2-bby1) / (images.size()[-1]*images.size()[-2]))

        return images, labels, shuffled_labels, lam 

        