from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torch

class PlantPathologyDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_np = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype(np.float32) 
        image = self.transform(image = image_np)['image']

        if self.labels is not None:
            label = torch.tensor(self.labels[idx])
            return image, label
        else:
            return image
    
def plant_data_loader(df, val_df=None, batch_size=16, train_transform=None, val_transform=None, is_test=False):
    # Check if target column exists
    has_target = 'target' in df.columns
    
    if has_target and not is_test:
        # Training/validation mode with labels
        train_dataset = PlantPathologyDataset(df['image_path'].tolist(), df['target'].tolist(), train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_df is not None:
            val_dataset = PlantPathologyDataset(val_df['image_path'].tolist(), val_df['target'].tolist(), val_transform)
            val_loader = DataLoader(val_dataset, batch_size=batch_size*4, shuffle=False)
            return train_loader, val_loader
        else:
            return train_loader
    else:
        # Test mode or no target column - no labels
        train_dataset = PlantPathologyDataset(df['image_path'].tolist(), labels=None, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        if val_df is not None:
            val_dataset = PlantPathologyDataset(val_df['image_path'].tolist(), labels=None, transform=val_transform)
            val_loader = DataLoader(val_dataset, batch_size=batch_size*4, shuffle=False)
            return train_loader, val_loader
        else:
            return train_loader