# 修改后的 dataset.py (只负责读取 .obj)
import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FontObjDataset(Dataset):
    def __init__(self, obj_file_path):
        with open(obj_file_path, 'rb') as f:
            self.data = pickle.load(f)
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 将 PIL Image 转为 Tensor
        source_tensor = self.transform(item['source_img'])
        target_tensor = self.transform(item['target_img'])
        label = torch.tensor(item['label'], dtype=torch.long)
        
        # 返回 source, target 以及重要的 字体类别标签
        return source_tensor, target_tensor, label