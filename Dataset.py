import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# DATASET
# ============================================================================

class BaseTextDataset(Dataset):
    """Base dataset for text restoration"""
    def __init__(self, img_size=(64, 256), augment=False, config):
        self.img_size = img_size
        self.augment = augment
        self.config = config
        self.image_pairs = []
        
        base_transforms = [transforms.Resize(img_size), transforms.ToTensor()] if img_size else [transforms.ToTensor()]
        self.transform = transforms.Compose(base_transforms)
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST) if img_size else transforms.Lambda(lambda x: x),
            transforms.ToTensor()
        ])
    
    def apply_corruption(self, image):
        """Apply random corruption"""
        noise = torch.randn_like(image) * random.uniform(0.05, 0.15)
        return torch.clamp(image + noise, 0, 1)
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        
        if 'is_coco' in pair and pair['is_coco']:
            image = pair['image']
            mask = pair.get('mask', torch.ones(1, self.img_size[0], self.img_size[1]))
            corrupted = self.apply_corruption(image.clone())
            
            return {
                'corrupted': torch.clamp(corrupted, 0, 1),
                'clean': torch.clamp(image, 0, 1),
                'mask': torch.clamp(mask, 0, 1),
                'image_id': pair['id'],
                'dataset_type': 'coco'
            }
        else:
            corrupted = Image.open(pair['corrupted']).convert('RGB')
            clean = Image.open(pair['clean']).convert('RGB')
            mask = Image.open(pair['mask']).convert('L')
            
            corrupted = self.transform(corrupted)
            clean = self.transform(clean)
            mask = self.mask_transform(mask)
            
            if self.augment and random.random() < 0.7:
                corrupted = self.apply_corruption(corrupted)
            
            return {
                'corrupted': torch.clamp(corrupted, 0, 1),
                'clean': torch.clamp(clean, 0, 1),
                'mask': torch.clamp(mask, 0, 1),
                'image_id': pair['id'],
                'dataset_type': pair.get('type', 'unknown')
            }


class MergedTextDataset(BaseTextDataset):
    """Merged dataset for synthetic and handwritten text"""
    
    def __init__(self, synthetic_dir=None, handwritten_dir=None, split='train', 
                 augment=False, config=None, img_size=None):
        super().__init__(img_size, augment, config)
        
        if synthetic_dir:
            self._load_synthetic(Path(synthetic_dir), split)
        if handwritten_dir:
            self._load_handwritten(Path(handwritten_dir))
    
    def _load_synthetic(self, root_dir, split):
        if split == 'train':
            incomplete_dir = root_dir / 'train_incomplete'
            complete_dir = root_dir / 'train_f'
            mask_dir = root_dir / 'train_mask'
        else:
            incomplete_dir = root_dir / 't_incomplete'
            complete_dir = root_dir / 't_f'
            mask_dir = root_dir / 't_mask'
        
        if not incomplete_dir.exists():
            logger.warning(f"Synthetic directory not found: {incomplete_dir}")
            return
        
        for img_path in incomplete_dir.glob('*.png'):
            img_id = img_path.stem
            complete_path = complete_dir / f"{img_id}.png"
            mask_path = mask_dir / f"{img_id}.png"
            
            if complete_path.exists() and mask_path.exists():
                self.image_pairs.append({
                    'id': f'syn_{img_id}',
                    'corrupted': str(img_path),
                    'clean': str(complete_path),
                    'mask': str(mask_path),
                    'type': 'synthetic'
                })
    
    def _load_handwritten(self, root_dir):
        corrupted_dir = root_dir / 'Corrupted_Image'
        intact_dir = root_dir / 'Intact_Image'
        mask_dir = root_dir / 'Corrupted_mask'
        
        if not corrupted_dir.exists():
            logger.warning(f"Handwritten directory not found: {corrupted_dir}")
            return
        for img_path in corrupted_dir.glob('*.png'):
            img_id = img_path.stem
            intact_path = intact_dir / f"{img_id}.png"
            mask_path = mask_dir / f"{img_id}.png"
            
            if intact_path.exists() and mask_path.exists():
                self.image_pairs.append({
                    'id': f'hw_{img_id}',
                    'corrupted': str(img_path),
                    'clean': str(intact_path),
                    'mask': str(mask_path),
                    'type': 'handwritten'
                })