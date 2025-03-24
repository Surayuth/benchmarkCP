import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class TransformedSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        
    def __getitem__(self, idx):
        img, target = self.dataset[self.indices[idx]]
        
        if self.transform:
            img = self.transform(img)
            
        return img, target
    
    def __len__(self):
        return len(self.indices)

def get_data(val_size=0.1, calib_size=0.1, split_seed1=42, split_seed2=42):
    """
    train_size = tot_size - (val_size + calib_size)
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./cifar10/train", 
        train=True, 
        download=True,
        )
    
    class_dict = dataset.class_to_idx

    train_idx, val_calib_idx = train_test_split(
        list(range(len(dataset))),
        test_size=val_size+calib_size,
        random_state=split_seed1,
        shuffle=True,
        stratify=dataset.targets
    )

    val_calib_targets = [dataset.targets[i] for i in val_calib_idx]

    val_idx_temp, calib_idx_temp = train_test_split(
        list(range(len(val_calib_idx))),
        test_size=calib_size/(val_size+calib_size),
        random_state=split_seed2,
        shuffle=True,
        stratify=val_calib_targets
    )

    # Map back to original dataset indices
    val_idx = [val_calib_idx[i] for i in val_idx_temp]
    calib_idx = [val_calib_idx[i] for i in calib_idx_temp]

    # Create final datasets
    train_dataset = TransformedSubset(dataset, train_idx, train_transform)
    val_dataset = TransformedSubset(dataset, val_idx, eval_transform)
    calib_dataset = TransformedSubset(dataset, calib_idx, eval_transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root="./cifar10/test",
        train=False,
        download=True,
        transform=eval_transform
    )

    return class_dict, train_dataset, val_dataset, calib_dataset, test_dataset