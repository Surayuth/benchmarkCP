import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import json

"""
At this stage, we would like to use pretrained models with the ImageNet dataset for calibration 
and measure the performance of each CP methods. You may need to implement the get_train_data if you
would like to train the model with imagenet dataset.
"""

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
    

def get_test_calib_data(val_size=0.1, calib_size=0.1, split_seed1=42, split_seed2=42):
    """
    val set --> split to test set and calib set
    """

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = torchvision.datasets.ImageNet(
        root="./imagenet1k",
        split="val"
    )

    with open("imagenet1k/ImageNet_class_index.json") as f:
        class_dict = {v[0]:int(k) for k, v in json.load(f).items()}

    assert len(class_dict) == 1000, "Missing classes"
    assert len(set(class_dict.values())) == 1000, "Duplicated Classes"

    test_idx, calib_idx = train_test_split(
        list(range(len(dataset))),
        test_size=calib_size, 
        random_state=split_seed1,
        shuffle=True,
        stratify=dataset.targets
    )

    test_dataset = TransformedSubset(dataset, test_idx, eval_transform)
    calib_dataset = TransformedSubset(dataset, calib_idx, eval_transform)

    return class_dict, None, None, calib_dataset, test_dataset

if __name__ == "__main__":
    class_dict,_,_, calib_dataset, test_dataset = get_test_calib_data()
    print(class_dict)
    print(len(calib_dataset), len(test_dataset)) # 5000, 45000