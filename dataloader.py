from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Class to create a dataset for classification
class ClassificationDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.transform = transform
        self.imgs = imageFolderDataset.imgs

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert("L")
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label

    def __len__(self):
        return len(self.imgs)

def create_dataloader(
        TRAIN_DATA_DIR=r"D:\Pedro\Docs\siamese_network_test\test\outputs_all_data_filtered_firmino\train",
        TEST_DATA_DIR=r"D:\Pedro\Docs\siamese_network_test\test\outputs_all_data_filtered_firmino\test",
        batch_size=128,
):
    transformation = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    folder_train_dataset = datasets.ImageFolder(root=TRAIN_DATA_DIR+"\\")
    folder_test_dataset = datasets.ImageFolder(root=TEST_DATA_DIR+"\\")

    classification_train_dataset = ClassificationDataset(
        imageFolderDataset=folder_train_dataset, transform=transformation)
    classification_test_dataset = ClassificationDataset(
        imageFolderDataset=folder_test_dataset, transform=transformation)

    train_dataloader = DataLoader(classification_train_dataset, shuffle=True,
                                  num_workers=0, batch_size=batch_size)
    test_dataloader = DataLoader(classification_test_dataset,
                                 num_workers=0, batch_size=batch_size)
    
    print("Done!\n")

    return train_dataloader, test_dataloader