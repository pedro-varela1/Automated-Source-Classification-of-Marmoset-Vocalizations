import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from inceptionResnetV1 import InceptionResnetV1
from tqdm import tqdm
import torch.nn.functional as F

class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Dataset for loading all test images with their respective labels
        
        Args:
            data_dir: Directory containing subdirectories for each individual
            transform: Optional transform to be applied on images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_to_id = {}
        
        # Walk through all subdirectories
        for label_idx, person_dir in enumerate(sorted(os.listdir(data_dir))):
            person_path = os.path.join(data_dir, person_dir)
            if os.path.isdir(person_path):
                self.label_to_id[person_dir] = label_idx
                for img_name in os.listdir(person_path):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(person_path, img_name)
                        self.images.append(img_path)
                        self.labels.append(label_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path

def get_best_images(model, dataloader, device):
    """Extract embeddings for all images"""
    model.eval()
    most_confident_images = {}

    with torch.no_grad():
        for images, batch_labels, batch_paths in tqdm(dataloader, desc="Extracting best images"):
            images = images.to(device)

            logits = model.forward(images)
            predicted_labels = torch.argmax(logits, dim=1)
            confidences, _ = torch.max(F.softmax(logits, dim=1), dim=1)
            for i in range(len(predicted_labels)):
                if predicted_labels[i] == batch_labels[i]:
                    label = predicted_labels[i].item()
                    confidence = confidences[i].item()
                    path = batch_paths[i]
                    if label not in most_confident_images or confidence >= most_confident_images.get(label, {'confidence': -1})['confidence']:
                        most_confident_images[label] = {
                            'path': path,
                            'confidence': confidence
                        }
    
    return most_confident_images

def get_embeddings(model, dataloader, device):
    """Extract embeddings for all images"""
    model.eval()
    embeddings = []
    labels = []
    paths = []
    
    with torch.no_grad():
        for images, batch_labels, batch_paths in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)

            ##### Get features from the model
            features = model.forward_once(images)
            features = F.normalize(features, p=2, dim=1)  # Normalize for cosine similarity

            # Store results
            embeddings.append(features.cpu().numpy())
            labels.extend(batch_labels.numpy())
            paths.extend(batch_paths)
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    return embeddings, labels, paths

def plot_tsne(embeddings, labels, output_path, label_to_id=None):
    """Plot t-SNE visualization of the embeddings with distinct colors"""
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, init='random', perplexity=40)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Generate distinct colors
    n_classes = len(np.unique(labels))
 
    individual_colors = {
        'Blue2A': ['#000000', 'o'],
        'Blue3C': ['#004aad', 's'],
        'Pink3C': ['#38b6ff', 's'],
        'Blue3B': ['#ff3131', '*'],
        'Pink3B': ['#fb7a7a', '*'],
        'Blue3A': ['#07ae27', 'D'],
        'Pink3A': ['#5bf77b', 'D'],
        'Blue2C': ['#ffde59', 'v'],
        'Pink2C': ['#ffbd59', 'v'],
    }
    
    # Plot each class separately with different colors and markers
    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        color = individual_colors[list(label_to_id.keys())[list(label_to_id.values()).index(label)]][0]
        marker = individual_colors[list(label_to_id.keys())[list(label_to_id.values()).index(label)]][1]
        # marker = 'o'
        
        plt.scatter(embeddings_2d[mask, 0], 
                   embeddings_2d[mask, 1],
                   c=[color],
                   marker=marker,
                   s=100,  # Increased marker size
                   alpha=0.7,
                   label=list(label_to_id.keys())[list(label_to_id.values()).index(label)] if label_to_id else f"Class {label}")
    
    # Customize plot
    plt.title('t-SNE visualization of embeddings', fontsize=14, pad=20, fontweight='bold')
    plt.xlabel('t-SNE dimension 1', fontsize=12, fontweight='bold')
    plt.ylabel('t-SNE dimension 2', fontsize=12, fontweight='bold')
    
    # Add legend with two columns
    legend = plt.legend(bbox_to_anchor=(1.05, 1), 
                        loc='upper left', 
                        borderaxespad=0.,
                        fontsize=10,
                        ncol=2,
                        title='Individuals')
    legend.get_title().set_fontweight('bold')
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add a border around the plot
    plt.box(True)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

    # Create a second plot with larger figure for better legend visibility
    plt.figure(figsize=(12, n_classes//3))
    plt.axis('off')
    
    # Create legend only plot
    for i, label in enumerate(np.unique(labels)):
        color = individual_colors[list(label_to_id.keys())[list(label_to_id.values()).index(label)]][0]
        marker = individual_colors[list(label_to_id.keys())[list(label_to_id.values()).index(label)]][1]
        plt.scatter([], [], 
                   c=[color],
                   marker=marker,
                   s=100,
                   label=list(label_to_id.keys())[list(label_to_id.values()).index(label)])
    
    legend = plt.legend(bbox_to_anchor=(0.5, 0.5),
              loc='center',
              fontsize=12,
              ncol=3,
                title='Individuals')
    legend.get_title().set_fontweight('bold')
    
    plt.savefig('legend_tsne.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

def main():
    # Parameters
    TEST_DATA_DIR = "./data/test"  # Directory with subdirectories for each person
    MODEL_PATH = "checkpoints/best_model_classification.pth"  # Path to trained model
    BATCH_SIZE = 128
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define transforms
    transform = transforms.Compose([transforms.Resize((160, 160)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5], std=[0.5])])
    
    # Create dataset and dataloader
    dataset = TestDataset(TEST_DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load model
    # model = SiameseNetwork().to(DEVICE)
    model = InceptionResnetV1(device='cuda' if torch.cuda.is_available() else 'cpu', classify=True, num_classes=9)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    # Handle both cases: full model save and state_dict save
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)
    
    # Get embeddings
    print("Extracting embeddings...")
    embeddings, labels, paths = get_embeddings(model, dataloader, DEVICE)

    # Get most confident images
    """print("Extracting most confident images...")
    most_confident_images = get_best_images(model, dataloader, DEVICE)
    for label, info in most_confident_images.items():
        person_name = [k for k, v in dataset.label_to_id.items() if v == label][0]
        print(f"{person_name} (Class {label}): {info['path']} (Confidence: {info['confidence']:.4f})")"""
    
    # Plot t-SNE
    print("Creating t-SNE visualization...")
    plot_tsne(embeddings, labels, 'embeddings_tsne.png', dataset.label_to_id)
    
    print("Done! Visualization saved as 'embeddings_tsne.png'")
    
    # Print some statistics
    unique_labels = np.unique(labels)
    print(f"\nNumber of unique individuals: {len(unique_labels)}")
    print(f"Total number of images: {len(labels)}")
    print("\nImages per individual:")
    for label in unique_labels:
        person_name = [k for k, v in dataset.label_to_id.items() if v == label][0]
        count = np.sum(labels == label)
        print(f"{person_name}: {count} images")

if __name__ == "__main__":
    main()