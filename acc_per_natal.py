import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import torchvision.datasets as datasets
from dataloader import create_dataloader
from inceptionResnetV1 import InceptionResnetV1

def load_model(checkpoint_path, model, device):
    """
    Carrega o modelo treinado a partir do checkpoint
    
    Args:
        checkpoint_path: Caminho para o arquivo do checkpoint
        model: Instância do modelo
        device: Dispositivo onde o modelo será carregado
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model

# @torch.no_grad()
# def get_accuracy(model, test_loader, device):
#     """
#     Calcula a acurácia do modelo no conjunto de teste

#     Args:
#         model: Modelo treinado
#         test_loader: DataLoader do conjunto de teste
#         device: Dispositivo onde o modelo está

#     Returns:
#         accuracy: Acurácia do modelo no conjunto de teste
#     """
#     model.eval()
#     num_correct = 0
#     total = 0

#     for inputs, labels in tqdm(test_loader, desc='Calculando acurácia'):
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         outputs = model(inputs)
#         _, predictions = torch.max(outputs, 1)

#         num_correct += (predictions == labels).sum().item()
#         total += labels.size(0)

#     accuracy = num_correct / total

#     return accuracy

def get_accuracy(true_labels, pred_labels):
    """
    Calcula a acurácia do modelo no conjunto de teste

    Args:
        true_labels: Rótulos verdadeiros
        pred_labels: Rótulos preditos

    Returns:
        accuracy: Acurácia do modelo no conjunto de teste
    """
    num_correct = 0
    total = 0

    for true_label, pred_label in zip(true_labels, pred_labels):
        num_correct += true_label == pred_label
        total += 1

    accuracy = num_correct / total

    return accuracy

@torch.no_grad()
def get_predictions(model, test_loader, device):
    """
    Obtém as predições do modelo para todo o conjunto de teste
    
    Args:
        model: Modelo treinado
        test_loader: DataLoader do conjunto de teste
        device: Dispositivo onde o modelo está
    
    Returns:
        true_labels: Lista com os rótulos verdadeiros
        pred_labels: Lista com as predições do modelo
    """
    model.eval()
    true_labels = []
    pred_labels = []
    
    for inputs, labels in tqdm(test_loader, desc='Gerando predições'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predictions.cpu().numpy())
    
    return true_labels, pred_labels

def get_twin_confusion_matrix(true_labels, pred_labels, class_names):

    # Calcula a matriz de confusão
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Normaliza a matriz
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    Twins = {
        'Blue2A': '',

        'Blue2C': 'Pink2C',
        'Blue3A': 'Pink3A',
        'Blue3B': 'Pink3B',
        'Blue3C': 'Pink3C',

        'Pink2C': 'Blue2C',
        'Pink3A': 'Blue3A',
        'Pink3B': 'Blue3B',
        'Pink3C': 'Blue3C'
    }

    # Get mean cofusion for twins
    twin_confusion = []
    for i, class_name in enumerate(class_names):
        for j, twin_name in enumerate(class_names):
            if class_name == Twins[twin_name]:
                twin_confusion.append(cm_normalized[i, j])
    
    return np.mean(twin_confusion)

def plot_accuracy_per_natal(vector_per_natal, vector_accuracy, vector_confusion, color_plot='#008888'):
    """
    Plota a acurácia e o erro do modelo para cada conjunto de teste

    Args:
        vector_per_natal: Vetor com o nome de cada conjunto de teste
        vector_accuracy: Vetor com a acurácia para cada conjunto de teste
        vector_confusion: Vetor com o erro para cada conjunto de teste
    """
    per_natal = [f"{int(natal.split('/')[-1].split('_')[1]):02d}-{int(natal.split('/')[-1].split('_')[2]):02d}" for natal in vector_per_natal]
    
    fig, ax1 = plt.subplots(figsize=(18, 7))

    # Plot da acurácia
    ax1.plot(per_natal, vector_accuracy, marker='o', color=color_plot, linestyle='-', linewidth=2, label='Accuracy')
    ax1.set_xlabel("Days after birth", fontweight='bold', fontsize=12)
    ax1.set_ylabel("Accuracy", fontweight='bold', fontsize=12, color=color_plot)
    ax1.set_ylim(0.8, 1)
    ax1.set_yticks(np.arange(0.8, 1.01, 0.05))
    ax1.tick_params(axis='y', labelcolor=color_plot)

    # Cria um segundo eixo y para o erro
    ax2 = ax1.twinx()
    ax2.plot(per_natal, vector_confusion, marker='o', color='red', linestyle='-', linewidth=2, label='Error')
    ax2.set_ylabel("Mean Twin Confusion", fontweight='bold', fontsize=12, color='red')
    ax2.set_ylim(0, 0.05)
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title("Model accuracy and mean twin confusion by age selected in the test dataset", fontsize=16, fontweight='bold')

    # Combina as legendas
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.savefig("./model_classification/accuracy_error_per_natal.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Configurações
    BATCH_SIZE = 128
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CHECKPOINT_PATH = "./checkpoints/best_model_classification.pth"
    
    vector_dir = [f"./data/test_weeks/test_{i}_{i+6}" for i in np.arange(7, 84, 7)]
    # vector_dir = ["./data/test_8_29", "./data/test_30_50", "./data/test_64_84"]
    vector_per_natal = []
    vector_accuracy = []
    vector_confusion = []

    for dir in vector_dir:
        # Obtém os nomes das classes diretamente do diretório de teste
        dataset = datasets.ImageFolder(root=dir)
        CLASS_NAMES = dataset.classes
        num_classes = len(CLASS_NAMES)
        
        # Cria o dataloader de teste
        _, test_loader = create_dataloader(
            TRAIN_DATA_DIR=dir,  # não precisamos do train_loader
            TEST_DATA_DIR=dir,
            batch_size=BATCH_SIZE
        )
        
        # Cria e carrega o modelo
        model = InceptionResnetV1(device=DEVICE, classify=True, num_classes=num_classes)
        model = load_model(CHECKPOINT_PATH, model, DEVICE)
        model = model.to(DEVICE)

        true_labels, pred_labels = get_predictions(model, test_loader, DEVICE)

        # accuracy = get_accuracy(model, test_loader, DEVICE)
        accuracy = get_accuracy(true_labels, pred_labels)
        twin_confusion = get_twin_confusion_matrix(true_labels, pred_labels, CLASS_NAMES)
        print(f"Acurácia para {dir}: {accuracy:.4f}")
        print(f"Confusion for {dir}: {twin_confusion:.4f}")

        vector_per_natal.append(dir)
        vector_accuracy.append(accuracy)
        vector_confusion.append(twin_confusion)

    plot_accuracy_per_natal(vector_per_natal, vector_accuracy, vector_confusion)
    

if __name__ == "__main__":
    # Define seeds para reprodutibilidade
    seed = 43
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    main()