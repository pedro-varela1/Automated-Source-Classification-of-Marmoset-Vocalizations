import torchvision.datasets as datasets
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os
from tqdm import tqdm
import matplotlib.patches as patches
import random
from inceptionResnetV1 import InceptionResnetV1

def predict_with_occlusion(model, input_tensor, region_idx, region_size, num_regions, device):
    """
    Faz uma predição com uma região específica ocluída (pixels zerados)
    
    Args:
        model: Modelo treinado
        input_tensor: Tensor da imagem original (1, C, H, W)
        region_idx: Índice da região a ser ocluída (0-9)
        region_size: Tamanho da região (ex: 16)
        num_regions: Número de regiões por dimensão (ex: 10)
        device: Dispositivo (cuda/cpu)
    
    Returns:
        output: Saída do modelo
        occluded_tensor: Tensor da imagem com a região ocluída
    """
    # Cria uma cópia do tensor de entrada
    occluded_tensor = input_tensor.clone()
    
    # Calcula a posição da região a ser ocluída
    row = region_idx // num_regions
    col = region_idx % num_regions
    
    # Calcula as coordenadas da região
    start_y = row * region_size
    start_x = col * region_size
    end_y = start_y + region_size
    end_x = start_x + region_size
    # Aplica a oclusão (zera os pixels na região) para o menor valor possível de pixel no total
    occluded_tensor[0, :, start_y:end_y, start_x:end_x] = occluded_tensor.min()
    
    # Faz a predição
    with torch.no_grad():
        output = model(occluded_tensor)
    
    return output, occluded_tensor

def find_important_region(model, img_path, class_names, device, num_regions=10):
    """
    Encontra a região mais importante para a predição correta usando oclusão
    
    Args:
        model: Modelo treinado
        img_path: Caminho para a imagem de entrada
        class_names: Lista com nomes das classes
        device: Dispositivo (cuda/cpu)
        num_regions: Número de regiões por dimensão
    
    Returns:
        img_np: Array numpy da imagem original
        important_region_idx: Índice da região mais importante
        bbox_coords: Coordenadas do bounding box (x_min, y_min, width, height)
        true_class: Classe verdadeira predita
        orig_prob: Probabilidade original
        min_prob: Probabilidade mínima após oclusão
    """
    # Carrega e pré-processa a imagem
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Carrega a imagem em escala de cinza
    img = Image.open(img_path).convert('L')
    img_np = np.array(img.resize((160, 160)))
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Faz a predição original
    with torch.no_grad():
        output = model(input_tensor)
        true_class = torch.argmax(output).item()
        orig_prob = torch.softmax(output, dim=1)[0, true_class].item()
    
    # Calcula o tamanho de cada região
    image_size = 160
    region_size = image_size // num_regions
    total_regions = num_regions * num_regions
    
    # Lista para armazenar a probabilidade para cada região ocluída
    probs = []
    
    # Testa cada região
    for region_idx in range(total_regions):
        # Faz a predição com a região ocluída
        output, _ = predict_with_occlusion(model, input_tensor, region_idx, region_size, num_regions, device)
        
        # Calcula a probabilidade da classe verdadeira
        prob = torch.softmax(output, dim=1)[0, true_class].item()
        probs.append(prob)
    
    # Encontra a região com a menor probabilidade (mais importante)
    important_region_idx = np.argmin(probs)
    min_prob = probs[important_region_idx]
    
    # Calcula as coordenadas do bounding box
    row = important_region_idx // num_regions
    col = important_region_idx % num_regions
    x_min = col * region_size
    y_min = row * region_size
    
    bbox_coords = (x_min, y_min, region_size, region_size)
    
    return img_np, important_region_idx, bbox_coords, true_class, orig_prob, min_prob

def visualize_important_region(img_np, bbox_coords, true_class, class_names, orig_prob, min_prob):
    """
    Cria uma visualização com bounding box para a região mais importante
    
    Args:
        img_np: Array numpy da imagem original
        bbox_coords: Coordenadas do bounding box (x_min, y_min, width, height)
        true_class: Classe verdadeira predita
        class_names: Lista com nomes das classes
        orig_prob: Probabilidade original
        min_prob: Probabilidade mínima após oclusão
    
    Returns:
        fig: Figura matplotlib com a visualização
    """
    # Desempacota as coordenadas
    x_min, y_min, width, height = bbox_coords
    
    # Cria a figura
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Mostra a imagem original
    ax.imshow(img_np, cmap='gray')
    
    # Adiciona o bounding box
    rect = patches.Rectangle((x_min, y_min), width, height, 
                           linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    # Adiciona o valor da queda de probabilidade como texto acima do bounding box
    prob_drop = orig_prob - min_prob
    ax.text(x_min, y_min - 10, f'Prob: {min_prob:.4f} (queda: {prob_drop:.4f})', 
          color='red', fontsize=12, backgroundcolor='white')
    
    ax.set_title(f'Região mais importante para a classe: {class_names[true_class]}')
    ax.axis('off')
    
    # Adiciona informações sobre a predição original
    plt.figtext(0.5, 0, f'Classe: {class_names[true_class]} (Prob. Original: {orig_prob:.4f})', 
              ha='center', fontsize=12)
    
    return fig

def save_occlusion_importance_visualization(model, img_path, save_path, class_names, device, num_regions=10):
    """
    Salva a visualização da região mais importante usando oclusão
    
    Args:
        model: Modelo treinado
        img_path: Caminho para a imagem de entrada
        save_path: Caminho para salvar a visualização
        class_names: Lista com nomes das classes
        device: Dispositivo (cuda/cpu)
        num_regions: Número de regiões por dimensão
    """
    # Encontra a região mais importante
    img_np, important_region_idx, bbox_coords, true_class, orig_prob, min_prob = find_important_region(
        model, img_path, class_names, device, num_regions)
    
    # Cria a visualização
    fig = visualize_important_region(img_np, bbox_coords, true_class, class_names, orig_prob, min_prob)
    
    # Salva a figura
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Configurações
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TEST_DATA_DIR = r"J:\all_animais_phee\firmino_img\exp"
    CHECKPOINT_PATH = "./checkpoints/best_model_classification.pth"
    OUTPUT_DIR = "./occlusion_importance_visualizations"
    
    # Cria diretório de saída se não existir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Carrega o modelo
    dataset = datasets.ImageFolder(root=TEST_DATA_DIR)
    CLASS_NAMES = dataset.classes
    num_classes = len(CLASS_NAMES)
    
    model = InceptionResnetV1(device=DEVICE, classify=True, num_classes=num_classes)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    # Processa todas as imagens de teste
    for root, dirs, files in os.walk(TEST_DATA_DIR):
        image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        sample_files = random.sample(image_files, min(len(image_files), 40))

        for file in tqdm(sample_files, desc='Processando imagens'):
            img_path = os.path.join(root, file)
            os.makedirs(os.path.join(OUTPUT_DIR, os.path.basename(root)), exist_ok=True)
            save_path = os.path.join(OUTPUT_DIR, os.path.basename(root),
                                   f'occlusion_{os.path.basename(img_path)}')
            try:
                save_occlusion_importance_visualization(model, img_path, save_path, CLASS_NAMES, DEVICE)
            except Exception as e:
                print(f"Erro ao processar {img_path}: {str(e)}")

if __name__ == "__main__":
    main()