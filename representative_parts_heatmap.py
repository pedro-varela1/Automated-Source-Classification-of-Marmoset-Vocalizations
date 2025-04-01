import torchvision.datasets as datasets
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os
from tqdm import tqdm
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import random
from inceptionResnetV1 import InceptionResnetV1

def predict_with_occlusion(model, input_tensor, region_idx, region_size, num_regions, device):
    """
    Faz uma predição com uma região específica ocluída (pixels zerados)
    
    Args:
        model: Modelo treinado
        input_tensor: Tensor da imagem original (1, C, H, W)
        region_idx: Índice da região a ser ocluída (0-99 para grid 10x10)
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
    
    # Aplica a oclusão (zera os pixels na região)
    occluded_tensor[0, :, start_y:end_y, start_x:end_x] = 0
    
    # Faz a predição
    with torch.no_grad():
        output = model(occluded_tensor)
    
    return output, occluded_tensor

def create_occlusion_importance_map(model, img_path, class_names, device, num_regions=10):
    """
    Cria um mapa de importância para todas as regiões usando oclusão
    
    Args:
        model: Modelo treinado
        img_path: Caminho para a imagem de entrada
        class_names: Lista com nomes das classes
        device: Dispositivo (cuda/cpu)
        num_regions: Número de regiões por dimensão
    
    Returns:
        img_np: Array numpy da imagem original
        importance_map: Mapa de importância (matriz num_regions x num_regions)
        true_class: Classe verdadeira predita
        orig_prob: Probabilidade original
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
    
    # Armazena o impacto de ocluir cada região
    occlusion_impact = np.zeros((num_regions, num_regions))
    
    # Testa cada região
    for region_idx in range(total_regions):
        # Calcula a posição da região
        row = region_idx // num_regions
        col = region_idx % num_regions
        
        # Faz a predição com a região ocluída
        output, _ = predict_with_occlusion(model, input_tensor, region_idx, region_size, num_regions, device)
        
        # Calcula a probabilidade da classe verdadeira
        prob = torch.softmax(output, dim=1)[0, true_class].item()
        
        # Calcula o impacto (queda na probabilidade)
        impact = orig_prob - prob
        occlusion_impact[row, col] = impact
    
    # O impacto mais alto indica a região mais importante
    importance_map = occlusion_impact
    
    return img_np, importance_map, true_class, orig_prob

def visualize_importance_heatmap(img_np, importance_map, true_class, class_names, orig_prob, region_size=16):
    """
    Cria uma visualização com mapa de calor para a importância de todas as regiões
    
    Args:
        img_np: Array numpy da imagem original
        importance_map: Mapa de importância (matriz)
        true_class: Classe verdadeira predita
        class_names: Lista com nomes das classes
        orig_prob: Probabilidade original
        region_size: Tamanho de cada região
    
    Returns:
        fig: Figura matplotlib com a visualização
    """
    # Encontra os valores mínimo e máximo para normalização do mapa de calor
    min_impact = importance_map.min()
    max_impact = importance_map.max()
    
    # Encontra a região mais importante
    max_row, max_col = np.unravel_index(np.argmax(importance_map), importance_map.shape)
    max_impact = importance_map[max_row, max_col]
    
    # Cria a figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Mostra a imagem original
    ax1.imshow(img_np, cmap='gray')
    ax1.set_title('Imagem Original')
    ax1.axis('off')
    
    # Cria um mapa de calor interpolado para sobrepor à imagem
    # Redimensiona o mapa de importância para o tamanho da imagem
    heatmap_img = np.zeros((160, 160))
    for i in range(len(importance_map)):
        for j in range(len(importance_map[0])):
            y_start = i * region_size
            x_start = j * region_size
            heatmap_img[y_start:y_start+region_size, x_start:x_start+region_size] = importance_map[i, j]
    
    # Normaliza o mapa de calor
    heatmap_img = (heatmap_img - min_impact) / (max_impact - min_impact + 1e-8)
    
    # Mostra a imagem com o mapa de calor sobreposto
    img_with_heatmap = ax2.imshow(img_np, cmap='gray')
    
    # Define uma colormap personalizada de azul (menos importante) para vermelho (mais importante)
    cmap = plt.cm.jet
    
    # Sobrepõe o mapa de calor com transparência
    heatmap = ax2.imshow(heatmap_img, cmap=cmap, alpha=0.6)
    
    # Destaca a região mais importante com uma caixa branca
    x_min = max_col * region_size
    y_min = max_row * region_size
    rect = patches.Rectangle((x_min, y_min), region_size, region_size, 
                           linewidth=2, edgecolor='white', facecolor='none')
    ax2.add_patch(rect)
    
    # Adiciona o valor da queda de probabilidade como texto acima do retângulo
    ax2.text(x_min, y_min - 5, f'Impacto: {max_impact:.4f}', 
           color='white', fontsize=10, backgroundcolor='black')
    
    ax2.set_title(f'Mapa de Importância (Queda na Probabilidade)')
    ax2.axis('off')
    
    # Adiciona uma barra de cores
    cbar = plt.colorbar(heatmap, ax=ax2, orientation='vertical', label='Importância (Queda na Probabilidade)')
    
    # Adiciona informações sobre a predição original
    plt.suptitle(f'Análise de Importância para Classe: {class_names[true_class]} (Prob. Original: {orig_prob:.4f})')
    
    return fig

def save_occlusion_heatmap_visualization(model, img_path, save_path, class_names, device, num_regions=10):
    """
    Salva a visualização do mapa de calor de importância usando oclusão
    
    Args:
        model: Modelo treinado
        img_path: Caminho para a imagem de entrada
        save_path: Caminho para salvar a visualização
        class_names: Lista com nomes das classes
        device: Dispositivo (cuda/cpu)
        num_regions: Número de regiões por dimensão
    """
    # Cria o mapa de importância
    img_np, importance_map, true_class, orig_prob = create_occlusion_importance_map(
        model, img_path, class_names, device, num_regions)
    
    # Cria a visualização
    region_size = 160 // num_regions
    fig = visualize_importance_heatmap(img_np, importance_map, true_class, class_names, orig_prob, region_size)
    
    # Salva a figura
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Configurações
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TEST_DATA_DIR = r"J:\all_animais_phee\firmino_img\exp"
    CHECKPOINT_PATH = "./checkpoints/best_model_classification.pth"
    OUTPUT_DIR = "./occlusion_importance_visualizations_heatmap"
    
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
                                   f'occlusion_heatmap_{os.path.basename(img_path)}')
            try:
                save_occlusion_heatmap_visualization(model, img_path, save_path, CLASS_NAMES, DEVICE)
            except Exception as e:
                print(f"Erro ao processar {img_path}: {str(e)}")

if __name__ == "__main__":
    main()