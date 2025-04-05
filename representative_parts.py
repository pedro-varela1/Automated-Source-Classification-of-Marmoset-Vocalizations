import torchvision.datasets as datasets
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
import random
from inceptionResnetV1 import InceptionResnetV1
import pandas as pd

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

def create_occlusion_importance_map(model, img_path, device, num_regions=10):
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
        
        # Calcula o impacto (1 - (prob / orig_prob))
        if prob > orig_prob:
            impact = 0
        else:
            impact = 1 - (prob / orig_prob)
        occlusion_impact[row, col] = impact
    
    # O impacto mais alto indica a região mais importante
    importance_map = occlusion_impact
    
    return img_np, importance_map, true_class, orig_prob
    
def main():
    # Configurações
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TEST_DATA_DIR = r"J:\all_animais_phee\firmino_img\exp"
    CHECKPOINT_PATH = "./checkpoints/best_model_classification.pth"
    NUM_REGIONS = 10  # Número de regiões por dimensão (10x10)
    NUM_SAMPLES = 50  # Número de amostras a serem processadas por diretório
    OUTPUT_FILE = f"occlusion_results_r{NUM_REGIONS}_s{NUM_SAMPLES}.csv"
    # Get class names in alphabetical order
    class_names = sorted(os.listdir(TEST_DATA_DIR))

    
    # Carrega o modelo
    dataset = datasets.ImageFolder(root=TEST_DATA_DIR)
    CLASS_NAMES = dataset.classes
    num_classes = len(CLASS_NAMES)
    
    model = InceptionResnetV1(device=DEVICE, classify=True, num_classes=num_classes)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    # Create a single DataFrame to store all results
    all_results = pd.DataFrame()
    
    # Processa todas as imagens de teste
    for root, _, files in os.walk(TEST_DATA_DIR):
        image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        sample_files = random.sample(image_files, min(len(image_files), NUM_SAMPLES))
        
        print(f"Processing {len(sample_files)} images in {root}")
        for file in tqdm(sample_files, desc='Processando imagens'):
            img_path = os.path.join(root, file)
            try:
                _, importance_map, true_class, orig_prob = create_occlusion_importance_map(
                        model, img_path, DEVICE, num_regions=NUM_REGIONS)
                
                # Create DataFrame for this image
                df = pd.DataFrame(importance_map.flatten(), columns=['impact'])
                df['image_path'] = img_path
                df['class_name'] = os.path.basename(root)
                df['orig_prob'] = orig_prob
                df['row_region'] = np.repeat(np.arange(NUM_REGIONS), NUM_REGIONS)
                df['col_region'] = np.tile(np.arange(NUM_REGIONS), NUM_REGIONS)
                df['pred_class'] = class_names[int(true_class)]
                
                # Append to the combined results
                all_results = pd.concat([all_results, df], ignore_index=True)
                
            except Exception as e:
                print(f"Erro ao processar {img_path}: {str(e)}")
    
    # Save all results to a single CSV
    all_results.to_csv(OUTPUT_FILE, index=False)
    print(f"All results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()