import torchvision.datasets as datasets
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os
from tqdm import tqdm
import matplotlib.colors as mcolors
import random
from train_resnet import create_resnet50_model

class GradCAMPlusPlus:
    def __init__(self, model):
        """
        Args:
            model: Modelo treinado
        """
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Registra hooks para capturar gradientes e ativações
        target = self._get_target_layer()
        target.register_forward_hook(self._save_activation)
        target.register_backward_hook(self._save_gradient)
    
    def _get_target_layer(self):
        """Obtém a camada alvo do modelo"""
        return self.model.layer4
    
    def _save_activation(self, module, input, output):
        self.activations = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Gera o mapa de ativação Grad-CAM
        
        Args:
            input_tensor: Tensor de entrada (1, C, H, W)
        
        Returns:
            cam: Mapa de ativação normalizado
        """
        # Forward pass
        model_output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(model_output)
        
        # Zero todos os gradientes
        self.model.zero_grad()
        
        # Backward pass
        class_score = model_output[0, target_class]
        class_score.backward()
        
        gradients = self.gradients  # Gradientes da camada alvo
        activations = self.activations  # Ativações da camada alvo
        
        # Calcula os pesos do Grad-CAM
        weights = gradients.mean(dim=[2, 3], keepdim=True)
        
        # Gera o mapa de ativação
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU para manter apenas valores positivos
        print(f"Mapa de ativação: {cam.squeeze().cpu().detach().numpy()}")
        
        # Normalização e redimensionamento para 160x160
        cam = F.interpolate(cam, size=(160, 160), mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.squeeze().cpu().detach().numpy()

def apply_colormap(cam, img_array):
    """
    Aplica o mapa de calor sobre a imagem original usando matplotlib
    
    Args:
        cam: Mapa de ativação (160, 160)
        img_array: Array numpy da imagem original (160, 160)
    
    Returns:
        visualization: Imagem com mapa de calor sobreposto
    """
    # Normaliza a imagem original para [0, 1]
    img_norm = (img_array - img_array.min()) / (img_array.max() - img_array.min())
    
    # Cria um mapa de cores personalizado
    cmap = plt.cm.jet
    colored_cam = cmap(cam)
    
    # Expande a imagem em escala de cinza para 3 canais
    img_rgb = np.stack([img_norm] * 3, axis=-1)
    
    # Cria a visualização final mesclando a imagem original com o mapa de calor
    visualization = 0.3 * img_rgb + 0.7 * colored_cam[:, :, :3]
    
    return visualization.clip(0, 1)

def save_gradcam_visualization(model, img_path, save_path, class_names, device):
    """
    Salva a visualização Grad-CAM++ para uma imagem
    
    Args:
        model: Modelo treinado
        img_path: Caminho para a imagem de entrada
        save_path: Caminho para salvar a visualização
        class_names: Lista com nomes das classes
        device: Dispositivo (cuda/cpu)
    """
    # Carrega e pré-processa a imagem
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Carrega a imagem em escala de cinza
    img = Image.open(img_path).convert('L')
    input_tensor = transform(img).unsqueeze(0).to(device)
    print(input_tensor)
    # Inicializa Grad-CAM++
    grad_cam = GradCAMPlusPlus(model)
    
    # Obtém a predição
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output).item()
        prob = torch.softmax(output, dim=1)[0, pred_class].item()
    
    # Gera o mapa de ativação
    cam = grad_cam.generate_cam(input_tensor, pred_class)
    
    # Prepara a imagem original para visualização
    img_np = np.array(img.resize((160, 160)))
    
    # Gera a visualização
    visualization = apply_colormap(cam, img_np)
    
    # Cria a figura com a imagem original e o mapa de calor
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot da imagem original em escala de cinza
    ax1.imshow(img_np, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot do mapa de calor
    ax2.imshow(visualization)
    ax2.set_title(f'Grad-CAM\nPrediction: {class_names[pred_class]}\nConfidence: {prob:.2%}')
    ax2.axis('off')
    
    # Adiciona uma barra de cores modificada para mostrar apenas o intervalo 0.8-1.0
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.jet), 
                       ax=ax2, orientation='vertical', label='Importance')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Configurações
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TEST_DATA_DIR = r"J:\all_animais_phee\firmino_img\exp"
    CHECKPOINT_PATH = "./checkpoints/best_model_classification_resnet.pth"
    OUTPUT_DIR = "./gradcam_visualizations_resnet"
    
    # Cria diretório de saída se não existir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Carrega o modelo
    dataset = datasets.ImageFolder(root=TEST_DATA_DIR)
    CLASS_NAMES = dataset.classes
    num_classes = len(CLASS_NAMES)
    
    model = create_resnet50_model(num_classes=num_classes)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
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
                                     f'gradcam_{os.path.basename(img_path)}')
            try:
                save_gradcam_visualization(model, img_path, save_path, CLASS_NAMES, DEVICE)
            except Exception as e:
                print(f"Erro ao processar {img_path}: {str(e)}")

if __name__ == "__main__":
    main()