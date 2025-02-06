from inceptionResnetV1 import InceptionResnetV1
import torchvision.datasets as datasets
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os
from tqdm import tqdm
import random
import matplotlib.colors as mcolors

class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        """
        Args:
            model: Modelo InceptionResnetV1 treinado
            target_layer: Camada alvo para visualização (ex: 'block8')
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Registra hooks para capturar gradientes e ativações
        target = self._get_target_layer()
        target.register_forward_hook(self._save_activation)
        target.register_backward_hook(self._save_gradient)
    
    def _get_target_layer(self):
        """Obtém a camada alvo do modelo"""
        return self.model.block8
    
    def _save_activation(self, module, input, output):
        self.activations = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Gera o mapa de ativação Grad-CAM++
        
        Args:
            input_tensor: Tensor de entrada (1, C, H, W)
            target_class: Classe alvo para visualização (opcional)
        
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
        
        gradients = self.gradients
        activations = self.activations
        
        b, k, u, v = gradients.size()
        
        alpha_num = gradients.pow(2)
        alpha_denom = alpha_num.mul(2) + \
                     activations.mul(gradients.pow(3)).sum(dim=[2, 3], keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
        
        alpha = alpha_num.div(alpha_denom + 1e-7)
        positive_gradients = F.relu(class_score.exp() * gradients)
        weights = (positive_gradients * alpha).sum(dim=[2, 3], keepdim=True)
        
        # Gera o mapa de ativação
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        # Normalização e redimensionamento para 160x160
        cam = F.interpolate(cam, size=(160, 160), mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.squeeze().cpu().detach().numpy()

# def apply_colormap(cam, img_array):
#     """
#     Aplica o mapa de calor sobre a imagem original usando matplotlib
    
#     Args:
#         cam: Mapa de ativação (160, 160)
#         img_array: Array numpy da imagem original (160, 160)
    
#     Returns:
#         visualization: Imagem com mapa de calor sobreposto
#     """
#     # Normaliza a imagem original para [0, 1]
#     img_norm = (img_array - img_array.min()) / (img_array.max() - img_array.min())
    
#     # Cria uma máscara para valores entre 0.8 e 1.0
#     mask = (cam >= 0.8)
    
#     # Renormaliza os valores dentro do intervalo [0.8, 1.0] para [0, 1]
#     cam_thresholded = np.zeros_like(cam)
#     cam_thresholded[mask] = (cam[mask] - 0.8) / 0.2  # Normaliza o intervalo [0.8, 1.0] para [0, 1]
    
#     # Cria um mapa de cores personalizado
#     cmap = plt.cm.jet
#     colored_cam = cmap(cam_thresholded)
    
#     # Expande a imagem em escala de cinza para 3 canais
#     img_rgb = np.stack([img_norm] * 3, axis=-1)
    
#     # Cria a visualização final com transparência
#     visualization = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 3))
    
#     # Aplica o mapa de calor apenas onde a máscara é True
#     visualization[mask] = (0.3 * img_rgb[mask] + 0.7 * colored_cam[mask, :3])
#     visualization[~mask] = img_rgb[~mask]
    
#     return visualization.clip(0, 1)

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
    colored_cam = cmap(cam)[:, :, :3]
    
    # Expande a imagem em escala de cinza para 3 canais
    img_rgb = np.stack([img_norm] * 3, axis=-1)
    
    # Combina a imagem original com o mapa de calor
    visualization = (0.3 * img_rgb + 0.7 * colored_cam).clip(0, 1)
    
    return visualization

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
    
    # Inicializa Grad-CAM++
    grad_cam = GradCAMPlusPlus(model, target_layer='block8')
    
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
    ax2.set_title(f'Grad-CAM++\nPrediction: {class_names[pred_class]}\nConfidence: {prob:.2%}')
    ax2.axis('off')
    
    # Adiciona uma barra de cores normalizada
    norm = mcolors.Normalize(vmin=0, vmax=1.0)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet), 
                       ax=ax2, orientation='vertical', label='Importance')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Configurações
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TEST_DATA_DIR = r"J:\all_animais_phee\firmino_img\exp"
    CHECKPOINT_PATH = "./checkpoints/best_model_classification.pth"
    OUTPUT_DIR = f"./gradcam_visualizations_block8"
    
    # Cria diretório de saída se não existir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Carrega o modelo
    dataset = datasets.ImageFolder(root=TEST_DATA_DIR)
    CLASS_NAMES = dataset.classes
    num_classes = len(CLASS_NAMES)
    
    model = InceptionResnetV1(device=DEVICE, classify=True, num_classes=num_classes)
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