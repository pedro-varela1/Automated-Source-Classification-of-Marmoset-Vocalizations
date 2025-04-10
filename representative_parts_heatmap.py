import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
import pandas as pd

def visualize_importance_heatmap(img_np, importance_map, class_name, orig_prob, region_size=16):
    """
    Cria uma visualização com mapa de calor para a importância de todas as regiões
    
    Args:
        img_np: Array numpy da imagem original
        importance_map: Mapa de importância (matriz)
        class_name: Nome da classe
        orig_prob: Probabilidade original
        region_size: Tamanho de cada região
    
    Returns:
        fig: Figura matplotlib com a visualização
    """
    
    # Encontra a região mais importante
    max_row, max_col = np.unravel_index(np.argmax(importance_map), importance_map.shape)
    max_impact = importance_map[max_row, max_col]
    min_row, min_col = np.unravel_index(np.argmin(importance_map), importance_map.shape)
    min_impact = importance_map[min_row, min_col]
    
    # Cria a figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Mostra a imagem original
    ax1.imshow(img_np, cmap='gray')
    ax1.set_title('Imagem Original')
    ax1.axis('off')
    
    # Heatmap
    heatmap_img = np.zeros((160, 160))
    for i in range(len(importance_map)):
        for j in range(len(importance_map[0])):
            y_start = i * region_size
            x_start = j * region_size
            heatmap_img[y_start:y_start+region_size, x_start:x_start+region_size] = importance_map[i, j]
        
    # Show  heatmap
    img_with_heatmap = ax2.imshow(img_np, cmap='gray')
    
    cmap = plt.cm.jet   # Colormap: blue to red
    
    norm = plt.Normalize(vmin=0, vmax=1)    # Normalize range from 0 to 1
    heatmap = ax2.imshow(heatmap_img, cmap=cmap, alpha=0.3, norm=norm)
    
    # Destaca a região mais importante com uma caixa branca
    x_min = max_col * region_size
    y_min = max_row * region_size
    rect = patches.Rectangle((x_min, y_min), region_size, region_size, 
                           linewidth=2, edgecolor='white', facecolor='none')
    ax2.add_patch(rect)
    
    ax2.text(x_min, y_min - 5, f'Impacto: {max_impact:.4f}', 
           color='white', fontsize=5, backgroundcolor='black')
    
    ax2.set_title(f'Mapa de Impacto')
    ax2.axis('off')
    
    # Adiciona uma barra de cores
    cbar = plt.colorbar(heatmap, ax=ax2, orientation='vertical', label='Impacto')
    
    # Adiciona informações sobre a predição original
    plt.suptitle(f'Análise de Impacto para Classe: {class_name} (Prob. Original: {orig_prob:.4f})')
    
    return fig

def main():
    
    INPUT_PATH = "occlusion_results_r10_s100.csv"
    OUTPUT_DIR = "./occlusion_importance_visualizations_heatmap"
    OUTPUT_DIR_MEAN = "./occlusion_importance_visualizations_heatmap_mean"
    MEAN = False

    NUM_REGIONS = int(INPUT_PATH.split("_")[2].split(".")[0][1:])  # 10
    # SAMPLES = int(INPUT_PATH.split("_")[3][1:])  # 50

    # Load dataframe
    df = pd.read_csv(INPUT_PATH)

    if not MEAN:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Loop through each unique "image_path"
        for image_path in df['image_path'].unique():

            # Selct rows for the current image
            image_df = df[df['image_path'] == image_path]

            # Get the fixed values to image as the most frequent
            pred_class = image_df['pred_class'].mode()[0]
            orig_prob = image_df['orig_prob'].mode()[0]
            class_name = image_df['class_name'].mode()[0]

            if pred_class != class_name:
                print(f"Skipping image {image_path} due to class mismatch.")
                continue

            # Load image
            img_np = np.array(
                Image.open(image_path).convert('L').resize((160, 160))
                )
            
            # Create the importance map matrix with "row_region" and "col_region"
            importance_map = np.zeros((NUM_REGIONS, NUM_REGIONS))
            for _, row in image_df.iterrows():
                row_region = row['row_region']
                col_region = row['col_region']
                impact = row['impact']
                importance_map[row_region, col_region] = impact

            # Create the heatmap visualization
            fig = visualize_importance_heatmap(
                img_np, importance_map, class_name, orig_prob, 160//NUM_REGIONS
                )
            
            # Save the figure
            # Create a unique filename
            image_name = os.path.basename(image_path)
            output_filename = os.path.join(OUTPUT_DIR, f"heatmap_{image_name}")
            plt.savefig(output_filename, bbox_inches='tight', dpi=300)
            plt.close(fig) 
                
    else:
        os.makedirs(OUTPUT_DIR_MEAN, exist_ok=True)
        # Get mean impact for each region in the images that have the same class_name
        # Loop through each unique "class_name"
        for class_name in df['class_name'].unique():
            # Select rows for the current class
            class_df = df[df['class_name'] == class_name]
            class_df = class_df[class_df['pred_class'] == class_name]

            # Get a random sample image path for the class
            sample_image_path = class_df['image_path'].sample().values[0]
            img_np = np.array(
                Image.open(sample_image_path).convert('L').resize((160, 160))
                )
            orig_prob = class_df['orig_prob'].mean()    # Use mean probability for the class

            importance_map = np.zeros((len(class_df['image_path'].unique()), NUM_REGIONS, NUM_REGIONS))   # Initialize the importance map
            for images in class_df['image_path'].unique():
                i=0    
                # Select rows for the current image
                image_df = class_df[class_df['image_path'] == images]

                # Create the importance map matrix with "row_region" and "col_region"
                for _, row in image_df.iterrows():
                    row_region = row['row_region']
                    col_region = row['col_region']
                    impact = row['impact']
                    importance_map[i, row_region, col_region] = impact
                i+=1
            # importance map as mean for each region, new shape is (NUM_REGIONS, NUM_REGIONS)
            mean_importance_map = np.mean(importance_map, axis=0)

            # Create the heatmap visualization
            fig = visualize_importance_heatmap(
                img_np, mean_importance_map, class_name, orig_prob, 160//NUM_REGIONS
                )
            
            # Save the figure
            output_filename = os.path.join(OUTPUT_DIR_MEAN, f"mean_heatmap_{class_name}.png")
            plt.savefig(output_filename, bbox_inches='tight', dpi=300)
            plt.close(fig)

if __name__ == "__main__":
    main()