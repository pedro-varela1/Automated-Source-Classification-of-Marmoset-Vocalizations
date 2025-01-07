from flask import Flask, request, send_file, jsonify, render_template
import torch
from werkzeug.utils import secure_filename
import os
from inceptionResnetV1 import InceptionResnetV1
import zipfile
from io import BytesIO
import numpy as np
from preprocessing import DataPreparation
from dataloader import PredictionDataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader


app = Flask(__name__)

dp = DataPreparation()  # Initialize the DataPreparation class

# Configurações
UPLOAD_FOLDER = 'uploads'
ALLOWED_AUDIO_EXTENSIONS = {'wav'}
ALLOWED_CSV_EXTENSIONS = {'csv'}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = "./checkpoints/best_model_classification.pth"
CLASS_NAMES = ["Blue2A", "Blue2C", "Blue3A", "Blue3B", "Blue3C",
               "Pink2C", "Pink3A", "Pink3B", "Pink3C"]
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 128
NUM_WORKERS = 0

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

def allowed_csv_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_CSV_EXTENSIONS

def load_model(checkpoint_path, model, device):
    """
    Carrega o modelo treinado a partir do checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
    return model.eval()

def preprocess_audio(audio_path, time_segments):
    """
    Função que será implementada para processar o áudio e gerar espectrogramas
    
    Args:
        audio_path: Caminho para o arquivo de áudio
        time_segments: DataFrame com os tempos de início e fim
    
    Returns:
        List de caminhos para as imagens dos espectrogramas gerados
    """
    pass

@torch.no_grad()
def classify_spectrograms(model, temp_spec_folder, device):
    """
    Classifica os espectrogramas gerados

    Args:
        model: Modelo treinado
        temp_spec_folder: Pasta com os espectrogramas
        device: Dispositivo onde o modelo está
    
    Returns:
        Dicionário com os índices sendo o caminho da imagem  e os valores sendo um dicionário com a predição e a confiança
    """
    dataset = PredictionDataset(temp_spec_folder)
    classification_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    softmax = torch.nn.Softmax(dim=1)

    # Gerar as predições
    predictions_index = {}
    for inputs, batch_paths in classification_loader:
        inputs = inputs.to(device)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        probs = softmax(outputs)

        for i, path in enumerate(batch_paths):
            predictions_index[path] = {
                'prediction': CLASS_NAMES[predictions[i].item()],
                'confidence': probs[i][predictions[i]].item()
            }

    return predictions_index


def update_csv(csv_file, predictions, paths_index):
    """
    Atualiza o arquivo CSV com as predições
    
    Args:
        csv_file: Caminho para o arquivo CSV
        predictions: Dicionário com as predições
        paths_index: Dicionário com os índices sendo o caminho da imagem e os valores sendo o índice
    """
    df = pd.read_csv(csv_file)
    df['prediction'] = np.nan
    df['confidence'] = np.nan
    df['base64_image'] = np.nan

    for index, row in df.iterrows():
        if str(index) in paths_index:
            image_path = paths_index[str(index)]['image_path']
            if image_path:
                df.at[index, 'prediction'] = predictions[image_path]['prediction']
                df.at[index, 'confidence'] = predictions[image_path]['confidence']
                df.at[index, 'base64_image'] = paths_index[str(index)]['base64_image']
    df.to_csv(csv_file, index=False)

@app.route('/classify', methods=['POST'])
def classify_vocalizations():
    # Check if both files are present in request
    if 'audio' not in request.files or 'csv' not in request.files:
        return jsonify({'error': 'Missing audio or CSV file'}), 400
    
    audio_file = request.files['audio']
    csv_file = request.files['csv']
    
    # Check if files are valid
    if not allowed_audio_file(audio_file.filename):
        return jsonify({'error': 'Invalid audio file format'}), 400
    if not allowed_csv_file(csv_file.filename):
        return jsonify({'error': 'Invalid CSV file format'}), 400
    
    # Create temporary paths for uploaded files
    audio_path = os.path.join(UPLOAD_FOLDER, secure_filename(audio_file.filename))
    csv_path = os.path.join(UPLOAD_FOLDER, secure_filename(csv_file.filename))
    
    # Save uploaded files
    audio_file.save(audio_path)
    csv_file.save(csv_path)
    
    try:
            # Create temporary folder for spectrograms
            temp_spec_folder = os.path.join(UPLOAD_FOLDER, 'spectrograms')
            os.makedirs(temp_spec_folder, exist_ok=True)
    
            # Generate spectrograms
            paths_index = dp.transform_data(csv_path, audio_path, temp_spec_folder)
            
            # Load and prepare model
            try:
                model = InceptionResnetV1(device=DEVICE, classify=True, num_classes=NUM_CLASSES)
                model = load_model(CHECKPOINT_PATH, model, DEVICE)
                model = model.to(DEVICE)
            except Exception as e:
                print(f"Error loading model: {e}")
                return jsonify({'error': 'Error loading model'}), 500
            
            # Get predictions
            try:
                predictions = classify_spectrograms(model, temp_spec_folder, DEVICE)
            except Exception as e:
                print(f"Error classifying spectrograms: {e}")
                return jsonify({'error': 'Error classifying spectrograms'}), 500
            try:
                # Update CSV file with predictions
                update_csv(csv_path, predictions, paths_index)
            except Exception as e:
                print(f"Error updating CSV file: {e}")
                return jsonify({'error': 'Error updating CSV file'}), 500
            
            # Send updated CSV file
            memory_file = BytesIO()
            with zipfile.ZipFile(memory_file, 'w') as zf:
                zf.write(csv_path, os.path.basename(csv_path))
                # Add all images in spectrogram folder to zip file
                for root, _, files in os.walk(temp_spec_folder):
                    for file in files:
                        zf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_spec_folder))
            memory_file.seek(0)
                
            
            # Clean up temporary files
            os.remove(audio_path)
            os.remove(csv_path)
            if os.path.exists(temp_spec_folder):
                import shutil
                shutil.rmtree(temp_spec_folder)
            
            return send_file(
                memory_file,
                as_attachment=True,
                download_name='predictions.zip'
                )
    
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(csv_path):
            os.remove(csv_path)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)