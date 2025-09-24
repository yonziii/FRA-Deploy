from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Define class names (same as in Colab)
CLASS_NAMES = ["AI_GENERATED", "NON_AI_GENERATED"]

# Load model (your Colab logic)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True

# Custom classifier for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_ftrs, 2)
)

model.load_state_dict(torch.load('model/ResNet50_best_model_NewData.pth', map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# Use the same transform as in Colab
inference_transform = weights.transforms()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Copy file to static folder for display
        static_filepath = os.path.join('static', 'uploads', filename)
        if not os.path.exists('static/uploads'):
            os.makedirs('static/uploads')
        import shutil
        shutil.copy2(filepath, static_filepath)

        img = Image.open(filepath).convert("RGB")
        input_tensor = inference_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
            pred_label = CLASS_NAMES[idx.item()]
            confidence = int(conf.item() * 100)  # Convert to percentage

        return render_template('result.html', result=pred_label, confidence=confidence,filename=filename)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5000, debug=True)
