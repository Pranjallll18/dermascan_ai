from flask import Flask, render_template, request, session, redirect, url_for
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from auth import login_required, USERS
from model.cnn_ctrnn_model import CNN_CTRNN
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)
app.secret_key = 'secret123'

# Load PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_CTRNN()
model.load_state_dict(torch.load('model/model/skin_cancer_model.pth', map_location=device))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['username']
        pwd = request.form['password']
        if USERS.get(user) == pwd:
            session['username'] = user
            return redirect(url_for('index'))
        return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    img = None

    # Check if base64 image from camera is present
    base64_image = request.form.get('capturedImage')
    if base64_image:
        try:
            header, encoded = base64_image.split(',', 1)
            decoded = base64.b64decode(encoded)
            img = Image.open(BytesIO(decoded)).convert('RGB')
        except Exception as e:
            print("Error decoding base64 image:", e)
            return render_template('result.html', label="Invalid image data", confidence=0)

    # Else check if file upload (gallery)
    elif 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            npimg = np.frombuffer(file.read(), np.uint8)
            img_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

    if img is None:
        return render_template('result.html', label="No image received", confidence=0)

    # Preprocess and predict
    img_tensor = transform(img).unsqueeze(0).unsqueeze(0).to(device)  # [batch, seq_len, C, H, W]

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "Malignant" if pred == 1 else "Benign"
    return render_template('result.html', label=label, confidence=round(confidence * 100, 2))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
