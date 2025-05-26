from flask import Flask, render_template, request, session, redirect, url_for
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from auth import login_required, USERS
from model.cnn_ctrnn_model import CNN_CTRNN

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
    transforms.ToPILImage(),
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
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).unsqueeze(0).to(device)  # [batch, seq_len, C, H, W]

    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "Malignant" if pred == 1 else "Benign"
    return render_template('result.html', label=label, confidence=round(confidence * 100, 2))

if __name__ == '__main__':
    app.run(debug=True)
