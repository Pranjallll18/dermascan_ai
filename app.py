from flask import Flask, render_template, request, session, redirect, url_for
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from auth import login_required, generate_otp, store_otp, send_otp_email, verify_otp
from model.cnn_ctrnn_model import CNN_CTRNN
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)
app.secret_key = 'secret123'

# --------- Class Metadata (must match training order) ---------
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
NUM_CLASSES = len(CLASS_NAMES)

CLASS_INFO = {
    'akiec': {
        'name': 'Actinic Keratosis',
        'risk': 'High',
        'description': 'A rough, scaly patch caused by years of sun exposure. Considered precancerous — it can develop into squamous cell carcinoma if untreated.',
        'color': '#e74c3c',
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'risk': 'High',
        'description': 'The most common type of skin cancer. It grows slowly and rarely spreads, but can cause significant local tissue damage.',
        'color': '#e74c3c',
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'risk': 'Low',
        'description': 'Non-cancerous skin growths such as seborrheic keratoses or solar lentigines. Generally harmless and do not require treatment.',
        'color': '#27ae60',
    },
    'df': {
        'name': 'Dermatofibroma',
        'risk': 'Low',
        'description': 'A common, harmless fibrous nodule typically found on the legs. Usually requires no treatment.',
        'color': '#27ae60',
    },
    'mel': {
        'name': 'Melanoma',
        'risk': 'High',
        'description': 'The most dangerous form of skin cancer. It can spread rapidly to other organs if not caught early. Immediate medical attention is essential.',
        'color': '#c0392b',
    },
    'nv': {
        'name': 'Melanocytic Nevus',
        'risk': 'Low',
        'description': 'A common mole — a benign growth of pigment-producing cells (melanocytes). Most moles are harmless.',
        'color': '#27ae60',
    },
    'vasc': {
        'name': 'Vascular Lesion',
        'risk': 'Low',
        'description': 'Benign growths made of blood vessels, including cherry angiomas and angiokeratomas. Typically harmless.',
        'color': '#27ae60',
    },
}

# --------- Load PyTorch model ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_CTRNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load('model/model/skin_cancer_model.pth', map_location=device))
model.to(device)
model.eval()

# --------- Image preprocessing (matches training normalization) ---------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_email' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        if not email or '@' not in email:
            return render_template('login.html', error="Please enter a valid email address.")

        otp = generate_otp()
        store_otp(email, otp)

        if send_otp_email(email, otp):
            session['pending_email'] = email
            return redirect(url_for('verify'))
        else:
            return render_template('login.html', error="Failed to send OTP. Please try again.")

    return render_template('login.html')

@app.route('/verify-otp', methods=['GET', 'POST'])
def verify():
    email = session.get('pending_email')
    if not email:
        return redirect(url_for('login'))

    if request.method == 'POST':
        entered_otp = request.form.get('otp', '').strip()
        success, error_msg = verify_otp(email, entered_otp)

        if success:
            session.pop('pending_email', None)
            session['user_email'] = email
            return redirect(url_for('index'))
        else:
            return render_template('verify_otp.html', email=email, error=error_msg)

    return render_template('verify_otp.html', email=email)

@app.route('/resend-otp')
def resend_otp():
    email = session.get('pending_email')
    if not email:
        return redirect(url_for('login'))

    otp = generate_otp()
    store_otp(email, otp)
    send_otp_email(email, otp)
    return redirect(url_for('verify'))

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    session.pop('pending_email', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    img = None
    image_data = None

    # Check if base64 image from camera is present
    base64_image = request.form.get('capturedImage')
    if base64_image:
        try:
            header, encoded = base64_image.split(',', 1)
            decoded = base64.b64decode(encoded)
            img = Image.open(BytesIO(decoded)).convert('RGB')
            image_data = base64_image  # Pass to template for display
        except Exception as e:
            print("Error decoding base64 image:", e)
            return render_template('result.html',
                                   diagnosis_code='error',
                                   diagnosis_name='Invalid Image',
                                   risk_level='Unknown',
                                   description='Could not decode the provided image.',
                                   confidence=0,
                                   color='#95a5a6',
                                   all_probs=[])

    # Else check if file upload (gallery)
    elif 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            file_bytes = file.read()
            npimg = np.frombuffer(file_bytes, np.uint8)
            img_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
            # Convert to base64 for display on result page
            encoded = base64.b64encode(file_bytes).decode('utf-8')
            image_data = f"data:image/jpeg;base64,{encoded}"

    if img is None:
        return render_template('result.html',
                               diagnosis_code='error',
                               diagnosis_name='No Image Received',
                               risk_level='Unknown',
                               description='Please upload or capture an image to analyze.',
                               confidence=0,
                               color='#95a5a6',
                               all_probs=[])

    # ---- Build multi-view sequence for CTRNN (matches training seq_len=4) ----
    # View 0: base transform
    view_base = transform(img)

    # View 1: horizontal flip
    tta_hflip = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # View 2: vertical flip
    tta_vflip = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # View 3: both flips
    tta_both = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    views = torch.stack([view_base, tta_hflip(img), tta_vflip(img), tta_both(img)])
    img_tensor = views.unsqueeze(0).to(device)  # [1, 4, C, H, W]

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    # Build per-class probability list for the result page
    all_probs = []
    for i, cls_code in enumerate(CLASS_NAMES):
        info = CLASS_INFO[cls_code]
        all_probs.append({
            'code': cls_code,
            'name': info['name'],
            'prob': round(probs[0][i].item() * 100, 2),
            'color': info['color'],
            'risk': info['risk'],
        })
    # Sort by probability descending
    all_probs.sort(key=lambda x: x['prob'], reverse=True)

    predicted_class = CLASS_NAMES[pred]
    info = CLASS_INFO[predicted_class]

    return render_template('result.html',
                           diagnosis_code=predicted_class,
                           diagnosis_name=info['name'],
                           risk_level=info['risk'],
                           description=info['description'],
                           confidence=round(confidence * 100, 2),
                           color=info['color'],
                           all_probs=all_probs,
                           image_data=image_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
