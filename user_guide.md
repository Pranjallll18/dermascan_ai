# Skin Cancer Detection Web App

## 🔐 Logging In
- Enter your email address on the login page.
- A **6-digit OTP** (one-time password) will be sent to your email.
- Enter the code on the verification page — no password needed!
- The code expires after **5 minutes**. Use "Resend Code" if it expires.

## 📸 Uploading Images
- Use your phone or desktop to upload an image of a skin lesion.
- You can also use the **live camera** to capture directly.
- The image will be automatically resized and processed.

## 📊 Understanding the Output
- **Diagnosis**: One of 7 skin conditions — Melanoma, Basal Cell Carcinoma, Actinic Keratosis, Benign Keratosis, Dermatofibroma, Melanocytic Nevus, or Vascular Lesion.
- **Risk Level**: High (cancer/precancer) or Low (benign).
- **Confidence Score**: Probability of the top prediction (e.g., 92%).
- **Probability Breakdown**: Confidence for all 7 classes.

## 💡 Tips for Accurate Results
- Use a clear, well-lit image.
- Crop to focus on the lesion.
- Avoid shadows and blur.

## 🔒 Privacy Note
- Images are not saved or stored.
- Predictions are made in real-time and discarded.

## ⚙️ Running the App

### Set up SMTP for email OTP (optional — without this, OTP prints to console):
```bash
set SMTP_EMAIL=your_email@gmail.com
set SMTP_PASSWORD=your_app_password
set SMTP_HOST=smtp.gmail.com
set SMTP_PORT=587
```

### Run:
```bash
python train_model.py       # Train the CNN+CTRNN model (run from model/ dir)
python app.py               # Start the Flask server
```
Visit http://127.0.0.1:5000 in your browser.
