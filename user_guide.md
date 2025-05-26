# Skin Cancer Detection Web App

## 📸 Uploading Images
- Use your phone or desktop to upload an image of a skin lesion.
- The image will be automatically resized and processed.

## 📊 Understanding the Output
- **Prediction**: "Benign" or "Malignant"
- **Confidence Score**: Probability of the prediction (e.g., 92%).

## 💡 Tips for Accurate Results
- Use a clear, well-lit image.
- Crop to focus on the lesion.
- Avoid shadows and blur.

## 🔒 Privacy Note
- Images are not saved or stored.
- Predictions are made in real-time and discarded.

## ⚙️ Running the App
```bash
python train_model.py       # Train the CNN+CTRNN model
python app.py               # Start the Flask server
Visit http://127.0.0.1:5000 in your browser
