from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'your-secret-key-123'

# Configuration
CLASSES = ['Coccidiosis', 'Healthy', 'Newcastle', 'Salmonella']
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Disease-specific information for display
DISEASE_INFO = {
    'Coccidiosis': {
        'name': 'Coccidiosis',
        'colorClass': 'coccidiosis',
        'bgClass': 'coccidiosis-bg',
        'symptoms': [
            'Mucosal shedding',
            'Watery or loose droppings'
        ]
    },
    'Healthy': {
        'name': 'Healthy',
        'colorClass': 'healthy',
        'bgClass': 'healthy-bg',
        'symptoms': [
            'Firm and well-formed',
            'Dark brown with white cap',
            'No blood or mucus'
        ]
    },
    'Newcastle': {
        'name': 'Newcastle Disease',
        'colorClass': 'newcastle',
        'bgClass': 'newcastle-bg',
        'symptoms': [
            'Greenish, watery diarrhea'
        ]
    },
    'Salmonella': {
        'name': 'Salmonellosis',
        'colorClass': 'salmonella',
        'bgClass': 'salmonella-bg',
        'symptoms': [
            'Uneven texture with mucus',
            'Frothy appearance'
        ]
    }
}

def load_safe_model(model_path='model.h5'):
    """Load the pre-trained model with error handling."""
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = load_model(model_path)
        if model.output_shape[1] != len(CLASSES):
            raise ValueError(f"Model expects {model.output_shape[1]} classes, but {len(CLASSES)} defined")
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return None

model = load_safe_model()

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    """Predict the disease from an image."""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        preds = model.predict(img_array)[0]
        class_idx = np.argmax(preds)
        predicted_class = CLASSES[class_idx]
        
        return {
            'name': DISEASE_INFO[predicted_class]['name'],
            'confidence': f"{float(preds[class_idx] * 100):.2f}%",
            'colorClass': DISEASE_INFO[predicted_class]['colorClass'],
            'bgClass': DISEASE_INFO[predicted_class]['bgClass'],
            'symptoms': DISEASE_INFO[predicted_class]['symptoms']
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return None

@app.route('/', methods=['GET', 'POST'])
def home():
    """Handle GET and POST requests for the main page."""
    if model is None:
        return render_template('index.html', error="Model failed to load. Please check the server configuration.")

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file selected")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(save_path)
                if not os.path.exists(save_path):
                    return render_template('index.html', error="Failed to save uploaded file")
                
                result = predict_image(save_path)
                if result is None:
                    return render_template('index.html', error="Failed to process image", filename=filename)
                
                image_path = f"/{save_path.replace(os.sep, '/')}"
                return render_template('index.html',
                                    filename=filename,
                                    image_path=image_path,
                                    result=result,
                                    loading=False)
            except Exception as e:
                logger.error(f"File upload failed: {str(e)}")
                return render_template('index.html', error=f"Upload failed: {str(e)}")
        
        return render_template('index.html', error="Allowed file types: PNG, JPG, JPEG")
    
    return render_template('index.html', loading=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
