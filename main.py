from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
from datetime import datetime

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Blood cell class names
CLASS_NAMES = ['basophil', 'eosinophil', 'erythroblast', 'ig', 
               'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

# Global model variable
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_pytorch_model(model_path):
    """Load the trained PyTorch ResNet-50 model"""
    global model
    try:
        print(f"🔍 Loading model from: {model_path}")
        print(f"📁 File exists: {os.path.exists(model_path)}")
        print(f"📊 File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        # Create ResNet-50 architecture
        print("🏗️ Creating ResNet-50 architecture...")
        model = models.resnet50(weights=None)  # Updated to use weights parameter
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, len(CLASS_NAMES))
        )
        
        # Load the trained weights
        print("📥 Loading checkpoint...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different save formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded model from checkpoint format")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded model from state_dict format")
            
        model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully on {device}")
        print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print(f"🔍 Error type: {type(e).__name__}")
        import traceback
        print(f"📋 Full traceback:\n{traceback.format_exc()}")
        return False

def get_image_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    """Make prediction on uploaded image"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = get_image_transforms()
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class_idx = torch.argmax(outputs, 1).item()
            confidence = probabilities[predicted_class_idx].item()
            
            # Get top 3 predictions
            top3_prob, top3_indices = torch.topk(probabilities, 3)
            top3_predictions = []
            for i in range(3):
                top3_predictions.append({
                    'class': CLASS_NAMES[top3_indices[i].item()],
                    'probability': top3_prob[i].item(),
                    'percentage': top3_prob[i].item() * 100
                })
        
        return {
            'success': True,
            'predicted_class': CLASS_NAMES[predicted_class_idx],
            'confidence': confidence,
            'percentage': confidence * 100,
            'top3_predictions': top3_predictions,
            'all_probabilities': probabilities.cpu().numpy().tolist()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# Routes
@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

# Remove the custom static route since Flask handles it automatically

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, BMP, or TIFF images.'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server configuration.'}), 500
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_image(filepath)
        
        if result['success']:
            # Return JSON response for AJAX
            if request.headers.get('Content-Type') == 'application/json':
                result['image_url'] = url_for('static', filename=f'uploads/{filename}')
                return jsonify(result)
            
            # Return HTML template for regular form submission
            return render_template('result.html', 
                                 image_url=url_for('static', filename=f'uploads/{filename}'),
                                 predicted_class=result['predicted_class'],
                                 confidence=result['confidence'],
                                 percentage=result['percentage'],
                                 top3_predictions=result['top3_predictions'],
                                 class_names=CLASS_NAMES,
                                 all_probabilities=result['all_probabilities'])
        else:
            return jsonify({'error': result['error']}), 500
            
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    return predict()

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'classes': CLASS_NAMES,
        'model_info': f"Model type: {'ResNet50' if model else 'None'}"
    })

@app.route('/reload-model')
def reload_model():
    """Force reload the model"""
    global model
    model_path = 'model/blood_cell_resnet50.pth'
    if os.path.exists(model_path):
        success = load_pytorch_model(model_path)
        return jsonify({
            'success': success,
            'message': 'Model reloaded successfully' if success else 'Failed to reload model',
            'model_loaded': model is not None
        })
    else:
        return jsonify({
            'success': False,
            'message': f'Model file not found: {model_path}',
            'model_loaded': False
        })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    # Check current directory
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📂 Directory contents: {os.listdir('.')}")
    
    # Try to load the model
    model_paths = [
        'model/blood_cell_resnet50.pth',  # Primary model in subfolder
        'blood_cell_resnet50.pth',        # Fallback in root directory
    ]
    
    model_loaded = False
    for model_path in model_paths:
        print(f"🔍 Checking: {model_path}")
        if os.path.exists(model_path):
            print(f"✅ Found model at: {model_path}")
            if load_pytorch_model(model_path):
                model_loaded = True
                break
            else:
                print(f"❌ Failed to load model from: {model_path}")
        else:
            print(f"❌ Model not found at: {model_path}")
    
    if not model_loaded:
        print("⚠️  WARNING: No model loaded!")
        print("   Model paths checked:")
        for path in model_paths:
            exists = "✅ EXISTS" if os.path.exists(path) else "❌ NOT FOUND"
            print(f"   - {path} {exists}")
        print("\n   🔧 To fix this:")
        print("   1. Make sure 'blood_cell_resnet50.pth' is in the 'model' folder")
        print("   2. Check file permissions")
        print("   3. Verify the file is not corrupted")
    else:
        print("✅ Model loaded successfully!")
    
    print(f"\n🚀 Starting Flask app...")
    print(f"📱 Open your browser to: http://localhost:5000")
    print(f"🔧 Device: {device}")
    print(f"📝 Classes: {CLASS_NAMES}")
    print(f"🤖 Model loaded: {'✅ YES' if model else '❌ NO'}")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)