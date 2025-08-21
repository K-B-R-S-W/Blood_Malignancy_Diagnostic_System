# 🩸 Blood Malignancy Diagnostic System
---
<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)
![Flask](https://img.shields.io/badge/Flask-3.1.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

**Advanced AI-powered blood cell classification system using ResNet-50 deep learning architecture**



</div>

---

## 📖 Overview

The **Blood Malignancy Diagnostic System** is a state-of-the-art machine learning application that leverages deep learning to classify blood cells with high accuracy. Built using PyTorch's ResNet-50 architecture, this system can identify 8 different types of blood cells, making it a valuable tool for medical professionals, researchers, and educational institutions.

### 🎯 Key Highlights

- **95%+ Accuracy** on test dataset
- **8 Blood Cell Types** classification
- **Real-time Processing** with GPU acceleration
- **Professional Web Interface** with modern UI/UX
- **Comprehensive Analysis** with confidence scores and probability distributions
- **Medical Information** database for educational purposes

---

## 🔬 Supported Blood Cell Types

| Cell Type | Description | Normal Range |
|-----------|-------------|--------------|
| **Basophil** | Least common white blood cells, role in allergic reactions | 0-1% of WBC |
| **Eosinophil** | Combat parasitic infections and allergic responses | 1-4% of WBC |
| **Erythroblast** | Immature red blood cells, normally in bone marrow | Not in peripheral blood |
| **Immature Granulocyte (IG)** | Early forms of neutrophils, eosinophils, basophils | 0-2% of WBC |
| **Lymphocyte** | Key components of adaptive immune system | 20-40% of WBC |
| **Monocyte** | Large white blood cells, become macrophages | 2-8% of WBC |
| **Neutrophil** | Most abundant white blood cells, first responders | 50-70% of WBC |
| **Platelet** | Cell fragments crucial for blood clotting | 150K-450K per μL |

---

## ✨ Features

### 🤖 AI-Powered Classification
- **ResNet-50 Architecture** - Deep convolutional neural network
- **Transfer Learning** - Pre-trained weights with custom classification head
- **CUDA Support** - GPU acceleration for faster processing
- **Batch Processing** - Efficient image preprocessing pipeline

### 🌐 Modern Web Interface
- **Drag & Drop Upload** - Intuitive file upload system
- **Real-time Analysis** - Instant predictions with progress indicators
- **Interactive Results** - Confidence scores and probability distributions
- **Responsive Design** - Works on desktop, tablet, and mobile devices

### 📊 Comprehensive Analysis
- **Top-3 Predictions** - Multiple classification options with confidence
- **Probability Charts** - Visual representation using Chart.js
- **Medical Information** - Educational content about each cell type
- **Professional Reports** - Detailed analysis with disclaimers

### 🔧 Developer Features
- **RESTful API** - `/api/predict` endpoint for integration
- **Health Monitoring** - `/health` endpoint for system status
- **Error Handling** - Comprehensive error management and logging
- **Modular Design** - Clean, maintainable code structure

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+**
- **CUDA-compatible GPU** (recommended)
- **Conda** or **virtualenv**

### Step 1: Clone the Repository

```bash
git clone https://github.com/K-B-R-S-W/blood-malignancy-diagnostic-system.git
cd blood-malignancy-diagnostic-system
```

### Step 2: Create Environment

```bash
# Using Conda (recommended)
conda create -n blood-cell-ai python=3.9
conda activate blood-cell-ai

# Or using venv
python -m venv blood-cell-ai
# Windows
blood-cell-ai\Scripts\activate
# Linux/Mac
source blood-cell-ai/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Model

Ensure the trained model `blood_cell_resnet50.pth` is in the `model/` directory.

---

## 🔧 Configuration

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | GTX 1060 | RTX 3070+ |
| **Storage** | 2 GB | 5 GB+ |
| **CPU** | Intel i5 | Intel i7+ |

### Environment Variables

Create a `.env` file for configuration:

```env
FLASK_ENV=production
MODEL_PATH=model/blood_cell_resnet50.pth
UPLOAD_FOLDER=static/uploads
MAX_CONTENT_LENGTH=16777216
DEBUG=False
```

---

## 🚀 Usage

### Quick Start

#### Option 1: Using Batch File (Windows)
```batch
run_with_conda.bat
```

#### Option 2: Using PowerShell
```powershell
.\run_app.ps1
```

#### Option 3: Direct Python Execution
```bash
# Activate environment first
conda activate blood-cell-ai
python main.py
```

### Web Interface

1. Open browser to `http://localhost:5000`
2. Upload a blood cell image (PNG, JPG, JPEG, BMP, TIFF)
3. Click "Analyze Image"
4. View comprehensive results with confidence scores

### API Usage

```python
import requests

# Upload image for prediction
with open('blood_cell_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/predict', files=files)
    result = response.json()
    
print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Health Check

```bash
curl http://localhost:5000/health
```

---

## 📊 Model Performance

### Training Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 95.2% |
| **Training Time** | ~3 hours |
| **Model Size** | 90.05 MB |
| **Parameters** | 23.5M |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Basophil | 0.94 | 0.92 | 0.93 |
| Eosinophil | 0.96 | 0.95 | 0.95 |
| Erythroblast | 0.93 | 0.94 | 0.94 |
| IG | 0.91 | 0.89 | 0.90 |
| Lymphocyte | 0.97 | 0.98 | 0.97 |
| Monocyte | 0.95 | 0.93 | 0.94 |
| Neutrophil | 0.98 | 0.97 | 0.97 |
| Platelet | 0.96 | 0.97 | 0.96 |

---

## 📁 Project Structure

```
blood-malignancy-diagnostic-system/
├── 📁 model/                          # Trained model files
│   ├── blood_cell_resnet50.pth       # Main trained model
│   └── blood_cell_results.pth        # Training results & metrics
├── 📁 static/                         
│   ├── 📁 images/                    
│   └── 📁 uploads/                   
├── 📁 templates/                      
│   ├── index.html                     
│   ├── result.html                    
│   └── error.html                     
├── 📄 main.py                        
├── 📄 requirements.txt               
├── 📄 ResNet50_Model_Training.ipynb   
├── 📄 extract_results.py              
├── 📄 simple_extract.py               
├── 📄 run_with_conda.bat              
├── 📄 run_app.ps1                     
└── 📄 README.md                       
```

---

## 🔬 Technical Details

### Model Architecture

```python
ResNet50(
  (fc): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=2048, out_features=8, bias=True)
  )
)
```

### Data Preprocessing

- **Image Resizing**: 224×224 pixels
- **Normalization**: ImageNet statistics
- **Data Augmentation**: 
  - Random horizontal/vertical flips
  - Random rotation (±30°)
  - Color jittering
  - Random affine transformations

### Training Configuration

- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Loss Function**: CrossEntropyLoss with class weights
- **Scheduler**: StepLR (step_size=15, gamma=0.1)
- **Batch Size**: 32
- **Epochs**: 50

---

## 📈 Results Visualization

### Extract Training Results

Generate comprehensive visualizations of training results:

```bash
# Full visualization suite
python extract_results.py

# Simple extraction
python simple_extract.py
```

### Generated Visualizations

- 📊 Training/Validation curves
- 🎯 Confusion matrices
- 📈 Per-class accuracy charts
- 📋 Confidence distributions
- 📄 Classification reports

---

## 🔒 Medical Disclaimer

> **⚠️ IMPORTANT MEDICAL DISCLAIMER**
> 
> This AI system is designed for **educational and research purposes only**. It should **NOT** be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical decisions. The developers are not responsible for any medical decisions made based on this system's output.

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create feature branch** 
3. **Commit changes** 
4. **Push to branch** 
5. **Open Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black main.py
flake8 main.py
```

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

### Acknowledgments

- PyTorch team for the deep learning framework
- Flask community for the web framework  
- Medical professionals who provided domain expertise
- Open source community for various tools and libraries

---

## 📞 Contact & Support

### Get Help

- 📧 **Email**: support@bloodcellai.com
- 💬 **Issues**: [GitHub Issues](https://github.com/K-B-R-S-W/blood-malignancy-diagnostic-system/issues)
- 📖 **Documentation**: [Wiki](https://github.com/K-B-R-S-W/blood-malignancy-diagnostic-system/wiki)

### Citation

If you use this work in your research, please cite:

```bibtex
@software{blood_cell_ai_2025,
  title={Blood Malignancy Diagnostic System},
  author={K-B-R-S-W},
  year={2025},
  url={https://github.com/K-B-R-S-W/blood-malignancy-diagnostic-system}
}
```

---

## 📮 Support

**📧 Email:** [k.b.ravindusankalpaac@gmail.com](mailto:k.b.ravindusankalpaac@gmail.com)  
**🐞 Bug Reports:** [GitHub Issues](https://github.com/K-B-R-S-W/Blood_Malignancy_Diagnostic_System/issues)  
**📚 Documentation:** [Project Wiki](https://github.com/K-B-R-S-W/Blood_Malignancy_Diagnostic_System/wiki)  
**💭 Discussions:** [GitHub Discussions](https://github.com/K-B-R-S-W/Blood_Malignancy_Diagnostic_System/discussions)  

---

## ⭐ Support This Project
If you find this project helpful, please give it a **⭐ star** on GitHub — it motivates me to keep improving! 🚀
