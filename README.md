# 💊 AI in Pharmacy — Smart Pharmaceutical System

AI-assisted pharmaceutical web application combining **Drug Discovery, Tablet Quality Control, and Medication Safety** into one unified platform.

## 🚀 Overview

AI in Pharmacy demonstrates how **Artificial Intelligence, Machine Learning, Computer Vision, and pharmaceutical APIs** can support different stages of the pharmaceutical workflow.

**🔬 Drug Discovery → 💊 Quality Control → 🩺 Medication Safety**

## ✨ Features

### 🔬 Drug Discovery
Disease and molecular-target based compound search using the **PubChem PUG REST API**. The system displays compound name, PubChem CID, molecular formula, molecular weight, Canonical SMILES and Isomeric SMILES.

### 💊 Tablet Quality Control
Real-time tablet detection using **OpenCV** and classification using a **MobileNetV2 Transfer Learning** model. The system analyzes tablet images and provides Normal, Defective, Manual Review or No Tablet Detected results along with visual metrics and confidence.

### 📱 Phone Camera Inspection
Capture a tablet directly using a smartphone camera and send the image to the Flask backend for AI-based inspection.

**Camera → OpenCV → MobileNetV2 → Result → History**

### 🩺 Medication Safety
Uses the **openFDA Drug Label API** to retrieve available drug-label information including drug interactions, warnings, contraindications, boxed warnings and supporting label evidence.

### 📊 History Dashboard
Stores Drug Discovery searches, Tablet Inspections and Medication Safety analyses using **SQLite**, with timestamps, status and analysis details.

## 🧠 Technology Stack

**Frontend:** HTML5, CSS3, JavaScript  
**Backend:** Python, Flask  
**Machine Learning:** TensorFlow, Keras, MobileNetV2  
**Computer Vision:** OpenCV, NumPy  
**Database:** SQLite  
**APIs:** PubChem PUG REST, openFDA  
**Mobile Camera:** Browser Camera API  
**HTTPS Demo:** ngrok

## 🔄 Tablet Inspection Workflow

📷 Camera Image  
↓  
🔍 OpenCV Tablet Detection  
↓  
✂️ Tablet Crop  
↓  
🧠 MobileNetV2  
↓  
🟢 NORMAL / 🔴 DEFECTIVE  
↓  
📊 Confidence & Visual Metrics  
↓  
🗄️ SQLite History

## 📁 Project Structure

AI-Pharmacy/  
├── app.py  
├── train_model.py  
├── tablet_model.keras  
├── class_names.txt  
├── dataset/  
├── templates/  
│   ├── index.html  
│   ├── drug_discovery.html  
│   ├── tablet_inspection.html  
│   ├── medication_safety.html  
│   ├── history.html  
│   └── mobile_camera.html  
├── static/  
│   └── style.css  
├── .gitignore  
└── README.md

## ⚙️ Installation & Setup

```bash
git clone https://github.com/YOUR_USERNAME/AI-Pharmacy.git
cd AI-Pharmacy
python -m venv venv

Windows
venv\Scripts\activate
Install Dependencies
pip install flask opencv-python numpy tensorflow pillow requests
Run Application
python app.py

Open:

http://127.0.0.1:5000

📱 Phone Camera

For smartphone camera access:

ngrok http 5000

Open the generated HTTPS URL on your phone and navigate to:

/mobile-camera

🌱 Future Scope
Real-world large-scale tablet datasets
Multi-class pharmaceutical defect detection
OCR-based medicine identification
Advanced drug interaction analysis
Explainable AI
Cloud deployment
Pharmacist dashboard
Advanced drug candidate prediction
