# 🛡️ IMPROVED HYBRID CNN–BiLSTM–ATTENTION WITH FOCAL LOSS FOR ROBUST CREDIT CARD FRAUD DETECTION

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![Twilio](https://img.shields.io/badge/Twilio-SMS-green)


---
# 🎥 Live Demo

<video 
    src="demo video(1).mp4"
    controls
    autoplay
    muted
    loop
    width="100%">
</video>

# 📌 Project Overview

This project presents an advanced hybrid deep learning framework for detecting fraudulent credit card transactions with high accuracy and reliability.

The proposed system combines:

- **CNN (Convolutional Neural Networks)** for spatial feature extraction
- **BiLSTM (Bidirectional Long Short-Term Memory)** for temporal sequence learning
- **Attention Mechanism** for focusing on important transaction patterns
- **Focal Loss with Class Weights** for handling highly imbalanced fraud datasets

The system is enhanced with:

- 📲 Real-time Twilio SMS alerts
- 📊 Interactive Streamlit dashboard
- 🧠 SHAP explainability
- 📈 Visualization & monitoring tools

This project was developed as a collaborative academic research project by the **Department of Artificial Intelligence and Data Science** during the academic year **2025–2026**.

---

# 👨‍💻 Team Details

## 📌 Project Title

**Improved Hybrid CNN–BiLSTM–Attention with Focal Loss for Robust Credit Card Fraud Detection**

## 🏫 Department

Department of Artificial Intelligence and Data Science

## 📅 Academic Year

2025 – 2026

## 👥 Team Members

- **Devalekka P** 
- **Keerthi A**
- **Sahana M**
  
---

# 📖 Abstract

Credit card fraud causes billions of dollars in annual financial losses worldwide and continues to evolve with increasingly sophisticated attack patterns. Traditional machine learning and rule-based systems often fail to capture complex transaction behavior, temporal dependencies, and severe class imbalance.

To overcome these limitations, this project proposes a hybrid deep learning architecture integrating CNN, BiLSTM, and Attention mechanisms with Focal Loss optimization. The model improves fraud recall while maintaining strong precision and robustness.

The proposed system also integrates:

- SHAP explainability
- Attention visualization
- Real-time fraud alerts using Twilio SMS
- Interactive fraud analytics dashboard

This makes the solution suitable for real-world banking and financial security applications.

---

# 🎯 Objectives

- Develop a hybrid deep learning model combining CNN, BiLSTM, and Attention mechanisms
- Handle severe class imbalance using Focal Loss and class-weighted training
- Capture both spatial and temporal transaction patterns
- Improve fraud detection accuracy and robustness
- Support explainable AI through SHAP analysis
- Enable real-time fraud notification through SMS alerts
- Build a scalable dashboard for monitoring and prediction

---

# ✨ Key Features

✅ Hybrid Deep Learning Architecture  
✅ CNN + BiLSTM + Attention Integration  
✅ Focal Loss Optimization  
✅ Real-time Twilio SMS Alerts  
✅ SHAP Explainable AI Integration  
✅ Interactive Streamlit Dashboard  
✅ ROC-AUC & Confusion Matrix Analysis  
✅ Multiple Dataset Support  
✅ Fraud Alert Logging System  
✅ Time-based Sequential Transaction Analysis  
✅ Plotly Interactive Visualizations  
✅ Customer Phone Mapping Support  

---

# 🧠 Proposed Model Architecture

```text
Input Transaction Data
            ↓
      Data Preprocessing
(Cleaning, Normalization)
            ↓
      Sequence Creation
(Time-based Grouping)
            ↓
        CNN Layer
(Feature Extraction)
            ↓
      BiLSTM Layer
(Temporal Learning)
            ↓
   Attention Mechanism
(Focus Important Features)
            ↓
       Dense Layers
(Classification)
            ↓
      Output Layer
(Sigmoid Probability)
            ↓
     Fraud Decision
            ↓
SMS Alert / Monitoring
```

---

# 🔍 Why This Model?

Traditional fraud detection systems struggle because:

- Fraud transactions are extremely rare
- Fraud patterns continuously evolve
- Sequential transaction behavior matters
- Explainability is important in banking systems

Our hybrid architecture solves these problems by:

- Learning both spatial and temporal transaction patterns
- Focusing on important fraudulent behavior using attention
- Improving minority class learning using focal loss
- Providing explainable predictions using SHAP

---

# ☁️ Model Training Environment
use this 
EUROPEAN & INDIAN DATASET TRAINING.ipynb

The deep learning model was trained using **Google Colab** with GPU acceleration for faster computation and efficient training.

### Training Environment

- Platform: Google Colab
- Framework: TensorFlow / Keras
- GPU Support: NVIDIA Tesla T4
- Python Version: 3.10
- Model Storage: Google Drive

### Google Drive Saved Model Path

```python
/content/drive/MyDrive/Fraud_Detection_Project/models/
```

### Training Notebook

The complete model training, preprocessing, evaluation, and visualization code is available in:

```text
EUROPEAN & INDIAN DATASET TRAINING.ipynb
```

### Saved Files

The following files are saved in Google Drive after training:

```text
models/
├── hybrid_cnn_bilstm_attention_model.h5
├── tokenizer.pkl
├── scaler.pkl
└── label_encoder.pkl
```

This notebook includes:

- Data preprocessing
- Feature engineering
- CNN–BiLSTM–Attention model implementation
- Focal Loss integration
- Model evaluation
- SHAP explainability
- Performance visualization
---


# 📁 Project Structure in vs code side

```text
project/
├── app.py
├── sms_alerts.py
├── phone_mapping.csv
├── requirements.txt
├── saved/
├── .env
└── README.md
```

---

# 📂 File Descriptions

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit dashboard application |
| `sms_alerts.py` | Twilio SMS integration module |
| `phone_mapping.csv` | Customer phone mapping database |
| `requirements.txt` | Python dependencies |
| `EUROPEAN & INDIAN DATASET TRAINING.ipynb` | Training & evaluation notebook |
| `models/` | Saved trained model files |
| `logs/` | SMS alert and prediction logs |

---

# 🚀 Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/your-repository-name.git
cd IMPROVED-HYBRID-CNN-BILSTM-ATTENTION-WITH-FOCAL-LOSS-FOR-ROBUST-CREDIT-CARD-FRAUD-DETECTION
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Configure Environment Variables

Create a `.env` file:

```env
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE=Twilio_phone_number (example = +1234567890)
```

---

# 📲 Twilio SMS Setup

## Steps to Configure Twilio

1. Create an account at:  
https://www.twilio.com

2. Verify your mobile number

3. Go to Dashboard → Account Information

4. Copy:
   - Account SID
   - Auth Token

5. Purchase or configure a Twilio phone number

---

# 📞 Customer Phone Mapping

Update `phone_mapping.csv`

create more dataset

```csv
transaction_id,customer_name,customer_phone
TXN001,Rajesh Kumar,your phone number
TXN002,Priya Singh,your phone number
TXN003,Amit Patel,your phone number
etc...
```

The system automatically formats Indian phone numbers using `+91`.

---

# ▶️ Run the Application

```bash
python -m streamlit run app.py
```

Default URL:

```text
http://localhost:8501
```

---

# 📦 Dependencies

```text
streamlit
tensorflow
numpy
pandas
scikit-learn
matplotlib
seaborn
plotly
shap
twilio
python-dotenv
```

---

# 📊 Supported Datasets

## 🇪🇺 European Dataset

- 284,807 transactions
- 492 fraudulent records
- Features: V1–V28, Amount, Time, Class

Dataset Source:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🇮🇳 Indian Dataset

Custom banking transaction dataset supporting:

- Transaction amount
- Merchant category
- Card type
- Transaction time
- Location details

Dataset Source: 
https://www.kaggle.com/datasets/kumarperiya/comprehensive-indian-online-fraud-dataset

---

# 📉 Dashboard Features

The Streamlit dashboard provides:

- Fraud prediction interface
- Transaction probability analysis
- ROC-AUC curve
- Confusion matrix
- SHAP explanations
- Fraud alert logs
- Interactive Plotly visualizations

---

# 📱 SMS Fraud Alert Example

```text
FRAUD ALERT!

Dear Rajesh Kumar,

Transaction TXN001 has been flagged as FRAUDULENT
with 87.3% risk probability.

Time: 08-May-2026 03:45 PM

If this was not you, contact your bank immediately.

- FraudShield
```

---

# 🔐 Security & Best Practices

✅ Never upload `.env` to GitHub  
✅ Store credentials securely  
✅ Validate all transaction inputs  
✅ Use encrypted communication  
✅ Regularly retrain the model  
✅ Monitor model drift and fraud trends  

---

# 📚 Research & Journal Publications

## 📄 Research Paper

### **A Hybrid Deep Learning Model for Detecting Credit Card Fraud Using CNN–BiLSTM with an Attention Mechanism and Focal Loss Optimization**

🔗 Research Paper Link:  
https://www.ijsart.com/a-hybrid-deep-learning-model-for-detecting-credit-card-fraud-using-cnnbilstm-with-an-attention-mechanism-and-focal-loss-optimization-105099

---

## 📄 Survey Paper

### **A Survey on Machine Learning and Deep Learning Approaches for Credit Card Fraud Detection**

🔗 Journal Paper Link:  
https://www.ijsart.com/a-survey-on-machine-learning-and-deep-learning-approaches-for-credit-card-fraud-detection-104577

---

# 📖 Base Research Paper

**Akour, I., Mohamed, N., & Salloum, S. (2025)**  
*Hybrid CNN–LSTM With Attention Mechanism for Robust Credit Card Fraud Detection*  
Published in IEEE Access, Vol. 13, pp. 114056–114065.

IEEE Access:  
[https://ieeexplore.ieee.org](https://ieeexplore.ieee.org/document/11050364)

---

# 🌍 SDG Goal Mapping

## SDG 8 – Decent Work & Economic Growth

- Reduces financial losses
- Supports secure digital transactions
- Enhances fintech innovation

## SDG 9 – Industry, Innovation & Infrastructure

- Strengthens AI-powered banking systems
- Improves cybersecurity infrastructure

## SDG 16 – Peace, Justice & Strong Institutions

- Reduces financial crime
- Improves institutional trust

## SDG 17 – Partnerships for the Goals

- Uses open-source datasets
- Encourages collaborative AI research

---

# 🧪 Project Modules

## Module 1 – Data Collection & Preprocessing

- Data cleaning
- Normalization
- Missing value handling

## Module 2 – Feature Selection & Model Building

- CNN architecture
- BiLSTM integration
- Attention layer implementation

## Module 3 – Model Evaluation

- ROC-AUC
- Precision & Recall
- Confusion Matrix

## Module 4 – SHAP Explainability

- Feature importance analysis
- Attention visualization

## Module 5 – Final Interface

- Streamlit deployment
- Fraud monitoring dashboard
- SMS integration

---

# 🐛 Troubleshooting

## SMS Alerts Not Working

- Verify Twilio credentials
- Ensure Twilio account is active
- Check internet connectivity

## Streamlit App Not Opening

```bash
pip install streamlit
```

## Model Loading Error

- Verify saved model path
- Reinstall TensorFlow dependencies

---

# 🔮 Future Enhancements

🔜 Voice Call Fraud Alerts  
🔜 Email Notifications  
🔜 REST API Integration  
🔜 Real-time Banking Deployment  
🔜 Blockchain Security Layer  
🔜 Multi-model Ensemble Detection  
🔜 Live Fraud Monitoring Dashboard  

---

# 🤝 Acknowledgements

We sincerely thank:

- Our project guide
- Department faculty members
- Open-source contributors
- Kaggle dataset providers
- Research communities in AI and cybersecurity

---

# ❤️ Conclusion

This project demonstrates the effectiveness of hybrid deep learning approaches in modern fraud detection systems. By integrating CNN, BiLSTM, Attention mechanisms, and Focal Loss optimization, the proposed framework achieves strong fraud detection performance while maintaining interpretability and real-time usability.

The addition of SHAP explainability, Streamlit deployment, and Twilio SMS alerts transforms the system from a research prototype into a practical, scalable, and industry-ready fraud detection solution.

---

# 🙏 Thank You

### Made with ❤️ by Team AI & DS for Secure Financial Systems
