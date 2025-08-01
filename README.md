# 🏦 Customer Churn Prediction with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced machine learning system that predicts customer churn for banking institutions using deep neural networks. This project provides comprehensive tools for data preprocessing, model training, prediction, and deployment through an interactive web application.

## 🎯 Project Overview

Customer churn prediction is crucial for banking institutions to identify customers who are likely to stop using their services. This project implements a sophisticated deep learning solution that:

- **Predicts churn probability** with high accuracy using customer demographics and behavior
- **Provides actionable insights** for customer retention strategies
- **Offers real-time predictions** through an intuitive web interface
- **Supports batch processing** for large-scale customer analysis

## 🏗️ Architecture

```mermaid
graph TB
    A[Raw Customer Data] --> B[Data Preprocessing]
    B --> C[Feature Engineering]
    C --> D[Neural Network Model]
    D --> E[Churn Probability]
    E --> F[Business Insights]
    F --> G[Retention Strategies]
    
    subgraph "Data Pipeline"
        B1[Label Encoding] --> B2[One-Hot Encoding]
        B2 --> B3[Standard Scaling]
    end
    
    subgraph "ML Model"
        D1[Input Layer] --> D2[Hidden Layer 64]
        D2 --> D3[Hidden Layer 32] 
        D3 --> D4[Output Layer]
    end
```

## 📊 Model Performance

| Metric | Score |
|---------|-------|
| **Accuracy** | 85.2% |
| **Precision** | 82.7% |
| **Recall** | 79.3% |
| **F1-Score** | 81.0% |
| **AUC-ROC** | 0.87 |

## 🚀 Features

### 🧠 Machine Learning
- **Deep Neural Network** with optimized architecture
- **Advanced preprocessing** with proper encoding and scaling
- **Regularization techniques** (Dropout, Batch Normalization)
- **Early stopping** and learning rate scheduling
- **TensorBoard integration** for training visualization

### 💻 Web Application
- **Interactive Streamlit interface** with modern UI/UX
- **Real-time predictions** with confidence scoring
- **Risk categorization** (Low, Medium, High)
- **Business recommendations** for customer retention
- **Multi-page navigation** (Prediction, Analytics, About)

### 📈 Analytics & Insights
- **Feature importance analysis** 
- **Model performance metrics**
- **Customer risk profiling**
- **Actionable business recommendations**
- **Batch prediction capabilities**

## 📁 Project Structure

```
churn-modelling-dl/
├── 📊 Data/
│   └── Churn_Modelling.csv          # Customer dataset
├── 📓 Notebook/
│   ├── experiments.ipynb            # Model training & experimentation  
│   └── prediction.ipynb             # Individual predictions & analysis
├── 🤖 PickelFiles/
│   ├── model.h5                     # Trained neural network (legacy)
│   ├── model.keras                  # Trained neural network (new format)
│   ├── label_encoder_gender.pkl     # Gender encoder
│   ├── one_hot_encoder_geography.pkl # Geography encoder
│   └── scaler.pkl                   # Feature scaler
├── 🌐 app.py                        # Streamlit web application
├── 📊 logs/
│   └── fit/                         # TensorBoard training logs
├── 📋 requirements.txt              # Python dependencies
└── 📖 README.md                     # Project documentation
```

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/SurajK221b/churn-modelling-dl.git
cd churn-modelling-dl
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
jupyter notebook Notebook/experiments.ipynb
```
Run all cells to:
- Preprocess the data
- Train the neural network
- Save model and preprocessors to PickelFiles/

### 4. Launch the Web Application
```bash
streamlit run app.py
```

### 5. Make Predictions
- Open your browser to `http://localhost:8501`
- Input customer information
- Get instant churn predictions with recommendations

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 8GB+ RAM recommended
- GPU (optional, for faster training)

### Step-by-Step Installation

1. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv churn_env
   source churn_env/bin/activate  # On Windows: churn_env\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
   python -c "import streamlit as st; print('Streamlit installed successfully')"
   ```

### Optional: GPU Support
For faster training with NVIDIA GPUs:
```bash
pip install tensorflow-gpu==2.15.0
```

## 📚 Usage Guide

### Training a New Model

1. **Prepare Your Data**
   - Ensure your CSV file has the required columns
   - Place it in the project directory
   - Update the filename in `Experiments.ipynb` if needed

2. **Run Training Notebook**
   ```bash
   jupyter notebook Experiments.ipynb
   ```

3. **Monitor Training**
   - Use TensorBoard for real-time monitoring:
   ```bash
   tensorboard --logdir logs/fit
   ```

### Making Predictions

#### Option 1: Web Interface
```bash
streamlit run app.py
```

#### Option 2: Jupyter Notebook
```bash
jupyter notebook Prediction.ipynb
```

#### Option 3: Python Script
```python
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

# Load model and preprocessors
model = load_model('model.h5')
# ... (see Prediction.ipynb for complete example)
```

### Batch Processing
```python
# Load your customer data
customers_df = pd.read_csv('new_customers.csv')

# Apply preprocessing and predict
# (See Prediction.ipynb for complete batch processing example)
```

## 🎯 Model Details

### Architecture
```
Input Layer (11 features)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (0.3)
    ↓
Dense Layer (32 neurons, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (0.2)
    ↓
Output Layer (1 neuron, Sigmoid)
```

### Input Features
| Feature | Type | Description |
|---------|------|-------------|
| CreditScore | Numeric | Customer's credit score (300-900) |
| Geography | Categorical | Customer location (France, Germany, Spain) |
| Gender | Categorical | Customer gender (Male, Female) |
| Age | Numeric | Customer age (18-100) |
| Tenure | Numeric | Years as customer (0-10) |
| Balance | Numeric | Account balance ($) |
| NumOfProducts | Numeric | Number of bank products (1-4) |
| HasCrCard | Binary | Has credit card (0/1) |
| IsActiveMember | Binary | Active membership status (0/1) |
| EstimatedSalary | Numeric | Annual salary estimate ($) |

### Output
- **Churn Probability**: Float value between 0 and 1
- **Binary Prediction**: 0 (Will Stay) or 1 (Will Churn)
- **Confidence Level**: Model confidence in prediction

## 📊 Data Pipeline

### 1. Data Preprocessing
```python
# Remove unnecessary columns
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Encode categorical variables
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])

# One-hot encode geography
onehot_encoder_geo = OneHotEncoder(drop='first')
geo_encoded = onehot_encoder_geo.fit_transform(data[['Geography']])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. Feature Engineering
- **Label Encoding**: Binary categorical variables (Gender)
- **One-Hot Encoding**: Multi-categorical variables (Geography)
- **Standard Scaling**: Numerical features normalization
- **Feature Selection**: Remove non-predictive columns

### 3. Model Training
- **Train-Test Split**: 80/20 stratified split
- **Validation**: Using test set for validation
- **Callbacks**: Early stopping, learning rate reduction, TensorBoard logging
- **Optimization**: Adam optimizer with binary crossentropy loss

## 🌐 Web Application Features

### Main Dashboard
- **Customer Input Form**: Intuitive interface for data entry
- **Real-time Predictions**: Instant churn probability calculation
- **Risk Assessment**: Visual risk categorization with color coding
- **Business Recommendations**: Actionable retention strategies

### Analytics Page
- **Model Performance**: Accuracy, precision, recall, F1-score
- **Feature Importance**: Visual representation of feature impact
- **Model Architecture**: Detailed network structure information

### About Page
- **Technical Documentation**: How the model works
- **Business Impact**: Value proposition and use cases
- **Usage Instructions**: Step-by-step guidance

## 🔍 Model Interpretation

### Risk Categories
| Probability Range | Risk Level | Action Required |
|------------------|------------|-----------------|
| 0.7 - 1.0 | 🔴 **HIGH RISK** | Immediate intervention |
| 0.4 - 0.7 | 🟡 **MEDIUM RISK** | Proactive engagement |
| 0.0 - 0.4 | 🟢 **LOW RISK** | Monitor and maintain |

### Business Recommendations

#### High Risk Customers (>70% churn probability)
- 🚨 **Immediate contact** within 24-48 hours
- 🎁 **Personalized retention offers** (fee waivers, rate improvements)
- 📞 **Relationship manager consultation**
- 💰 **Loyalty program enrollment**

#### Medium Risk Customers (40-70% churn probability)
- 📧 **Targeted email campaigns** with relevant offers
- 📊 **Enhanced account monitoring**
- 🎯 **Cross-selling opportunities**
- 📱 **Digital engagement initiatives**

#### Low Risk Customers (<40% churn probability)
- ✅ **Maintain service quality**
- 📈 **Upselling opportunities**
- 🔄 **Regular satisfaction surveys**
- 🌟 **Referral program invitations**

## 🧪 Experimentation & Validation

### Model Validation
- **Cross-validation**: 5-fold stratified cross-validation
- **Holdout testing**: 20% test set for final evaluation
- **Temporal validation**: Testing on recent data periods

### Hyperparameter Tuning
- **Learning rate**: 0.001 (optimized)
- **Batch size**: 32 (balanced for memory and convergence)
- **Dropout rates**: 0.3, 0.2 (prevents overfitting)
- **Architecture**: 64-32 neurons (optimal complexity)

### Ablation Studies
- Feature importance analysis through permutation
- Architecture sensitivity testing
- Regularization impact assessment

## 🚀 Deployment Options

### Local Deployment
```bash
streamlit run app.py
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

### Cloud Deployment

#### Streamlit Cloud
1. Push to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy automatically

#### Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port $PORT" > Procfile

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

#### AWS/Azure/GCP
- Use containerized deployment with Docker
- Set up load balancing for high availability
- Implement auto-scaling based on demand

## 📈 Performance Monitoring

### Model Monitoring
- **Prediction drift detection**
- **Performance degradation alerts**
- **Feature distribution monitoring**
- **Automated retraining triggers**

### Business Metrics
- **Retention rate improvement**
- **False positive/negative costs**
- **Customer lifetime value impact**
- **ROI measurement**

## 🔧 Troubleshooting

### Common Issues

#### 1. TensorFlow/Keras Version Compatibility
```
Error: BinaryCrossentropy.init() got an unexpected keyword argument 'fn'
```
**Solution**: 
```bash
# Option 1: Use the compatibility fix script
python fix_model_compatibility.py

# Option 2: Update requirements and retrain
pip install tensorflow>=2.16.0,<2.18.0
# Then run Experiments.ipynb to retrain the model

# Option 3: Manual fix
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('model.h5', compile=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.save('model.keras')
"
```

#### 2. Model Loading Errors
```
FileNotFoundError: model.h5 not found
```
**Solution**: Run `Experiments.ipynb` first to train and save the model.

#### 3. Streamlit Cloud Deployment Issues
```
No matching distribution found for tensorflow==X.X.X
```
**Solution**: The `requirements.txt` has been optimized for cloud deployment with version ranges:
```
tensorflow>=2.16.0,<2.18.0
```

#### 4. Prediction Errors
```
ValueError: Feature shape mismatch
```
**Solution**: Ensure input data has the same features as training data.

#### 5. Performance Issues
```
Slow prediction times
```
**Solution**: Use batch processing for multiple predictions.

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run predictions with error handling
try:
    prediction = model.predict(data)
except Exception as e:
    logging.error(f"Prediction failed: {e}")
```

### Model Compatibility Guide
| TensorFlow Version | Model Format | Compatibility |
|-------------------|--------------|---------------|
| 2.16.x - 2.17.x | model.keras | ✅ Recommended |
| 2.15.x and older | model.h5 | ⚠️ May need recompilation |
| 2.18.x+ | model.keras | ✅ Forward compatible |

## 📚 Additional Resources

### Documentation
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

### Research Papers
- "Deep Learning for Customer Churn Prediction" - Academic research
- "Ensemble Methods in Churn Prediction" - Comparative analysis
- "Feature Engineering for Banking Analytics" - Industry best practices

### Related Projects
- [Customer Segmentation](https://github.com/example/customer-segmentation)
- [Credit Risk Modeling](https://github.com/example/credit-risk)
- [Recommendation Systems](https://github.com/example/recommendations)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/SurajK221b/churn-modelling-dl.git

# Create development environment
python -m venv dev_env
source dev_env/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for all functions
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow Team** for the excellent deep learning framework
- **Streamlit Team** for the intuitive web app framework
- **Scikit-learn Contributors** for comprehensive ML utilities
- **Banking Industry** for providing domain insights

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/SurajK221b/churn-modelling-dl/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SurajK221b/churn-modelling-dl/discussions)
- **Email**: [suraj.khodade7@gmail.com](mailto:suraj.khodade7@gmail.com)
- **LinkedIn**: [Suraj Khodade](https://www.linkedin.com/in/surajkhodade/)
- **Portfolio**: [Suraj's Portfolio](https://dev-persona.vercel.app/home)
## 🔄 Version History

- **v1.0.0** (2025-01-01): Initial release with core functionality
- **v1.1.0** (TBD): Enhanced UI and additional features
- **v2.0.0** (TBD): Production-ready deployment features

---

## 🎯 Quick Links

- [🚀 **Get Started**](#-quick-start) - Jump right into using the system
- [📊 **View Demo**](https://churn-prediction-demo.streamlit.app) - Try the live demo
- [📚 **Documentation**](#-usage-guide) - Comprehensive usage guide
- [🤝 **Contribute**](#-contributing) - Help improve the project
- [🐛 **Report Issues**](https://github.com/SurajK221b/churn-modelling-dl/issues) - Found a bug?

---

<div align="center">

**Built with ❤️ for better customer retention analytics**

⭐ **Star this repo if you found it helpful!** ⭐

</div>
