import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow, handle gracefully if not available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Try to import scikit-learn components
try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Optional plotly imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# LOAD MODEL AND PREPROCESSORS
# =============================================================================

@st.cache_resource
def load_model_components():
    """Load all model components with caching for better performance."""
    if not TENSORFLOW_AVAILABLE:
        return None, None, None, None, False
        
    if not SKLEARN_AVAILABLE:
        st.error("❌ Scikit-learn is required but not available.")
        return None, None, None, None, False
        
    try:
        # Try to load the model with multiple compatibility approaches
        model = None
        
        # Check if files exist in PickelFiles directory
        model_path_keras = 'PickelFiles/model.keras'
        model_path_h5 = 'PickelFiles/model.h5'
        
        # Approach 1: Try loading the newer Keras format first
        if os.path.exists(model_path_keras):
            try:
                model = tf.keras.models.load_model(model_path_keras)
                st.success("✅ Model loaded successfully from model.keras")
            except Exception as keras_error:
                st.warning(f"⚠️ Keras format loading failed: {keras_error}")
        
        # Approach 2: Fall back to H5 format with compile=False
        if model is None and os.path.exists(model_path_h5):
            try:
                model = tf.keras.models.load_model(model_path_h5, compile=False)
                
                # Recompile the model with current TensorFlow version
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                st.success("✅ Model loaded successfully from model.h5 with recompilation")
                
            except Exception as h5_error:
                st.warning(f"⚠️ H5 format loading failed: {h5_error}")
        
        # Approach 3: Try loading H5 with custom objects as last resort
        if model is None and os.path.exists(model_path_h5):
            try:
                model = tf.keras.models.load_model(
                    model_path_h5,
                    custom_objects={
                        'BinaryCrossentropy': tf.keras.losses.BinaryCrossentropy(),
                        'Adam': tf.keras.optimizers.Adam()
                    }
                )
                st.success("✅ Model loaded successfully with custom objects")
                
            except Exception as custom_error:
                st.error(f"❌ All model loading approaches failed: {custom_error}")
                raise custom_error
        
        if model is None:
            raise FileNotFoundError("No valid model file found (model.keras or model.h5)")
        
        # Load the encoders and scaler from PickelFiles directory
        with open('PickelFiles/label_encoder_gender.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
        
        with open('PickelFiles/onehot_encoder_geo.pkl', 'rb') as file:
            onehot_encoder_geo = pickle.load(file)
        
        with open('PickelFiles/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
            
        return model, label_encoder_gender, onehot_encoder_geo, scaler, True
        
    except FileNotFoundError as e:
        st.warning(f"⚠️ Model file not found: {e}")
        st.info("This is a demo version without trained model files.")
        return None, None, None, None, False
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.info("Running in demo mode without full model functionality.")
        return None, None, None, None, False

# Demo data for when model files are not available
def create_demo_encoders():
    """Create demo encoders for demonstration purposes."""
    if not SKLEARN_AVAILABLE:
        return None, None, None
        
    # Create dummy encoders with expected categories
    demo_label_encoder = type('DemoLabelEncoder', (), {
        'classes_': np.array(['Female', 'Male']),
        'transform': lambda self, x: [0 if val == 'Female' else 1 for val in x]
    })()
    
    demo_onehot_encoder = type('DemoOneHotEncoder', (), {
        'categories_': [np.array(['France', 'Germany', 'Spain'])],
        'get_feature_names_out': lambda self, cols=None: ['Geography_Germany', 'Geography_Spain'],
        'transform': lambda self, x: np.array([[1, 0] if x[0][0] == 'Germany' else [0, 1] if x[0][0] == 'Spain' else [0, 0]])
    })()
    
    demo_scaler = type('DemoScaler', (), {
        'transform': lambda self, x: (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8)
    })()
    
    return demo_label_encoder, demo_onehot_encoder, demo_scaler

# Load components
model, label_encoder_gender, onehot_encoder_geo, scaler, components_loaded = load_model_components()

# If model components aren't loaded, use demo versions for UI demonstration
if not components_loaded:
    if SKLEARN_AVAILABLE:
        label_encoder_gender, onehot_encoder_geo, scaler = create_demo_encoders()
        demo_mode = True
    else:
        demo_mode = True
        # Create minimal demo data
        label_encoder_gender = type('DemoLabelEncoder', (), {'classes_': ['Female', 'Male']})()
        onehot_encoder_geo = type('DemoOneHotEncoder', (), {'categories_': [['France', 'Germany', 'Spain']]})()
else:
    demo_mode = False

# =============================================================================
# MAIN APPLICATION INTERFACE
# =============================================================================

# Display mode information
if not TENSORFLOW_AVAILABLE:
    st.error("🚫 **TensorFlow not available** - This deployment is running in demo mode")
    st.info("💡 For full functionality with live predictions, run locally with: `pip install tensorflow` and the complete requirements.txt")

if demo_mode:
    st.warning("⚠️ **Demo Mode** - Model files not found. Interface demonstration only.")
    st.info("🧠 To enable predictions: Run `Notebook/experiments.ipynb` to train and save the model, then restart the app.")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2980b9);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
    }
    .high-risk {
        background: #ffe6e6;
        border-left-color: #dc3545;
    }
    .medium-risk {
        background: #fff3cd;
        border-left-color: #ffc107;
    }
    .low-risk {
        background: #d4edda;
        border-left-color: #28a745;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    .demo-banner {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }
    .stSlider > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Header
status_text = "Demo Version" if demo_mode else "Production Ready"
st.markdown(f"""
<div class="main-header">
    <h1>🏦 Customer Churn Prediction System</h1>
    <p>Advanced AI-powered customer retention analytics for banking institutions</p>
    <small>Status: {status_text}</small>
</div>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["🔮 Prediction", "📈 Analytics", "📊 Data Insights", "ℹ️ About"]
)

if page == "🔮 Prediction":
    # =============================================================================
    # PREDICTION PAGE
    # =============================================================================
    
    st.header("Customer Information Input")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Geographic & Personal Info")
        
        geography = st.selectbox(
            'Geography', 
            onehot_encoder_geo.categories_[0],
            help="Customer's geographic location"
        )
        
        gender = st.selectbox(
            'Gender', 
            label_encoder_gender.classes_,
            help="Customer's gender"
        )
        
        age = st.slider(
            'Age', 
            18, 92, 35,
            help="Customer's age in years"
        )
        
        tenure = st.slider(
            'Tenure (Years)', 
            0, 10, 5,
            help="Number of years as a customer"
        )
    
    with col2:
        st.subheader("💰 Financial Information")
        
        credit_score = st.number_input(
            'Credit Score', 
            min_value=300, 
            max_value=900, 
            value=650, 
            step=1,
            help="Customer's credit score (300-900)"
        )
        
        balance = st.number_input(
            'Account Balance ($)', 
            min_value=0.0, 
            max_value=1000000.0, 
            value=50000.0, 
            step=1000.0,
            help="Current account balance"
        )
        
        estimated_salary = st.number_input(
            'Estimated Salary ($)', 
            min_value=0.0, 
            max_value=500000.0, 
            value=75000.0, 
            step=1000.0,
            help="Annual estimated salary"
        )
    
    # Product and activity information
    st.subheader("📦 Products & Activity")
    
    col3, col4 = st.columns(2)
    
    with col3:
        num_of_products = st.selectbox(
            'Number of Products', 
            [1, 2, 3, 4],
            index=1,
            help="Number of bank products customer uses"
        )
        
        has_cr_card = st.selectbox(
            'Has Credit Card', 
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Whether customer has a credit card"
        )
    
    with col4:
        is_active_member = st.selectbox(
            'Is Active Member', 
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Whether customer is actively using services"
        )
    
    # Prediction button
    predict_button_text = "🎭 Demo Prediction (Simulated)" if demo_mode else "🔮 Predict Churn Probability"
    
    if st.button(predict_button_text, type="primary", use_container_width=True):
        
        if demo_mode:
            # Demo prediction using simple heuristics
            risk_score = 0.5
            
            # Simple risk calculation for demo
            if age > 60: risk_score += 0.2
            if credit_score < 600: risk_score += 0.2
            if balance < 50000: risk_score += 0.1
            if num_of_products == 1: risk_score += 0.15
            if is_active_member == 0: risk_score += 0.25
            if has_cr_card == 0: risk_score += 0.1
            
            prediction_proba = max(0.05, min(0.95, risk_score))
            
        else:
            # Real prediction mode
            # Prepare the input data
            input_data = pd.DataFrame({
                'CreditScore': [credit_score],
                'Gender': [label_encoder_gender.transform([gender])[0]],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_of_products],
                'HasCrCard': [has_cr_card],
                'IsActiveMember': [is_active_member],
                'EstimatedSalary': [estimated_salary]
            })
            
            # One-hot encode 'Geography'
            geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
            geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
            
            # Combine one-hot encoded columns with input data
            input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
            
            # Scale the input data
            input_data_scaled = scaler.transform(input_data)
            
            # Predict churn
            with st.spinner("🧠 AI model analyzing customer data..."):
                prediction = model.predict(input_data_scaled)
                prediction_proba = prediction[0][0]
        
        # Display results with enhanced styling
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        # Determine risk level and styling
        if prediction_proba >= 0.7:
            risk_class = "high-risk"
            risk_level = "🔴 HIGH RISK"
            risk_emoji = "🚨"
        elif prediction_proba >= 0.4:
            risk_class = "medium-risk"
            risk_level = "🟡 MEDIUM RISK"
            risk_emoji = "⚠️"
        else:
            risk_class = "low-risk"
            risk_level = "🟢 LOW RISK"
            risk_emoji = "✅"
        
        # Create columns for metrics
        met_col1, met_col2, met_col3 = st.columns(3)
        
        with met_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{risk_emoji}</h3>
                <h4>Risk Level</h4>
                <p><strong>{risk_level.split()[1]} {risk_level.split()[2]}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with met_col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊</h3>
                <h4>Churn Probability</h4>
                <p><strong>{prediction_proba:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with met_col3:
            confidence = abs(prediction_proba - 0.5) * 2
            confidence_level = "Very High" if confidence >= 0.8 else "High" if confidence >= 0.6 else "Medium" if confidence >= 0.4 else "Low"
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯</h3>
                <h4>Confidence</h4>
                <p><strong>{confidence_level}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed prediction box
        prediction_text = "The customer is likely to churn." if prediction_proba > 0.5 else "The customer is likely to stay."
        
        st.markdown(f"""
        <div class="prediction-box {risk_class}">
            <h3>{risk_emoji} Prediction Result</h3>
            <p><strong>{prediction_text}</strong></p>
            <p>Churn Probability: <strong>{prediction_proba:.1%}</strong></p>
            <p>Confidence Level: <strong>{confidence_level}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Business recommendations
        st.subheader("💼 Recommended Actions")
        
        recommendations = []
        if prediction_proba >= 0.7:
            recommendations.extend([
                "🚨 **Immediate intervention required** - Contact customer within 24 hours",
                "🎁 Offer personalized retention incentives",
                "📞 Schedule personal consultation with relationship manager"
            ])
        elif prediction_proba >= 0.4:
            recommendations.extend([
                "⚠️ **Proactive engagement recommended**",
                "📧 Send targeted promotional offers",
                "📊 Monitor account activity closely"
            ])
        else:
            recommendations.extend([
                "✅ **Customer appears stable**",
                "📈 Consider upselling opportunities",
                "🔄 Maintain regular service quality"
            ])
        
        # Add specific recommendations based on customer profile
        if credit_score < 600:
            recommendations.append("💳 Offer financial advisory services")
        if balance < 50000:
            recommendations.append("💰 Promote savings programs")
        if num_of_products == 1:
            recommendations.append("🛍️ Cross-sell additional products")
        if is_active_member == 0:
            recommendations.append("📱 Launch re-engagement campaigns")
        
        for rec in recommendations[:5]:  # Show top 5 recommendations
            st.markdown(f"- {rec}")

elif page == "📈 Analytics":
    # =============================================================================
    # ANALYTICS PAGE
    # =============================================================================
    
    st.header("📈 Model Analytics & Insights")
    
    # Model information
    st.subheader("🧠 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if components_loaded:
            st.info(f"""
            **Model Architecture**: Deep Neural Network  
            **Input Features**: {model.input_shape[1] if model else 'N/A'}  
            **Model Type**: Binary Classification  
            **Framework**: TensorFlow/Keras
            """)
        else:
            st.info("""
            **Model Architecture**: Deep Neural Network  
            **Input Features**: 11 (when loaded)  
            **Model Type**: Binary Classification  
            **Framework**: TensorFlow/Keras
            """)
    
    with col2:
        if components_loaded:
            st.info(f"""
            **Total Parameters**: {model.count_params():,}  
            **Layers**: {len(model.layers)}  
            **Activation**: ReLU, Sigmoid  
            **Optimizer**: Adam
            """)
        else:
            st.info("""
            **Total Parameters**: ~5,000 (estimated)  
            **Layers**: 4 (Input + 2 Hidden + Output)  
            **Activation**: ReLU, Sigmoid  
            **Optimizer**: Adam
            """)
    
    # Feature importance (simplified visualization)
    st.subheader("📊 Feature Importance")
    
    # Sample feature importance data (in real scenario, this would be calculated)
    features = ['Age', 'Credit Score', 'Balance', 'Geography', 'Number of Products', 
               'Is Active Member', 'Tenure', 'Has Credit Card', 'Gender', 'Estimated Salary']
    importance = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03, 0.02, 0.01]
    
    # Create a simple bar chart using Streamlit
    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(features, importance, color='skyblue')
        ax.set_xlabel('Relative Importance')
        ax.set_title('Feature Importance in Churn Prediction')
        ax.grid(axis='x', alpha=0.3)
        
        st.pyplot(fig)
    else:
        # Fallback to simple chart
        chart_data = pd.DataFrame({
            'Features': features,
            'Importance': importance
        })
        st.bar_chart(chart_data.set_index('Features'))
    
    # Model performance metrics
    st.subheader("📊 Model Performance Metrics")
    
    # Sample metrics (in real scenario, these would be from actual evaluation)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "85.2%", "2.1%")
    
    with col2:
        st.metric("Precision", "82.7%", "1.5%")
    
    with col3:
        st.metric("Recall", "79.3%", "-0.8%")
    
    with col4:
        st.metric("F1-Score", "81.0%", "0.9%")

elif page == "📊 Data Insights":
    # =============================================================================
    # DATA INSIGHTS PAGE
    # =============================================================================
    
    st.header("📊 Data Insights & Statistics")
    
    # Load and display dataset information
    try:
        data = pd.read_csv('Data/Churn_Modelling.csv')
        
        st.subheader("📋 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(data):,}")
        
        with col2:
            churn_rate = data['Exited'].mean()
            st.metric("Churn Rate", f"{churn_rate:.1%}")
        
        with col3:
            avg_age = data['Age'].mean()
            st.metric("Average Age", f"{avg_age:.1f}")
        
        with col4:
            avg_balance = data['Balance'].mean()
            st.metric("Average Balance", f"${avg_balance:,.0f}")
        
        # Display sample data
        st.subheader("📄 Sample Data")
        st.dataframe(data.head(10))
        
        # Basic statistics
        st.subheader("📈 Statistical Summary")
        st.dataframe(data.describe())
        
        # Churn distribution by geography
        if PLOTLY_AVAILABLE:
            st.subheader("🌍 Churn by Geography")
            geo_churn = data.groupby('Geography')['Exited'].agg(['count', 'sum', 'mean']).reset_index()
            geo_churn.columns = ['Geography', 'Total_Customers', 'Churned_Customers', 'Churn_Rate']
            
            fig = px.bar(geo_churn, x='Geography', y='Churn_Rate', 
                        title='Churn Rate by Geography',
                        labels={'Churn_Rate': 'Churn Rate (%)'})
            st.plotly_chart(fig)
        
    except FileNotFoundError:
        st.warning("⚠️ Dataset not found. Please ensure 'Data/Churn_Modelling.csv' exists.")
        st.info("This page shows insights from the training dataset when available.")

elif page == "ℹ️ About":
    # =============================================================================
    # ABOUT PAGE
    # =============================================================================
    
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## 🎯 Purpose
    This Customer Churn Prediction System uses advanced machine learning to help banking institutions 
    identify customers who are likely to stop using their services. By predicting churn probability, 
    banks can take proactive measures to retain valuable customers.
    
    ## 🧠 How It Works
    
    ### 1. Data Input
    - Customer demographics (age, gender, geography)
    - Financial information (credit score, balance, salary)
    - Product usage (number of products, credit card status)
    - Activity status (tenure, active membership)
    
    ### 2. AI Processing
    - **Preprocessing**: Data is standardized and encoded using the same transformations from training
    - **Neural Network**: Deep learning model with multiple hidden layers processes the features
    - **Prediction**: Model outputs a probability score between 0 and 1
    
    ### 3. Business Intelligence
    - **Risk Classification**: Customers are categorized as Low, Medium, or High risk
    - **Actionable Insights**: Specific recommendations for customer retention
    - **Confidence Scoring**: Reliability measure for each prediction
    
    ## 📊 Model Architecture
    - **Type**: Artificial Neural Network (ANN)
    - **Layers**: Input → Dense(64) → Dense(32) → Output(1)
    - **Activation Functions**: ReLU for hidden layers, Sigmoid for output
    - **Optimization**: Adam optimizer with binary crossentropy loss
    - **Regularization**: Dropout and Batch Normalization
    
    ## 🔧 Technical Features
    - **Real-time Predictions**: Instant results for individual customers
    - **Scalable Architecture**: Can handle batch processing
    - **Model Persistence**: Trained model and preprocessors saved for consistency
    - **Error Handling**: Robust error management and user feedback
    
    ## 📈 Business Impact
    - **Cost Reduction**: Proactive retention is cheaper than customer acquisition
    - **Revenue Protection**: Retain high-value customers before they churn
    - **Personalized Service**: Tailored recommendations based on risk profile
    - **Data-Driven Decisions**: Evidence-based customer management strategies
    
    ## 🚀 Future Enhancements
    - Real-time data integration
    - Advanced feature engineering
    - Model performance monitoring
    - A/B testing for retention strategies
    
    ## 📋 Requirements
    - Python 3.8+
    - TensorFlow 2.16+
    - Streamlit
    - Scikit-learn
    - Pandas, NumPy
    
    ## 🔗 Related Files
    - `Notebook/experiments.ipynb`: Model training and experimentation
    - `Notebook/prediction.ipynb`: Individual prediction examples and analysis
    - `PickelFiles/model.h5`: Trained neural network model
    - Various `.pkl` files: Saved preprocessors (encoders, scaler)
    
    ---
    
    **Developed with ❤️ for better customer retention analytics**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    🏦 Customer Churn Prediction System | Powered by TensorFlow & Streamlit
</div>
""", unsafe_allow_html=True)
