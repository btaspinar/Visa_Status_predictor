# visa_predictor.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Visa Acceptance Predictor",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .prediction-accepted {
        padding: 20px;
        background-color: #d4edda;
        color: #155724;
        border-radius: 10px;
        font-size: 1.5rem;
        text-align: center;
        margin-top: 20px;
    }
    .prediction-rejected {
        padding: 20px;
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 10px;
        font-size: 1.5rem;
        text-align: center;
        margin-top: 20px;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .info-box {
        padding: 15px;
        background-color: #f0f7ff;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🌎 Visa Acceptance Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    This tool predicts the likelihood of your visa request being accepted based on various factors 
    such as your personal details, travel history, and purpose of visit. Please fill out the form 
    below to get a prediction.
</div>
""", unsafe_allow_html=True)

# Generate sample data for demonstration
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'employment_status': np.random.choice(['Employed', 'Student', 'Self-Employed', 'Unemployed'], n_samples),
        'annual_income': np.random.randint(20000, 150000, n_samples),
        'travel_history': np.random.choice(['None', 'Some', 'Extensive'], n_samples),
        'purpose': np.random.choice(['Tourism', 'Business', 'Education', 'Work'], n_samples),
        'duration': np.random.randint(7, 365, n_samples),
        'family_in_destination': np.random.choice([0, 1], n_samples),
        'previous_visas': np.random.randint(0, 5, n_samples),
        'documentation': np.random.choice(['Incomplete', 'Complete', 'Excellent'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create a target variable based on some rules (for demonstration)
    conditions = (
        (df['annual_income'] > 50000) &
        (df['travel_history'] != 'None') &
        (df['documentation'] != 'Incomplete') &
        (df['employment_status'] != 'Unemployed') &
        (df['purpose'].isin(['Business', 'Education']))
    )
    
    df['accepted'] = np.where(conditions, 1, 0)
    
    # Add some noise
    noise = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    df['accepted'] = np.where(noise, 1 - df['accepted'], df['accepted'])
    
    return df

# Train a simple model
@st.cache_resource
def train_model(df):
    # Preprocess the data
    df_processed = pd.get_dummies(df, drop_first=True)
    
    X = df_processed.drop('accepted', axis=1)
    y = df_processed['accepted']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X.columns

# Generate data and train model
df = generate_sample_data()
model, feature_columns = train_model(df)

# Create input form
with st.form("visa_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">Personal Information</div>', unsafe_allow_html=True)
        age = st.slider("Age", 18, 70, 30)
        education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
        employment_status = st.selectbox("Employment Status", ["Employed", "Student", "Self-Employed", "Unemployed"])
        annual_income = st.slider("Annual Income (USD)", 20000, 150000, 50000)
        family_in_destination = st.radio("Family in Destination Country", ["No", "Yes"])
    
    with col2:
        st.markdown('<div class="sub-header">Travel Information</div>', unsafe_allow_html=True)
        travel_history = st.selectbox("Travel History", ["None", "Some", "Extensive"])
        purpose = st.selectbox("Purpose of Visit", ["Tourism", "Business", "Education", "Work"])
        duration = st.slider("Intended Duration of Stay (days)", 7, 365, 30)
        previous_visas = st.slider("Previous Visas to Developed Countries", 0, 5, 0)
        documentation = st.selectbox("Documentation Quality", ["Incomplete", "Complete", "Excellent"])
    
    submitted = st.form_submit_button("Predict Visa Acceptance")

# When form is submitted
if submitted:
    # Prepare input data
    input_data = {
        'age': age,
        'education': education,
        'employment_status': employment_status,
        'annual_income': annual_income,
        'travel_history': travel_history,
        'purpose': purpose,
        'duration': duration,
        'family_in_destination': 1 if family_in_destination == "Yes" else 0,
        'previous_visas': previous_visas,
        'documentation': documentation
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode
    input_processed = pd.get_dummies(input_df)
    
    # Ensure all columns are present
    for col in feature_columns:
        if col not in input_processed.columns:
            input_processed[col] = 0
    
    # Reorder columns to match training data
    input_processed = input_processed[feature_columns]
    
    # Make prediction
    prediction = model.predict(input_processed)
    probability = model.predict_proba(input_processed)[0][1]
    
    # Display results
    st.markdown("---")
    st.markdown('<div class="sub-header">Prediction Results</div>', unsafe_allow_html=True)
    
    # Show progress bar for probability
    st.write(f"Probability of acceptance: {probability:.2%}")
    st.progress(probability)
    
    # Show prediction
    if prediction[0] == 1:
        st.markdown('<div class="prediction-accepted">Visa Likely to be ACCEPTED</div>', unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown('<div class="prediction-rejected">Visa Likely to be REJECTED</div>', unsafe_allow_html=True)
    
    # Show feature importance
    st.markdown("---")
    st.markdown('<div class="sub-header">Key Factors Influencing This Prediction</div>', unsafe_allow_html=True)
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create a DataFrame for visualization
    feat_imp = pd.DataFrame({
        'feature': [feature_columns[i] for i in indices][:5],
        'importance': [importances[i] for i in indices][:5]
    })
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feat_imp, ax=ax, palette='Blues_r')
    ax.set_title('Top 5 Factors Affecting Visa Decision')
    st.pyplot(fig)
    
    # Recommendations based on prediction
    st.markdown("---")
    st.markdown('<div class="sub-header">Recommendations</div>', unsafe_allow_html=True)
    
    if prediction[0] == 0:
        st.warning("Based on your inputs, here's how you might improve your chances:")
        if annual_income < 50000:
            st.write("💼 Consider showing additional financial support or sponsorships")
        if travel_history == "None":
            st.write("✈️ Build travel history with visits to other countries first")
        if documentation == "Incomplete":
            st.write("📋 Ensure all your documents are complete and properly organized")
        if employment_status == "Unemployed":
            st.write("👔 Show strong ties to your home country (property, family, etc.)")
    else:
        st.success("Your application looks strong! For best results:")
        st.write("✅ Double-check all documentation for accuracy")
        st.write("✅ Be prepared for a potential interview")
        st.write("✅ Apply well in advance of your planned travel date")

# Add sidebar with information
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1008/1008109.png", width=100)
    st.title("About This Tool")
    st.info("""
    This visa prediction tool uses machine learning to estimate the likelihood of your visa application being accepted.
    
    **Disclaimer**: This is a demonstration tool only and does not guarantee actual visa outcomes. Always consult official immigration sources.
    """)
    
    st.markdown("---")
    st.subheader("Tips for Successful Applications")
    st.write("""
    - 📋 Ensure all documents are complete and accurate
    - 💼 Demonstrate strong financial stability
    - 🏠 Show ties to your home country
    - ✈️ Maintain a good travel history
    - 📅 Apply well in advance of your travel date
    """)
    
    st.markdown("---")
    st.subheader("Sample Data Distribution")
    st.write(f"Training samples: {len(df)}")
    st.write(f"Acceptance rate in training data: {df['accepted'].mean():.2%}")
    
    # Plot acceptance by purpose
    fig, ax = plt.subplots(figsize=(8, 4))
    df.groupby('purpose')['accepted'].mean().sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Acceptance Rate by Purpose')
    ax.set_xlabel('Acceptance Rate')
    st.pyplot(fig)