# https://streamlitpython.com

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #E63946;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1e3a8a;
        font-weight: 700;
    }
    h2 {
        color: #3b82f6;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load the trained Iris classification model"""
    with open('iris_model.pkl', 'rb') as f:
        model = joblib.load(f)
    return model

# Load Iris data for reference
@st.cache_data
def load_iris_data():
    """Load Iris dataset information"""
    iris = load_iris()
    return iris

# Load model and data
model = load_model()
iris = load_iris_data()

# Header
st.title("🌸 Iris Flower Classifier")
st.markdown("### Predict the species of Iris flowers using Machine Learning")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/logos/scikit-learn-logo.png", width=200)
    st.markdown("## 📊 Input Features")
    st.markdown("Adjust the sliders to input flower measurements:")
    
    # Input sliders
    sepal_length = st.slider(
        "🌿 Sepal Length (cm)",
        min_value=4.0,
        max_value=8.0,
        value=5.4,
        step=0.1,
        help="Length of the sepal in centimeters"
    )
    
    sepal_width = st.slider(
        "🌿 Sepal Width (cm)",
        min_value=2.0,
        max_value=4.5,
        value=3.4,
        step=0.1,
        help="Width of the sepal in centimeters"
    )
    
    petal_length = st.slider(
        "🌺 Petal Length (cm)",
        min_value=1.0,
        max_value=7.0,
        value=1.3,
        step=0.1,
        help="Length of the petal in centimeters"
    )
    
    petal_width = st.slider(
        "🌺 Petal Width (cm)",
        min_value=0.1,
        max_value=2.5,
        value=0.2,
        step=0.1,
        help="Width of the petal in centimeters"
    )
    
    st.markdown("---")
    predict_button = st.button("🔮 Predict Species", use_container_width=True)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📈 Input Visualization")
    
    # Create feature comparison chart
    input_data = {
        'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
        'Value': [sepal_length, sepal_width, petal_length, petal_width],
        'Average': [
            np.mean(iris.data[:, 0]),
            np.mean(iris.data[:, 1]),
            np.mean(iris.data[:, 2]),
            np.mean(iris.data[:, 3])
        ]
    }
    
    df_input = pd.DataFrame(input_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_input['Feature'],
        y=df_input['Value'],
        name='Your Input',
        marker_color='#FF4B4B',
        text=df_input['Value'],
        textposition='auto',
    ))
    fig.add_trace(go.Bar(
        x=df_input['Feature'],
        y=df_input['Average'],
        name='Dataset Average',
        marker_color='#0068C9',
        text=np.round(df_input['Average'], 2),
        textposition='auto',
    ))
    
    fig.update_layout(
        barmode='group',
        title='Your Input vs Dataset Average',
        xaxis_title='Features',
        yaxis_title='Value (cm)',
        height=400,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("## 🎯 Prediction Result")
    
    if predict_button or 'prediction_made' not in st.session_state:
        # Make prediction
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_features)[0]
        prediction_proba = model.predict_proba(input_features)[0]
        
        # Store in session state
        st.session_state.prediction = prediction
        st.session_state.prediction_proba = prediction_proba
        st.session_state.prediction_made = True
    
    if 'prediction_made' in st.session_state:
        species_names = iris.target_names
        predicted_species = species_names[st.session_state.prediction]
        confidence = st.session_state.prediction_proba[st.session_state.prediction] * 100
        
        # Display prediction
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="color: white; margin: 0;">Predicted Species</h2>
            <h1 style="color: white; font-size: 48px; margin: 20px 0;">{predicted_species.title()}</h1>
            <p style="font-size: 24px; margin: 0;">Confidence: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Probability Distribution")
        
        # Probability chart
        prob_df = pd.DataFrame({
            'Species': species_names,
            'Probability': st.session_state.prediction_proba * 100
        })
        
        fig_prob = px.bar(
            prob_df,
            x='Species',
            y='Probability',
            color='Probability',
            color_continuous_scale='RdYlGn',
            text='Probability'
        )
        
        fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_prob.update_layout(
            height=300,
            showlegend=False,
            template='plotly_white',
            yaxis_title='Probability (%)',
            xaxis_title='Species'
        )
        
        st.plotly_chart(fig_prob, use_container_width=True)

# Information Section
st.markdown("---")
st.markdown("## 📚 About Iris Species")

col3, col4, col5 = st.columns(3)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>🌸 Setosa</h3>
        <p><strong>Characteristics:</strong></p>
        <ul>
            <li>Smallest petals</li>
            <li>Wide sepals</li>
            <li>Most distinct species</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3>🌺 Versicolor</h3>
        <p><strong>Characteristics:</strong></p>
        <ul>
            <li>Medium-sized petals</li>
            <li>Intermediate features</li>
            <li>Can overlap with virginica</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="metric-card">
        <h3>🌻 Virginica</h3>
        <p><strong>Characteristics:</strong></p>
        <ul>
            <li>Largest petals</li>
            <li>Long, narrow sepals</li>
            <li>Similar to versicolor</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Dataset exploration section
st.markdown("---")
st.markdown("## 🔍 Explore the Dataset")

tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview", "📈 Feature Distribution", "🔗 Correlation Matrix"])

with tab1:
    st.markdown("### Iris Dataset Sample")
    iris_df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    iris_df['species'] = [iris.target_names[i] for i in iris.target]
    
    st.dataframe(iris_df.head(10), use_container_width=True)
    
    col6, col7, col8 = st.columns(3)
    with col6:
        st.metric("Total Samples", len(iris_df))
    with col7:
        st.metric("Features", len(iris.feature_names))
    with col8:
        st.metric("Species", len(iris.target_names))

with tab2:
    st.markdown("### Feature Distribution by Species")
    
    feature_to_plot = st.selectbox(
        "Select feature to visualize:",
        iris.feature_names
    )
    
    fig_dist = px.violin(
        iris_df,
        y=feature_to_plot,
        x='species',
        color='species',
        box=True,
        points='all',
        color_discrete_sequence=['#FF4B4B', '#0068C9', '#83C9FF']
    )
    
    fig_dist.update_layout(
        height=500,
        template='plotly_white',
        showlegend=True
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    st.markdown("### Feature Correlation Heatmap")
    
    correlation_matrix = iris_df[iris.feature_names].corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    
    fig_corr.update_layout(
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built with ❤️ using Streamlit | Model: Random Forest Classifier | Dataset: Iris (scikit-learn)</p>
    <p>🚀 Deploy this app to Streamlit Cloud for free!</p>
</div>
""", unsafe_allow_html=True)