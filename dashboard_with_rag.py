# ==================== INTEGRATED CYBERSECURITY DASHBOARD WITH RAG ====================
"""
Complete Cybersecurity Dashboard with RAG-powered AI Assistant
This combines the ML prediction models with a RAG chatbot
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import joblib
import re
import urllib.parse as urlparse
import warnings
import os

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# RAG imports
try:
    from rag_system import RAGEngine, RAGChatbot, RAGConfig, EMBEDDINGS_AVAILABLE
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    EMBEDDINGS_AVAILABLE = False

warnings.filterwarnings('ignore')


# ==================== URL FEATURE EXTRACTOR ====================
def extract_url_features(url):
    """Extracts structural features from a URL."""
    features = {}
    features['url_length'] = len(url)
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_special_chars'] = len(re.findall(r'[\W_]', url))
    features['has_https'] = int(url.startswith("https"))
    
    netloc = urlparse.urlparse(url).netloc
    features['has_ip'] = int(bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', netloc))) 
    
    features['num_subdirs'] = url.count('/')
    features['num_dots'] = url.count('.')
    features['has_at_symbol'] = int('@' in url)
    features['has_hyphen'] = int('-' in url)
    features['domain_length'] = len(netloc)
    return pd.DataFrame([features])


# ==================== RAG CHATBOT TAB ====================
def render_chat_message(role: str, content: str, sources=None):
    """Render a chat message"""
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            st.markdown(content)
            if sources:
                with st.expander(f"📚 View {len(sources)} Source(s)"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"""
                        **Source {i}: {source.get('title', 'Unknown')}**  
                        *Category: {source.get('category', 'N/A')} | Relevance: {source.get('score', 0):.0%}*  
                        > {source.get('content', '')[:300]}...
                        """)


def run_rag_tab():
    """RAG Chatbot Tab"""
    st.header("🤖 AI Cybersecurity Assistant (RAG-Powered)")
    
    if not RAG_AVAILABLE:
        st.error("❌ RAG system not available. Ensure `rag_system.py` is in the project directory.")
        return
    
    if not EMBEDDINGS_AVAILABLE:
        st.error("""
        ❌ **sentence-transformers** is required for the RAG system.
        
        Install it by running:
        ```bash
        pip install sentence-transformers torch transformers
        ```
        """)
        return
    
    # Initialize RAG in session state
    if 'rag_initialized' not in st.session_state:
        with st.spinner("🔄 Initializing RAG Knowledge Base (first time only)..."):
            config = RAGConfig()
            st.session_state.rag_engine = RAGEngine(config)
            st.session_state.rag_engine.initialize_knowledge_base()
            st.session_state.rag_chatbot = RAGChatbot(st.session_state.rag_engine)
            st.session_state.rag_initialized = True
            st.session_state.rag_chat_history = []
    
    # Sidebar settings for RAG
    with st.sidebar:
        st.subheader("🤖 RAG Settings")
        show_sources = st.checkbox("Show retrieved sources", value=True, key="show_sources")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.rag_chat_history = []
            st.session_state.rag_chatbot.clear_history()
            st.rerun()
        
        st.metric("📚 Documents Indexed", 
                  len(st.session_state.rag_engine.vector_store.documents))
        
        # Optional API key
        st.markdown("---")
        api_key = st.text_input("Gemini API Key (Optional)", type="password", key="api_key_input")
        if api_key:
            st.session_state.rag_chatbot.api_key = api_key
    
    st.markdown("""
    Ask me anything about **cybersecurity**! I use Retrieval-Augmented Generation (RAG) 
    to search my knowledge base and provide accurate, contextual answers.
    """)
    
    # Quick questions
    st.markdown("#### 💡 Quick Questions")
    quick_cols = st.columns(3)
    quick_questions = [
        "How to prevent phishing?",
        "What is ransomware?",
        "DDoS mitigation strategies",
        "Incident response steps",
        "Cloud security practices",
        "Insider threat detection"
    ]
    
    for i, q in enumerate(quick_questions):
        if quick_cols[i % 3].button(q, key=f"quick_{i}"):
            st.session_state.quick_question = q
    
    st.markdown("---")
    
    # Display chat history
    for msg in st.session_state.rag_chat_history:
        render_chat_message(
            msg['role'], 
            msg['content'],
            msg.get('sources') if show_sources else None
        )
    
    # Chat input
    user_input = st.chat_input("Ask a cybersecurity question...")
    
    # Handle quick question
    if 'quick_question' in st.session_state:
        user_input = st.session_state.quick_question
        del st.session_state.quick_question
    
    if user_input:
        # Display user message
        st.session_state.rag_chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Get response
        with st.spinner("🔍 Searching knowledge base..."):
            response_data = st.session_state.rag_chatbot.get_response(
                user_input,
                use_api=bool(st.session_state.rag_chatbot.api_key)
            )
        
        # Add response to history
        st.session_state.rag_chat_history.append({
            'role': 'assistant',
            'content': response_data['response'],
            'sources': response_data['retrieved_docs']
        })
        
        st.rerun()


# ==================== MAIN DASHBOARD ====================
def run_integrated_dashboard():
    st.set_page_config(page_title="Cybersecurity Threat Predictor + RAG", layout="wide", page_icon="🛡️")
    st.title("🛡️ Integrated Cybersecurity Prediction Dashboard")
    st.markdown("*ML-powered threat prediction with RAG-enhanced AI assistant*")
    st.markdown("---")

    # Load models
    @st.cache_resource
    def load_all_models():
        try:
            models = {
                'Logistic Regression': joblib.load('baseline_model.pkl'),
                'Random Forest': joblib.load('random_forest_model.pkl'),
                'Gradient Boosting': joblib.load('gradient_boosting_model.pkl'),
                'Deep Neural Network (Incident)': load_model('incident_dl_model.h5')
            }
            incident_scaler = joblib.load('incident_scaler.pkl')
            le_attack = joblib.load('label_encoder_attack.pkl')
            le_industry = joblib.load('label_encoder_industry.pkl')
            le_country = joblib.load('label_encoder_country.pkl')
            url_model = load_model('url_dl_model.h5')
            url_scaler = joblib.load('url_scaler.pkl')
            url_feature_cols = joblib.load('url_feature_cols.pkl')
            
            return models, incident_scaler, le_attack, le_industry, le_country, url_model, url_scaler, url_feature_cols
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None, None, None, None, None, None, None, None

    models, incident_scaler, le_attack, le_industry, le_country, url_model, url_scaler, url_feature_cols = load_all_models()
    
    # Model comparison data
    comparison_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Deep Neural Network'],
        'Accuracy': [0.85, 0.92, 0.91, 0.89],
        'AUC Score': [0.88, 0.95, 0.94, 0.91]
    })

    # Create tabs including RAG
    if RAG_AVAILABLE and EMBEDDINGS_AVAILABLE:
        tab1, tab2, tab3 = st.tabs([
            "📊 Incident Risk Analyzer", 
            "🌐 URL Threat Checker",
            "🤖 AI Assistant (RAG)"
        ])
    else:
        tab1, tab2 = st.tabs([
            "📊 Incident Risk Analyzer", 
            "🌐 URL Threat Checker"
        ])
        st.sidebar.warning("⚠️ RAG not available. Install: `pip install sentence-transformers`")

    # ========== TAB 1: INCIDENT PREDICTION ==========
    with tab1:
        if models is None:
            st.warning("⚠️ Please run `complete_project.py` first to train and save the models.")
            return
            
        st.header("Incident Risk Analysis")
        st.subheader("Input Threat Parameters")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            financial_loss = st.slider("Financial Loss ($M)", 0.0, 200.0, 50.0, key='fl')
            affected_users = st.slider("Affected Users", 0, 50000, 10000, key='au')
            
        with col2:
            response_time = st.slider("Response Time (hours)", 0.0, 24.0, 5.0, key='rt')
            data_breach_size = st.slider("Data Breach Size (MB)", 0, 5000, 1000, key='dbs')
            
        with col3:
            network_traffic = st.slider("Network Traffic (GB)", 0.0, 2000.0, 800.0, key='nt')
            vulnerability_score = st.slider("Vulnerability Score", 1, 10, 5, key='vs')
        
        with col4:
            attack_type = st.selectbox("Attack Type", le_attack.classes_, key='at')
            industry = st.selectbox("Target Industry", le_industry.classes_, key='ti')
            country = st.selectbox("Country", le_country.classes_, key='ct')
            year = st.slider("Year", 2020, 2026, 2025, key='yr')
            month = st.slider("Month", 1, 12, 6, key='mo')
        
        loss_per_user = financial_loss / (affected_users + 1)
        
        input_data = pd.DataFrame({
            'financial_loss': [financial_loss],
            'affected_users': [affected_users],
            'response_time': [response_time],
            'data_breach_size': [data_breach_size],
            'network_traffic': [network_traffic],
            'vulnerability_score': [vulnerability_score],
            'loss_per_user': [loss_per_user],
            'attack_type_encoded': [le_attack.transform([attack_type])[0]],
            'industry_encoded': [le_industry.transform([industry])[0]],
            'country_encoded': [le_country.transform([country])[0]],
            'year': [year],
            'month': [month]
        })
        
        input_scaled = incident_scaler.transform(input_data)
        
        st.markdown("---")
        st.subheader("Model Predictions")
        
        cols = st.columns(4)
        
        for i, (name, model) in enumerate(models.items()):
            if "Deep Neural Network" in name:
                prob = model.predict(input_scaled, verbose=0)[0, 0]
            else:
                prob = model.predict_proba(input_scaled)[0, 1]
            
            risk = "HIGH RISK" if prob > 0.5 else "LOW RISK"
            color = "red" if prob > 0.5 else "green"
            
            with cols[i]:
                st.metric(f"{name.split()[0]}", f"{prob:.2%}")
                st.markdown(f"**{risk}**", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.bar(comparison_df, x='Model', y='Accuracy', color='Model', title="Accuracy")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.bar(comparison_df, x='Model', y='AUC Score', color='Model', title="AUC Score")
            st.plotly_chart(fig2, use_container_width=True)

    # ========== TAB 2: URL CHECKER ==========
    with tab2:
        st.header("🌐 URL Threat Detection")
        
        if url_model is None:
            st.warning("⚠️ URL model not loaded. Run `complete_project.py` first.")
            return

        user_url = st.text_input("🔗 Enter URL to check:", 
                                  value="http://secure-login.bank-update.com/verify-account")

        if st.button("🚨 Check URL Risk"):
            if user_url:
                features = extract_url_features(user_url)
                features_aligned = features.reindex(columns=url_feature_cols, fill_value=0)
                X_url_scaled_input = url_scaler.transform(features_aligned)
                
                prob = url_model.predict(X_url_scaled_input, verbose=0)[0, 0]
                prediction = 1 if prob > 0.5 else 0

                st.subheader("Analysis Result")
                
                if prediction == 1:
                    st.error(f"🚨 **MALICIOUS URL DETECTED** (Risk: {prob:.2%})")
                else:
                    st.success(f"✅ **URL Appears SAFE** (Risk: {prob:.2%})")
                    
                with st.expander("View Extracted Features"):
                    st.dataframe(features)

    # ========== TAB 3: RAG CHATBOT ==========
    if RAG_AVAILABLE and EMBEDDINGS_AVAILABLE:
        with tab3:
            run_rag_tab()


# ==================== MAIN ====================
if __name__ == "__main__":
    run_integrated_dashboard()
