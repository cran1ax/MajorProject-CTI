# ==================== STREAMLIT RAG CHATBOT INTERFACE ====================
"""
Streamlit interface for the Cybersecurity RAG Chatbot
Integrates with the main dashboard
"""

import streamlit as st
import pandas as pd
from typing import List, Dict
import time

# Import RAG system
try:
    from rag_system import RAGEngine, RAGChatbot, RAGConfig, EMBEDDINGS_AVAILABLE, create_rag_system
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"RAG system not available: {e}")


def initialize_rag_session():
    """Initialize RAG system in session state"""
    if 'rag_engine' not in st.session_state:
        if RAG_AVAILABLE and EMBEDDINGS_AVAILABLE:
            with st.spinner("🔄 Initializing RAG Knowledge Base..."):
                config = RAGConfig()
                st.session_state.rag_engine = RAGEngine(config)
                st.session_state.rag_engine.initialize_knowledge_base()
                st.session_state.chatbot = RAGChatbot(st.session_state.rag_engine)
            st.success("✅ RAG System initialized!")
        else:
            st.session_state.rag_engine = None
            st.session_state.chatbot = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def render_chat_message(role: str, content: str, sources: List[Dict] = None):
    """Render a chat message with optional sources"""
    if role == "user":
        st.markdown(f"""
        <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0;">
            <strong>🧑 You:</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #f5f5f5; padding: 10px; border-radius: 10px; margin: 5px 0;">
            <strong>🤖 CyberBot:</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)
        
        if sources:
            with st.expander(f"📚 View {len(sources)} Source(s)"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"""
                    **Source {i}: {source.get('title', 'Unknown')}**  
                    *Category: {source.get('category', 'N/A')} | Relevance: {source.get('score', 0):.0%}*  
                    > {source.get('content', '')[:300]}...
                    """)


def run_rag_chatbot_tab():
    """Main function to run the RAG chatbot tab in Streamlit"""
    
    st.header("🤖 AI Cybersecurity Assistant (RAG-Powered)")
    st.markdown("""
    This assistant uses **Retrieval-Augmented Generation (RAG)** to provide accurate, 
    contextual answers about cybersecurity topics by retrieving relevant information 
    from a curated knowledge base.
    """)
    
    # Check if RAG is available
    if not RAG_AVAILABLE:
        st.error("❌ RAG system module not found. Ensure `rag_system.py` is in the same directory.")
        return
    
    if not EMBEDDINGS_AVAILABLE:
        st.error("""
        ❌ **sentence-transformers** is required for the RAG system.
        
        Install it by running:
        ```bash
        pip install sentence-transformers
        ```
        """)
        return
    
    # Initialize RAG
    initialize_rag_session()
    
    if st.session_state.rag_engine is None:
        st.warning("RAG system could not be initialized.")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("⚙️ RAG Settings")
        top_k = st.slider("Number of sources to retrieve", 1, 10, 5)
        show_sources = st.checkbox("Show retrieved sources", value=True)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.chatbot.clear_history()
            st.rerun()
        
        st.markdown("---")
        st.subheader("📊 Knowledge Base Stats")
        if st.session_state.rag_engine.is_initialized:
            st.metric("Documents Indexed", len(st.session_state.rag_engine.vector_store.documents))
        
        # API Key input (optional)
        st.markdown("---")
        st.subheader("🔑 API Configuration (Optional)")
        api_key = st.text_input("Gemini API Key", type="password", 
                                help="Optional: Enter API key for enhanced AI responses")
        if api_key:
            st.session_state.chatbot.api_key = api_key
    
    # Chat interface
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.chat_history:
        render_chat_message(
            message['role'], 
            message['content'],
            message.get('sources') if show_sources else None
        )
    
    # Quick question buttons
    st.markdown("### 💡 Quick Questions")
    col1, col2, col3 = st.columns(3)
    
    quick_questions = [
        "How do I prevent phishing attacks?",
        "What is ransomware and how to respond?",
        "Explain the incident response process",
        "What are DDoS attack mitigations?",
        "How to detect insider threats?",
        "Cloud security best practices?"
    ]
    
    selected_question = None
    for i, question in enumerate(quick_questions):
        col = [col1, col2, col3][i % 3]
        if col.button(question, key=f"quick_{i}"):
            selected_question = question
    
    # Chat input
    st.markdown("---")
    user_input = st.text_input("💬 Ask a cybersecurity question:", 
                                value=selected_question if selected_question else "",
                                key="user_input",
                                placeholder="E.g., How can I protect against SQL injection attacks?")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        send_button = st.button("📤 Send", type="primary")
    with col2:
        use_api = st.checkbox("Use AI API (if configured)", value=False)
    
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Get RAG response
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                response_data = st.session_state.chatbot.get_response(
                    user_input, 
                    use_api=(use_api and st.session_state.chatbot.api_key)
                )
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response_data['response'],
                    'sources': response_data['retrieved_docs'] if show_sources else []
                })
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
        
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 12px;">
        🛡️ Powered by RAG (Retrieval-Augmented Generation) | 
        Knowledge base includes: Phishing, Ransomware, DDoS, Malware, Insider Threats, 
        Incident Response, Threat Intelligence, Network Security, Compliance, Cloud Security
    </div>
    """, unsafe_allow_html=True)


def run_standalone_rag_app():
    """Run the RAG chatbot as a standalone Streamlit app"""
    st.set_page_config(
        page_title="Cybersecurity RAG Assistant",
        page_icon="🛡️",
        layout="wide"
    )
    
    st.title("🛡️ Cybersecurity RAG Assistant")
    run_rag_chatbot_tab()


if __name__ == "__main__":
    run_standalone_rag_app()
