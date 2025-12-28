"""
SpamShield - A Real-Time Email Spam Detector
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
from spam_detector import SpamDetector
from nlp_processor import NLPProcessor
import time

# Page configuration
st.set_page_config(
    page_title="SpamShield",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .spam-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .ham-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = SpamDetector()
    st.session_state.nlp = NLPProcessor()
    st.session_state.trained = False
    st.session_state.df = None

# Header
st.markdown('<div class="main-header">📧 SpamShield - An Email Spam Detector</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "🤖 Train Model", "🔍 Detect Spam", "📊 Analysis Dashboard", "📚 About NLP"]
)

st.sidebar.markdown("---")

# ============================================================================
# PAGE 1: HOME
# ============================================================================
if page == "🏠 Home":
    st.header("Welcome to Email Spam Detector! 👋")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        This is a real-time NLP application that detects spam emails using advanced 
        Natural Language Processing techniques.
        
        ### ✨ Key Features
        - 🔤 **Tokenization & Stemming**
        - 📊 **N-grams Analysis** (Bigrams & Trigrams)
        - 🏷️ **POS Tagging** (Part-of-Speech)
        - 🧠 **Morphological Analysis**
        - 📚 **Lexicon-based Detection**
        - ⚡ **Rule-based Classification**
        - 🤖 **Machine Learning Model** (Naive Bayes)
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Expected Performance
        - **Accuracy**: 96-98%
        - **Precision**: 94-96%
        - **Recall**: 92-95%
        - **F1-Score**: 94-96%
        
        ### 🚀 How to Use
        1. **Train Model**: Upload dataset and train
        2. **Detect Spam**: Enter email text for real-time detection
        3. **View Analysis**: Explore NLP processing details
        4. **Dashboard**: Visualize model performance
        """)
    
    st.markdown("---")
    
    # Quick Stats
    if st.session_state.trained:
        st.success("✅ Model is trained and ready!")
        metrics = st.session_state.detector.get_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("Precision", f"{metrics['precision']:.2%}")
        col3.metric("Recall", f"{metrics['recall']:.2%}")
        col4.metric("F1-Score", f"{metrics['f1_score']:.2%}")
    else:
        st.warning("⚠️ Model not trained yet. Please go to 'Train Model' page.")
    
    # Sample demonstration
    st.markdown("---")
    st.subheader("🧪 Quick Demo")
    
    demo_text = st.text_area(
        "Try a sample email:",
        "WINNER!! You have been selected to receive £1000 cash prize! Call now 09061701461 to claim your reward. Limited time offer!",
        height=100
    )
    
    if st.button("🔍 Analyze Demo", type="primary"):
        with st.spinner("Analyzing..."):
            analysis = st.session_state.nlp.analyze_text_detailed(demo_text)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Tokens (First 15):**")
                st.code(', '.join(analysis['tokens'][:15]))
                
                st.write("**Bigrams (First 5):**")
                for bg in analysis['bigrams'][:5]:
                    st.write(f"• {bg}")
            
            with col2:
                st.write("**POS Tags (First 10):**")
                for word, tag in analysis['pos_tags'][:10]:
                    st.write(f"• {word}: `{tag}`")
                
                st.write("**Spam Indicators:**")
                for indicator in analysis['spam_indicators']:
                    st.write(f"⚠️ {indicator}")

# ============================================================================
# PAGE 2: TRAIN MODEL
# ============================================================================
elif page == "🤖 Train Model":
    st.header("🤖 Train Spam Detection Model")
    
    st.markdown("""
    Upload your dataset (emails.csv if no dataset available) to train the spam detector.
    The model uses Naive Bayes classifier with TF-IDF vectorization.
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload dataset:", type=['csv'])
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        st.success(f"✅ Dataset loaded: {len(df)} emails")
        
        # Display dataset info
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Emails", len(df))
        col2.metric("Spam Emails", int(df['spam'].sum()))
        col3.metric("Ham Emails", int(len(df) - df['spam'].sum()))
        
        # Show sample data
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Distribution chart
        st.subheader("📊 Class Distribution")
        fig = px.pie(
            values=[int(df['spam'].sum()), int(len(df) - df['spam'].sum())],
            names=['Spam', 'Ham'],
            color_discrete_sequence=['#f44336', '#4caf50']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Train button
        st.markdown("---")
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            with st.spinner("Training model... This may take a minute."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Training
                status_text.text("Preprocessing data...")
                progress_bar.progress(20)
                time.sleep(0.5)
                
                status_text.text("Extracting features...")
                progress_bar.progress(40)
                time.sleep(0.5)
                
                status_text.text("Training classifier...")
                progress_bar.progress(60)
                
                metrics = st.session_state.detector.train(df)
                
                progress_bar.progress(100)
                status_text.text("Training completed!")
                time.sleep(0.5)
                
                st.session_state.trained = True
                
                # Display results
                st.success("🎉 Model trained successfully!")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                col2.metric("Precision", f"{metrics['precision']:.2%}")
                col3.metric("Recall", f"{metrics['recall']:.2%}")
                col4.metric("F1-Score", f"{metrics['f1_score']:.2%}")
                
                # Confusion Matrix
                st.subheader("📊 Confusion Matrix")
                cm = metrics['confusion_matrix']
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam']
                )
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
                
                st.balloons()
    
    else:
        st.info("👆 Please upload your dataset file to begin training.")

# ============================================================================
# PAGE 3: DETECT SPAM
# ============================================================================
elif page == "🔍 Detect Spam":
    st.header("🔍 Real-Time Spam Detection")
    
    if not st.session_state.trained:
        st.error("❌ Model not trained! Please train the model first in the 'Train Model' page.")
    else:
        st.markdown("Enter an email text below to detect if it's spam or legitimate.")
        
        # Sample emails
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Load Spam Example"):
                st.session_state.sample_text = "WINNER!! You have been selected to receive £1000 cash or a £2000 prize! Call 09050000327. Claim NOW!"
        
        with col2:
            if st.button("📝 Load Ham Example"):
                st.session_state.sample_text = "Hi John, I hope this email finds you well. I wanted to follow up on our meeting yesterday. Please let me know your availability for next week."
        
        # Text input
        email_text = st.text_area(
            "Email Text:",
            value=st.session_state.get('sample_text', ''),
            height=150,
            placeholder="Enter email text here..."
        )
        
        if st.button("🔍 Detect Spam", type="primary", use_container_width=True):
            if email_text.strip():
                with st.spinner("Analyzing email..."):
                    result = st.session_state.detector.predict(email_text)
                    
                    # Display result
                    if result['is_spam']:
                        st.markdown(f"""
                        <div class="spam-box">
                            <h2>🚨 SPAM DETECTED</h2>
                            <h3>Confidence: {result['confidence']:.1%}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="ham-box">
                            <h2>✅ LEGITIMATE EMAIL</h2>
                            <h3>Confidence: {result['confidence']:.1%}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Metrics
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Spam Probability", f"{result['spam_probability']:.1%}")
                    col2.metric("Ham Probability", f"{result['ham_probability']:.1%}")
                    col3.metric("Rule-based Score", f"{result['rule_based_score']}/100")
                    
                    # Detailed Analysis
                    st.markdown("---")
                    st.subheader("🔬 Detailed NLP Analysis")
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["📝 Tokenization", "🔤 N-grams", "🏷️ POS Tags", "⚠️ Spam Indicators"])
                    
                    with tab1:
                        st.write("**Tokens (First 30):**")
                        st.code(', '.join(result['analysis']['tokens']))
                        
                        st.write("**Stemmed Tokens (First 30):**")
                        st.code(', '.join(result['analysis']['stemmed_tokens']))
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Total Tokens", result['analysis']['statistics']['total_tokens'])
                        col2.metric("Unique Tokens", result['analysis']['statistics']['unique_tokens'])
                    
                    with tab2:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Bigrams (Top 10):**")
                            for bg in result['analysis']['bigrams']:
                                st.write(f"• {bg}")
                        
                        with col2:
                            st.write("**Trigrams (Top 10):**")
                            for tg in result['analysis']['trigrams']:
                                st.write(f"• {tg}")
                    
                    with tab3:
                        st.write("**Part-of-Speech Tags (First 20):**")
                        pos_df = pd.DataFrame(result['analysis']['pos_tags'], columns=['Word', 'POS Tag'])
                        st.dataframe(pos_df, use_container_width=True)
                    
                    with tab4:
                        if result['analysis']['spam_indicators']:
                            st.write("**Detected Spam Indicators:**")
                            for indicator in result['analysis']['spam_indicators']:
                                st.warning(f"⚠️ {indicator}")
                        else:
                            st.success("✅ No spam indicators detected")
                        
                        # Feature breakdown
                        st.write("**Key Features:**")
                        features = result['features']
                        feature_data = {
                            'Feature': ['Exclamation Marks', 'All Caps Words', 'Money Symbols', 'URLs', 'Spam Keywords'],
                            'Value': [
                                features['exclamation_count'],
                                features['all_caps_words'],
                                features['has_dollar'] + features['has_pound'] + features['has_euro'],
                                features['has_url'],
                                features['spam_words_money'] + features['spam_words_urgency'] + features['spam_words_action']
                            ]
                        }
                        st.bar_chart(pd.DataFrame(feature_data).set_index('Feature'))
            
            else:
                st.warning("⚠️ Please enter email text to analyze.")

# ============================================================================
# PAGE 4: ANALYSIS DASHBOARD
# ============================================================================
elif page == "📊 Analysis Dashboard":
    st.header("📊 Model Performance Dashboard")
    
    if not st.session_state.trained:
        st.error("❌ Model not trained! Please train the model first.")
    else:
        metrics = st.session_state.detector.get_metrics()
        
        # Performance Metrics
        st.subheader("🎯 Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}", "High")
        col2.metric("Precision", f"{metrics['precision']:.2%}", "High")
        col3.metric("Recall", f"{metrics['recall']:.2%}", "High")
        col4.metric("F1-Score", f"{metrics['f1_score']:.2%}", "High")
        
        # Confusion Matrix
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Confusion Matrix")
            cm = metrics['confusion_matrix']
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='RdYlGn',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                cbar_kws={'label': 'Count'}
            )
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
        
        with col2:
            st.subheader("📈 Metrics Comparison")
            metrics_data = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Score': [
                    metrics['accuracy'],
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1_score']
                ]
            })
            fig = px.bar(
                metrics_data,
                x='Metric',
                y='Score',
                color='Score',
                color_continuous_scale='Viridis',
                text='Score'
            )
            fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig.update_layout(showlegend=False, yaxis_range=[0, 1.1])
            st.plotly_chart(fig, use_container_width=True)
        
        # Dataset Statistics
        st.markdown("---")
        st.subheader("📊 Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Training Samples", metrics['train_size'])
        col2.metric("Test Samples", metrics['test_size'])
        col3.metric("Total Spam", metrics['spam_count'])
        col4.metric("Total Ham", metrics['ham_count'])
        
        # Word Cloud
        if st.session_state.df is not None:
            st.markdown("---")
            st.subheader("☁️ Word Clouds")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Spam Emails Word Cloud**")
                spam_text = ' '.join(st.session_state.df[st.session_state.df['spam']==1]['text'].values)
                wordcloud_spam = WordCloud(
                    width=400, 
                    height=300, 
                    background_color='white',
                    colormap='Reds'
                ).generate(spam_text)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud_spam, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            
            with col2:
                st.write("**Ham Emails Word Cloud**")
                ham_text = ' '.join(st.session_state.df[st.session_state.df['spam']==0]['text'].values)
                wordcloud_ham = WordCloud(
                    width=400, 
                    height=300, 
                    background_color='white',
                    colormap='Greens'
                ).generate(ham_text)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud_ham, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

# ============================================================================
# PAGE 5: ABOUT NLP
# ============================================================================
elif page == "📚 About NLP":
    st.header("📚 Covered Natural Language Processing Concepts")
    
    # Section A Topics
    with st.expander("🔤 Tokenization", expanded=True):
        st.markdown("""
        **Tokenization** is the process of breaking text into smaller units called tokens (words, punctuation).
        
        **Example:**
        ```
        Text: "Hello! How are you?"
        Tokens: ['Hello', '!', 'How', 'are', 'you', '?']
        ```
        
        **Implementation:** Using NLTK's `word_tokenize()`
        """)
    
    with st.expander("🌱 Stemming"):
        st.markdown("""
        **Stemming** reduces words to their root/base form by removing suffixes.
        
        **Example:**
        ```
        running → run
        walked → walk
        better → better (irregular)
        ```
        
        **Implementation:** Porter Stemmer algorithm
        """)
    
    with st.expander("📊 N-grams Modeling"):
        st.markdown("""
        **N-grams** are contiguous sequences of n items from text.
        
        **Bigrams (n=2):**
        ```
        Text: "I love NLP"
        Bigrams: [('I', 'love'), ('love', 'NLP')]
        ```
        
        **Trigrams (n=3):**
        ```
        Text: "I love NLP projects"
        Trigrams: [('I', 'love', 'NLP'), ('love', 'NLP', 'projects')]
        ```
        
        **Use in Spam Detection:** Phrases like "click here", "free money" are strong spam indicators.
        """)
    
    with st.expander("🏷️ POS Tagging"):
        st.markdown("""
        **Part-of-Speech Tagging** assigns grammatical categories to each word.
        
        **Common Tags:**
        - **NN**: Noun
        - **VB**: Verb
        - **JJ**: Adjective
        - **RB**: Adverb
        
        **Example:**
        ```
        "The quick brown fox jumps"
        [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'), ('jumps', 'VBZ')]
        ```
        
        **Use in Spam Detection:** Spam emails often have excessive verbs (CLICK, BUY, CALL).
        """)
    
    with st.expander("🧬 Morphological Analysis"):
        st.markdown("""
        **Morphological Analysis** studies word structure and formation.
        
        Includes:
        - Word segmentation
        - Lemmatization
        - Handling inflections
        
        **Example:**
        ```
        "running" = run (stem) + -ing (suffix)
        "unhappy" = un- (prefix) + happy (root)
        ```
        """)
    
    with st.expander("📚 Lexicon Resources"):
        st.markdown("""
        **Lexicons** are dictionaries of words with associated properties.
        
        **Spam Lexicon used in this project:**
        - Money words: free, cash, prize, winner
        - Urgency words: now, urgent, limited, today
        - Action words: click, call, buy, subscribe
        - Suspicious: viagra, pills, weight loss
        
        **Use:** Words matched against lexicon increase spam score.
        """)
    
    with st.expander("⚙️ Rule-based System"):
        st.markdown("""
        **Rule-based Classification** uses predefined rules to make decisions.
        
        **Spam Rules in this project:**
        ```python
        if text.contains("FREE") and text.contains("$"):
            spam_score += 20
        
        if text.count("!") >= 3:
            spam_score += 15
        
        if text.has_url():
            spam_score += 10
        ```
        
        **Advantage:** Interpretable, no training needed.
        **Disadvantage:** Requires manual rule creation.
        """)
    
    with st.expander("🤖 Language Processors"):
        st.markdown("""
        **Language Processors** are systems that process natural language:
        
        1. **Recognizers**: Identify patterns (spam indicators)
        2. **Parsers**: Analyze structure (syntax)
        3. **Generators**: Create text (email responses)
        
        **In this project:**
        - Recognizer: Identifies spam patterns
        - Parser: Extracts features from email structure
        - Generator: Creates analysis reports
        """)
    
    st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📧 SpamShield | Developed using Python and NLTK</p>
</div>
""", unsafe_allow_html=True)