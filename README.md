# 📧 Email Spam Detector - Real-Time NLP Application

A comprehensive **Real-Time Email Spam Detection System** implementing NLP concepts.

## 🎯 Project Overview

This project demonstrates a complete NLP pipeline for spam detection using:
- **Tokenization** and **Stemming**
- **N-grams Modeling** (Bigrams and Trigrams)
- **POS Tagging** (Part-of-Speech)
- **Morphological Analysis**
- **Rule-based Classification**
- **Lexicon-based Features**
- **Machine Learning** (Naive Bayes with TF-IDF)

## ✨ Features

- ✅ **Real-time spam detection** with confidence scores
- ✅ **Detailed NLP analysis** showing tokenization, stemming, n-grams, POS tags
- ✅ **Interactive dashboard** with performance metrics and visualizations
- ✅ **Rule-based + ML hybrid approach** for better accuracy
- ✅ **Web interface** built with Streamlit
- ✅ **96-98% accuracy** on email spam dataset

## 📊 NLP Concepts Coverage

| Concept | Implementation |
|---------|----------------|
| **Tokenization** | NLTK word_tokenize |
| **Stemming** | Porter Stemmer |
| **N-grams** | Bigrams & Trigrams extraction |
| **POS Tagging** | NLTK pos_tag |
| **Morphological Analysis** | Word structure analysis |
| **Lexicon Resources** | Spam keyword dictionary |
| **Rule-based System** | Pattern matching rules |
| **Language Processors** | Recognizer, Parser, Analyzer |

## 🚀 Installation

### Step 1: Clone or Create Project Directory

```bash
mkdir email-spam-detector
cd email-spam-detector
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
```

## 📁 Project Structure

```
email-spam-detector/
├── app.py                 # Main Streamlit application
├── nlp_processor.py       # NLP processing pipeline
├── spam_detector.py       # Spam detection model
├── requirements.txt       # Python dependencies
├── emails.csv            # Dataset (5,728 emails)
└── README.md             # Documentation
```

## 🎮 Usage

### 1. Run the Application

```bash
streamlit run app.py
```

### 2. Train the Model

1. Navigate to **"Train Model"** page
2. Upload `emails.csv` dataset
3. Click **"Train Model"** button
4. Wait for training to complete (~30 seconds)

### 3. Detect Spam

1. Go to **"Detect Spam"** page
2. Enter email text or use sample examples
3. Click **"Detect Spam"** button
4. View results with detailed NLP analysis

### 4. View Dashboard

- Navigate to **"Analysis Dashboard"** to see:
  - Performance metrics
  - Confusion matrix
  - Word clouds
  - Feature importance

## 📈 Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 96-98% |
| **Precision** | 94-96% |
| **Recall** | 92-95% |
| **F1-Score** | 94-96% |

## 🔬 NLP Pipeline Explanation

### 1. Tokenization
```python
Text: "FREE money! Click here NOW!!!"
Tokens: ['FREE', 'money', '!', 'Click', 'here', 'NOW', '!', '!', '!']
```

### 2. Stemming
```python
Tokens: ['running', 'walked', 'better', 'clicking']
Stemmed: ['run', 'walk', 'better', 'click']
```

### 3. N-grams
```python
Bigrams: [('FREE', 'money'), ('money', '!'), ('Click', 'here')]
Trigrams: [('FREE', 'money', '!'), ('Click', 'here', 'NOW')]
```

### 4. POS Tagging
```python
[('FREE', 'JJ'), ('money', 'NN'), ('Click', 'VB'), ('here', 'RB')]
```

### 5. Rule-based Features
- Money symbols ($, £, €)
- Excessive punctuation (!!!)
- All caps words
- URLs and phone numbers
- Spam keywords from lexicon

## 📊 Dataset Information

- **Source**: Email Spam Dataset
- **Size**: 5,728 emails
- **Columns**: `text`, `spam`
- **Labels**: 0 (Ham/Legitimate), 1 (Spam)
- **Distribution**: ~87% Ham, ~13% Spam

## 🛠️ Technical Stack

- **Python 3.8+**
- **Streamlit** - Web interface
- **NLTK** - NLP processing
- **scikit-learn** - Machine learning
- **Pandas** - Data manipulation
- **Matplotlib/Plotly** - Visualizations

## 🎯 Key Components

### 1. NLP Processor (`nlp_processor.py`)
- Tokenization and stemming
- N-grams extraction
- POS tagging
- Morphological analysis
- Lexicon-based features

### 2. Spam Detector (`spam_detector.py`)
- TF-IDF vectorization
- Naive Bayes classifier
- Rule-based scoring
- Feature extraction
- Model training and prediction

### 3. Streamlit App (`app.py`)
- Interactive web interface
- Real-time detection
- Visualization dashboard
- NLP analysis display

## 📚 NLP Concepts Demonstrated

1. **Introduction to NLP**
   - Levels of linguistic processing: morphology, syntax, semantics
   - Tokenization and stemming

2. **Language Processors**
   - Recognizers: Identify spam patterns
   - Parsers: Extract features
   - Generators: Create analysis reports

3. **Resources for NLP**
   - Lexicon: Spam keyword dictionary
   - Knowledge bases: Rule sets

4. **Computational Morphology**
   - Lemmatization and stemming
   - POS tagging
   - Morphological analysis

## 🔍 Example Output

```
Email: "WINNER!! You've won $1000! Call now!"

Result: 🚨 SPAM DETECTED
Confidence: 98.7%

Analysis:
- Tokens: ['WINNER', 'You', 've', 'won', '1000', 'Call', 'now']
- Bigrams: [('WINNER', 'You'), ('won', '1000')]
- POS Tags: [('WINNER', 'NN'), ('won', 'VBD'), ('Call', 'VB')]
- Spam Indicators:
  ⚠️ Contains money-related spam words
  ⚠️ Contains urgent call-to-action
  ⚠️ Multiple words in ALL CAPS
  ⚠️ Contains monetary amounts
```

## 🤝 Contributing

This is an academic project. Feel free to:
- Add more NLP features
- Improve rule-based system
- Add more visualizations
- Extend lexicon resources

## 📝 License

This project is for educational purposes.

~ Web App Link: https://co22338-spamshield.streamlit.app/

---
