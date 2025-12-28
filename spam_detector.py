"""
Spam Detector Module
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nlp_processor import NLPProcessor
import pickle


class SpamDetector:
    """Email Spam Detector using NLP techniques and Machine Learning"""
    
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        self.model = MultinomialNB()
        self.is_trained = False
        self.metrics = {}
        
    def prepare_text(self, text):
        """Preprocess text using NLP pipeline"""
        tokens = self.nlp_processor.tokenize(text)
        tokens_clean = [t for t in tokens if t.isalnum()]
        stemmed = self.nlp_processor.stem_tokens(tokens_clean)
        return ' '.join(stemmed)
    
    def train(self, df):
        """
        Train the spam detector
        Uses TF-IDF vectorization and Naive Bayes
        """
        print("🔄 Starting training process...")
        
        # Prepare data
        X = df['text'].apply(self.prepare_text)
        y = df['spam']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {len(X_train)} emails")
        print(f"📊 Test set: {len(X_test)} emails")
        
        # Vectorize
        print("🔤 Vectorizing text data...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        print("🤖 Training Naive Bayes classifier...")
        self.model.fit(X_train_vec, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_vec)
        y_pred_proba = self.model.predict_proba(X_test_vec)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'spam_count': int(y.sum()),
            'ham_count': int(len(y) - y.sum())
        }
        
        self.is_trained = True
        
        print("✅ Training completed!")
        print(f"📈 Accuracy: {self.metrics['accuracy']:.2%}")
        print(f"📈 Precision: {self.metrics['precision']:.2%}")
        print(f"📈 Recall: {self.metrics['recall']:.2%}")
        print(f"📈 F1-Score: {self.metrics['f1_score']:.2%}")
        
        return self.metrics
    
    def predict(self, text):
        """
        Predict if email is spam or ham
        Returns prediction, probability, and analysis
        """
        if not self.is_trained:
            raise Exception("Model not trained! Train the model first.")
        
        # Prepare text
        prepared_text = self.prepare_text(text)
        text_vec = self.vectorizer.transform([prepared_text])
        
        # Predict
        prediction = self.model.predict(text_vec)[0]
        probability = self.model.predict_proba(text_vec)[0]
        
        # Get detailed NLP analysis
        analysis = self.nlp_processor.analyze_text_detailed(text)
        
        # Get features
        features, nlp_details = self.nlp_processor.extract_features(text)
        
        # Rule-based score (complementary to ML)
        rule_based_score = self._calculate_rule_based_score(features)
        
        return {
            'prediction': 'SPAM' if prediction == 1 else 'HAM (Legitimate)',
            'is_spam': bool(prediction),
            'confidence': float(probability[1] if prediction == 1 else probability[0]),
            'spam_probability': float(probability[1]),
            'ham_probability': float(probability[0]),
            'rule_based_score': rule_based_score,
            'analysis': analysis,
            'features': features,
            'nlp_details': nlp_details
        }
    
    def _calculate_rule_based_score(self, features):
        """
        Calculate spam score based on rules
        Language as a rule-based system
        """
        score = 0
        max_score = 100
        
        # Money indicators
        if features['has_dollar'] or features['has_pound'] or features['has_euro']:
            score += 10
        
        if features['money_pattern'] > 0:
            score += 15
        
        # Urgency indicators
        if features['exclamation_count'] >= 3:
            score += 15
        
        if features['all_caps_words'] >= 2:
            score += 10
        
        # Spam lexicon
        if features['spam_words_money'] > 0:
            score += 10
        
        if features['spam_words_urgency'] > 0:
            score += 10
        
        if features['spam_words_action'] > 0:
            score += 10
        
        # URL and suspicious patterns
        if features['has_url']:
            score += 10
        
        if features['special_char_count'] > 20:
            score += 10
        
        return min(score, max_score)
    
    def get_metrics(self):
        """Get training metrics"""
        return self.metrics
    
    def save_model(self, filepath='spam_model.pkl'):
        """Save trained model"""
        if not self.is_trained:
            raise Exception("Model not trained!")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'model': self.model,
                'metrics': self.metrics
            }, f)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='spam_model.pkl'):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.model = data['model']
            self.metrics = data['metrics']
            self.is_trained = True
        
        print(f"✅ Model loaded from {filepath}")