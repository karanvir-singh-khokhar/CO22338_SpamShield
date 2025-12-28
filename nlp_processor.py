"""
NLP Processor Module
"""

import re
import nltk
from collections import Counter
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag, ngrams


class NLPProcessor:
    """Comprehensive NLP Processing Pipeline for Email Analysis"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Spam keywords lexicon (Resources for NLP)
        self.spam_lexicon = {
            'money': ['free', 'cash', 'money', 'prize', 'winner', 'won', 'win'],
            'urgency': ['urgent', 'now', 'immediately', 'hurry', 'limited', 'today'],
            'action': ['click', 'call', 'buy', 'order', 'subscribe', 'claim'],
            'offers': ['offer', 'discount', 'deal', 'promotion', 'bonus'],
            'suspicious': ['viagra', 'pills', 'weight', 'loss', 'guarantee']
        }
    
    def tokenize(self, text):
        """
        Tokenization: Breaking text into tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize using NLTK
        tokens = word_tokenize(text)
        
        return tokens
    
    def stem_tokens(self, tokens):
        """
        Stemming: Reduce words to their root form
        """
        stemmed = [self.stemmer.stem(token) for token in tokens]
        return stemmed
    
    def extract_ngrams(self, tokens, n=2):
        """
        N-grams Modeling: Extract bigrams and trigrams
        """
        n_grams = list(ngrams(tokens, n))
        return n_grams
    
    def pos_tagging(self, tokens):
        """
        POS Tagging: Part-of-Speech tagging
        """
        pos_tags = pos_tag(tokens)
        return pos_tags
    
    def morphological_analysis(self, text):
        """
        Morphological Analysis: Analyze word structure
        """
        tokens = self.tokenize(text)
        
        analysis = {
            'original_tokens': tokens,
            'stemmed_tokens': self.stem_tokens(tokens),
            'pos_tags': self.pos_tagging(tokens),
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens))
        }
        
        return analysis
    
    def extract_features(self, text):
        """
        Comprehensive feature extraction using all NLP techniques
        """
        # Tokenization
        tokens = self.tokenize(text)
        tokens_clean = [t for t in tokens if t.isalnum()]
        
        # Stemming
        stemmed = self.stem_tokens(tokens_clean)
        
        # N-grams
        bigrams = self.extract_ngrams(tokens_clean, 2)
        trigrams = self.extract_ngrams(tokens_clean, 3)
        
        # POS Tagging
        pos_tags = self.pos_tagging(tokens_clean)
        
        # Extract POS counts
        pos_counts = Counter([tag for _, tag in pos_tags])
        
        # Rule-based features
        features = {
            # Basic features
            'token_count': len(tokens_clean),
            'unique_token_count': len(set(tokens_clean)),
            'avg_token_length': np.mean([len(t) for t in tokens_clean]) if tokens_clean else 0,
            
            # Character features
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_count': sum(1 for c in text if c.isupper()),
            'digit_count': sum(1 for c in text if c.isdigit()),
            'special_char_count': len(re.findall(r'[^a-zA-Z0-9\s]', text)),
            
            # Money and currency
            'has_dollar': 1 if '$' in text else 0,
            'has_pound': 1 if '£' in text else 0,
            'has_euro': 1 if '€' in text else 0,
            'money_pattern': len(re.findall(r'\$\d+|\£\d+|€\d+', text)),
            
            # URL and email patterns
            'has_url': 1 if re.search(r'http[s]?://|www\.', text) else 0,
            'has_email': 1 if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text) else 0,
            'has_phone': 1 if re.search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', text) else 0,
            
            # All caps words
            'all_caps_words': len(re.findall(r'\b[A-Z]{3,}\b', text)),
            
            # POS tag features
            'verb_count': pos_counts.get('VB', 0) + pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0),
            'noun_count': pos_counts.get('NN', 0) + pos_counts.get('NNS', 0),
            'adjective_count': pos_counts.get('JJ', 0),
            
            # Lexicon-based features (Resources for NLP)
            'spam_words_money': sum(1 for word in tokens_clean if word in self.spam_lexicon['money']),
            'spam_words_urgency': sum(1 for word in tokens_clean if word in self.spam_lexicon['urgency']),
            'spam_words_action': sum(1 for word in tokens_clean if word in self.spam_lexicon['action']),
            'spam_words_offers': sum(1 for word in tokens_clean if word in self.spam_lexicon['offers']),
            'spam_words_suspicious': sum(1 for word in tokens_clean if word in self.spam_lexicon['suspicious']),
            
            # N-gram features
            'bigram_count': len(bigrams),
            'trigram_count': len(trigrams),
        }
        
        return features, {
            'tokens': tokens_clean,
            'stemmed': stemmed,
            'bigrams': bigrams[:10],  # Top 10 bigrams
            'trigrams': trigrams[:10],  # Top 10 trigrams
            'pos_tags': pos_tags[:20]  # First 20 POS tags
        }
    
    def get_spam_indicators(self, text):
        """
        Rule-based spam detection indicators
        Language as a rule-based system
        """
        indicators = []
        
        # Check for spam patterns
        if re.search(r'free|prize|winner|won', text, re.IGNORECASE):
            indicators.append("Contains money-related spam words")
        
        if re.search(r'click here|click now|call now', text, re.IGNORECASE):
            indicators.append("Contains urgent call-to-action")
        
        if text.count('!') >= 3:
            indicators.append(f"Excessive exclamation marks ({text.count('!')})")
        
        if len(re.findall(r'\b[A-Z]{3,}\b', text)) >= 2:
            indicators.append("Multiple words in ALL CAPS")
        
        if re.search(r'\$\d+|\£\d+', text):
            indicators.append("Contains monetary amounts")
        
        if re.search(r'http[s]?://|www\.', text):
            indicators.append("Contains URLs")
        
        return indicators
    
    def analyze_text_detailed(self, text):
        """
        Complete linguistic analysis
        Covers: morphology, syntax, semantics
        """
        # Tokenization
        tokens = self.tokenize(text)
        tokens_clean = [t for t in tokens if t.isalnum()]
        
        # Morphological analysis
        stemmed = self.stem_tokens(tokens_clean)
        
        # Syntactic analysis (POS tagging)
        pos_tags = self.pos_tagging(tokens_clean)
        
        # N-grams (Syntax level)
        bigrams = self.extract_ngrams(tokens_clean, 2)
        trigrams = self.extract_ngrams(tokens_clean, 3)
        
        # Semantic analysis (Lexicon matching)
        spam_indicators = self.get_spam_indicators(text)
        
        return {
            'tokens': tokens_clean[:30],  # First 30 tokens
            'stemmed_tokens': stemmed[:30],
            'pos_tags': pos_tags[:20],
            'bigrams': [' '.join(bg) for bg in bigrams[:10]],
            'trigrams': [' '.join(tg) for tg in trigrams[:10]],
            'spam_indicators': spam_indicators,
            'statistics': {
                'total_tokens': len(tokens_clean),
                'unique_tokens': len(set(tokens_clean)),
                'total_bigrams': len(bigrams),
                'total_trigrams': len(trigrams)
            }
        }
