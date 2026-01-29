"""
Intent Matcher
Maps user speech to audio response files using keyword matching
"""

import logging
from typing import Optional
from fuzzywuzzy import fuzz
import time

logger = logging.getLogger(__name__)


class IntentMatcher:
    """
    Matches user speech to intents using fuzzy keyword matching
    
    Features:
    - Fuzzy string matching for robustness
    - Configurable similarity threshold
    - Multiple keywords per intent
    - Fallback to default response
    """
    
    def __init__(self, 
                 intent_keywords: dict,
                 fuzzy_threshold: int = 70,
                 default_intent: str = "default"):
        """
        Initialize intent matcher
        
        Args:
            intent_keywords: Dict mapping keywords to audio files
            fuzzy_threshold: Minimum similarity score (0-100)
            default_intent: Default intent key for unknown input
        """
        self.intent_keywords = intent_keywords
        self.fuzzy_threshold = fuzzy_threshold
        self.default_intent = default_intent
        
        # Build reverse mapping for better matching
        self.keyword_to_file = {}
        for keyword, filename in intent_keywords.items():
            if keyword != default_intent:
                self.keyword_to_file[keyword.lower()] = filename
        
        logger.info(f"IntentMatcher initialized with {len(self.keyword_to_file)} keywords")
    
    def match_intent(self, text: str) -> str:
        """
        Match text to best intent
        
        Args:
            text: User speech text
            
        Returns:
            Audio filename to play
        """
        if not text or not text.strip():
            logger.debug("Empty text, returning default")
            return self.intent_keywords[self.default_intent]
        
        text_lower = text.lower().strip()
        start_time = time.time()
        
        # Exact match check first
        for keyword, filename in self.keyword_to_file.items():
            if keyword in text_lower:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(f"🎯 Exact match: '{keyword}' → {filename} ({elapsed_ms:.1f}ms)")
                return filename
        
        # Fuzzy matching
        matches = {}
        for keyword, filename in self.keyword_to_file.items():
            # Partial ratio is more forgiving for substring matches
            score = fuzz.partial_ratio(keyword, text_lower)
            
            if score >= self.fuzzy_threshold:
                matches[filename] = score
                logger.debug(f"Fuzzy match: '{keyword}' → {filename} (score: {score})")
        
        # Return highest scoring match
        if matches:
            best_match = max(matches.items(), key=lambda x: x[1])
            filename, score = best_match
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"🎯 Fuzzy match: {filename} (score: {score}, {elapsed_ms:.1f}ms)")
            return filename
        
        # No match found
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"🎯 No match, using default ({elapsed_ms:.1f}ms)")
        return self.intent_keywords[self.default_intent]
    
    def add_intent(self, keyword: str, filename: str):
        """
        Add new intent mapping
        
        Args:
            keyword: Keyword to match
            filename: Audio file to play
        """
        self.intent_keywords[keyword] = filename
        self.keyword_to_file[keyword.lower()] = filename
        logger.info(f"Added intent: '{keyword}' → {filename}")
    
    def remove_intent(self, keyword: str):
        """
        Remove intent mapping
        
        Args:
            keyword: Keyword to remove
        """
        if keyword in self.intent_keywords:
            del self.intent_keywords[keyword]
            if keyword.lower() in self.keyword_to_file:
                del self.keyword_to_file[keyword.lower()]
            logger.info(f"Removed intent: '{keyword}'")
    
    def get_stats(self) -> dict:
        """Get matcher statistics"""
        return {
            'total_intents': len(self.keyword_to_file),
            'fuzzy_threshold': self.fuzzy_threshold,
            'default_intent': self.default_intent
        }


# =============================================================================
# ADVANCED INTENT MATCHING (For future enhancement)
# =============================================================================

class SemanticIntentMatcher:
    """
    Advanced intent matching using semantic similarity
    (Requires sentence transformers or similar)
    
    This is a placeholder for future enhancement.
    Can use models like sentence-transformers for better matching.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize semantic matcher
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self.model = None
        logger.info("SemanticIntentMatcher (not yet implemented)")
    
    def load_model(self):
        """Load sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded semantic model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed")
    
    def match_intent(self, text: str, intents: list) -> str:
        """
        Match text to intent using semantic similarity
        
        Args:
            text: User input
            intents: List of intent descriptions
            
        Returns:
            Best matching intent
        """
        # TODO: Implement semantic matching
        raise NotImplementedError("Semantic matching not yet implemented")