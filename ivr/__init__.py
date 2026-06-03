"""
IVR (Interactive Voice Response) Module
- ResponseHandler: Audio playback via FreeSWITCH (actively used)
- LLMAgent: AI-powered conversation (NEW — replaces FlowEngine)
- FlowEngine, IntentMatcher: Legacy keyword matching (kept for reference)
"""

from .response_handler import ResponseHandler

# New AI-powered agent
from .llm_agent import create_llm_agent, BaseLLMProvider

# Legacy imports — optional, may fail if sentence-transformers not installed
try:
    from .intent_matcher import IntentMatcher, SemanticIntentMatcher
    from .json_flow_engine import FlowEngine
except ImportError:
    IntentMatcher = None
    SemanticIntentMatcher = None
    FlowEngine = None

__all__ = [
    'ResponseHandler',
    'create_llm_agent',
    'BaseLLMProvider',
    'IntentMatcher',
    'SemanticIntentMatcher',
    'FlowEngine',
]