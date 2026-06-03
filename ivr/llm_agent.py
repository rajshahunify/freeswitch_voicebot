"""
LLM Agent — Multi-Provider with Streaming
Replaces the old FlowEngine with conversational AI.

Supports:
  - GeminiProvider: Google Gemini API (free tier, fast, high quality)
  - OllamaProvider: Local Ollama server (Qwen, offline, free forever)

Features:
  - Streaming responses (yields sentences as they form)
  - Conversation history management
  - Configurable system prompt
  - Timeout handling with fallback messages
  - Provider switching via env var
"""

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Generator, List, Dict, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# DEFAULT SYSTEM PROMPT
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """You are a friendly and professional AI customer service agent for "Unified Reach Fiber" — a fiber internet service provider.

Your role:
- Help customers with their internet service inquiries
- Be conversational, warm, and helpful
- Keep responses concise (1-3 sentences) since this is a voice call — long responses feel unnatural
- If you don't know something, say so honestly and offer to connect them with a human agent

You can help with:
- Account inquiries and billing
- Internet speed issues and troubleshooting
- Plan upgrades and new connections
- Service outages and maintenance schedules
- General questions about the company

Important voice-call guidelines:
- Never use markdown, bullet points, or formatting — this is spoken audio
- Don't say "click here" or reference any visual elements
- Use natural conversational language, as if talking on the phone
- Avoid overly long responses — keep it to 1-3 sentences per turn
- If the customer seems frustrated, acknowledge their feelings first
"""


# =============================================================================
# ABSTRACT BASE — all LLM providers implement this
# =============================================================================

class BaseLLMProvider(ABC):
    """Base class for all LLM providers"""
    
    def __init__(self, name: str, system_prompt: str = ""):
        self.name = name
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.total_requests = 0
        self.total_errors = 0
        self.total_response_time = 0.0
    
    @abstractmethod
    def get_response(self, user_text: str, history: List[Dict]) -> str:
        """
        Get a complete response from the LLM (blocking).
        
        Args:
            user_text: The user's message
            history: Conversation history as list of {"role": "user"/"assistant", "text": "..."}
            
        Returns:
            The LLM's response text
        """
        pass
    
    @abstractmethod
    def get_response_streaming(self, user_text: str, history: List[Dict]) -> Generator[str, None, None]:
        """
        Stream response sentences from the LLM.
        
        Yields complete sentences as they form from the token stream.
        Each yielded string is a complete sentence ready for TTS.
        
        Args:
            user_text: The user's message
            history: Conversation history
            
        Yields:
            Complete sentences as strings
        """
        pass
    
    def get_stats(self) -> dict:
        avg_time = (
            self.total_response_time / self.total_requests
            if self.total_requests > 0 else 0
        )
        return {
            'provider': self.name,
            'total_requests': self.total_requests,
            'total_errors': self.total_errors,
            'average_time': f"{avg_time*1000:.0f}ms",
        }


def _extract_sentences(buffer: str):
    """
    Extract complete sentences from a buffer of streamed tokens.
    
    Returns (sentences, remaining_buffer):
    - sentences: list of complete sentences found
    - remaining_buffer: text that doesn't yet form a complete sentence
    """
    sentences = []
    
    # Split on sentence-ending punctuation followed by space or end
    # This regex finds sentence boundaries
    pattern = r'([^.!?]*[.!?])(?:\s|$)'
    
    while True:
        match = re.match(pattern, buffer)
        if match:
            sentence = match.group(1).strip()
            if sentence and len(sentence) > 2:  # Skip tiny fragments like "."
                sentences.append(sentence)
            buffer = buffer[match.end():].lstrip()
        else:
            break
    
    return sentences, buffer


# =============================================================================
# GEMINI PROVIDER
# =============================================================================

class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini API provider.
    Uses the NEW google-genai SDK (the old google-generativeai is deprecated).
    Free tier: 15 RPM, 1M tokens/day.
    Includes automatic retry with backoff for rate limits.
    """
    
    MAX_RETRIES = 1
    RETRY_BASE_DELAY = 2  # seconds — doubles each retry: 2s, 4s, 8s
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", system_prompt: str = ""):
        super().__init__("gemini", system_prompt)
        self.api_key = api_key
        self.model_name = model
        self._client = None
        self._initialized = False
        self.daily_quota_exhausted = False  # Set when daily limit hits 0
        logger.info(f"✓ GeminiProvider initialized (model: {model})")
    
    def _init_client(self):
        """Lazy-initialize the Gemini client"""
        if self._initialized:
            return
        
        try:
            from google import genai
            
            self._client = genai.Client(api_key=self.api_key)
            self._initialized = True
            logger.info(f"✓ Gemini client connected (model: {self.model_name})")
            
        except ImportError:
            logger.error("❌ google-genai not installed. Run: pip install google-genai")
            raise
    
    def _is_retryable_error(self, error) -> bool:
        """Check if an error is a rate limit / quota error that we should retry.
        
        Important: If the daily quota limit is 0 (fully exhausted), retrying
        is pointless — we'd just waste 14+ seconds of backoff time.
        """
        error_str = str(error).lower()
        
        # Permanent quota exhaustion: daily limit is 0 — retrying won't help
        if 'limit: 0' in error_str or 'quota exceeded' in error_str:
            if 'per_day' in error_str or 'perday' in error_str:
                logger.warning("⛔ Gemini DAILY quota exhausted (limit: 0). Skipping retries.")
                self.daily_quota_exhausted = True
                return False
        
        return any(keyword in error_str for keyword in [
            "resourceexhausted", "429", "rate limit", "quota",
            "too many requests", "retry", "overloaded",
        ])
    
    def _build_contents(self, user_text: str, history: List[Dict]) -> list:
        """Convert our history format to Gemini's new SDK format"""
        from google.genai import types
        
        contents = []
        for entry in history:
            role = "user" if entry["role"] == "user" else "model"
            contents.append(
                types.Content(role=role, parts=[types.Part(text=entry["text"])])
            )
        
        # Add current user message
        contents.append(
            types.Content(role="user", parts=[types.Part(text=user_text)])
        )
        return contents
    
    def get_response(self, user_text: str, history: List[Dict]) -> str:
        self._init_client()
        self.total_requests += 1
        start_time = time.time()
        
        from google.genai import types
        
        contents = self._build_contents(user_text, history)
        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # Disable thinking for speed
        )
        last_error = None
        
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                if attempt > 0:
                    delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(f"⏳ Gemini retry {attempt}/{self.MAX_RETRIES} after {delay}s...")
                    time.sleep(delay)
                
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                
                response_time = time.time() - start_time
                self.total_response_time += response_time
                
                text = response.text.strip()
                logger.info(
                    f"🤖 LLM [gemini]: '{text[:80]}{'...' if len(text) > 80 else ''}' "
                    f"({response_time*1000:.0f}ms)"
                )
                return text
                
            except Exception as e:
                last_error = e
                if self._is_retryable_error(e) and attempt < self.MAX_RETRIES:
                    logger.warning(f"⚠️ Gemini rate limit (attempt {attempt+1}): {type(e).__name__}: {e}")
                    continue
                else:
                    break
        
        self.total_errors += 1
        logger.error(f"❌ Gemini error after {self.MAX_RETRIES+1} attempts: {type(last_error).__name__}: {last_error}")
        return "I'm sorry, I'm having a moment. Could you say that again?"
    
    def get_response_streaming(self, user_text: str, history: List[Dict]) -> Generator[str, None, None]:
        """Stream sentences from Gemini with retry for rate limits"""
        self._init_client()
        self.total_requests += 1
        start_time = time.time()
        
        from google.genai import types
        
        contents = self._build_contents(user_text, history)
        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # Disable thinking for speed
        )
        last_error = None
        
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                if attempt > 0:
                    delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(f"⏳ Gemini streaming retry {attempt}/{self.MAX_RETRIES} after {delay}s...")
                    time.sleep(delay)
                
                response = self._client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                
                buffer = ""
                first_sentence = True
                got_content = False
                
                for chunk in response:
                    if chunk.text:
                        got_content = True
                        buffer += chunk.text
                        
                        # Extract complete sentences
                        sentences, buffer = _extract_sentences(buffer)
                        for sentence in sentences:
                            if first_sentence:
                                first_time = time.time() - start_time
                                logger.info(f"🤖 LLM [gemini] first sentence in {first_time*1000:.0f}ms")
                                first_sentence = False
                            yield sentence
                
                # Yield any remaining text as final sentence
                if buffer.strip():
                    yield buffer.strip()
                    got_content = True
                
                if got_content:
                    total_time = time.time() - start_time
                    self.total_response_time += total_time
                    return  # Success — exit the retry loop
                    
            except Exception as e:
                last_error = e
                if self._is_retryable_error(e) and attempt < self.MAX_RETRIES:
                    logger.warning(f"⚠️ Gemini streaming rate limit (attempt {attempt+1}): {type(e).__name__}: {e}")
                    continue
                else:
                    break
        
        self.total_errors += 1
        logger.error(f"❌ Gemini streaming error after {self.MAX_RETRIES+1} attempts: {type(last_error).__name__}: {last_error}")
        yield "I'm sorry, I'm having a moment. Could you say that again?"


# =============================================================================
# OLLAMA PROVIDER
# =============================================================================

class OllamaProvider(BaseLLMProvider):
    """
    Ollama API provider for local LLMs.
    Supports any Ollama-hosted model (Qwen, Llama, Mistral, etc.)
    """
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen2.5:0.5b",
        system_prompt: str = "",
    ):
        super().__init__("ollama", system_prompt)
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        logger.info(f"✓ OllamaProvider initialized (model: {model}, url: {ollama_url})")
    
    def _get_effective_url(self) -> str:
        """
        Dynamically determine the reachable Ollama URL.
        If configured to http://ollama:11434 but DNS fails or container is not found,
        it automatically falls back to http://host.docker.internal:11434 (Windows/Mac host).
        """
        # Cache the resolved URL so we don't do TCP/DNS checks on every single request
        if hasattr(self, '_resolved_url') and self._resolved_url:
            return self._resolved_url
            
        import requests
        
        # Test original URL
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=1.0)
            if r.status_code == 200:
                self._resolved_url = self.ollama_url
                return self._resolved_url
        except Exception:
            pass
            
        # If original failed and had 'ollama' in it, try host.docker.internal
        if "ollama" in self.ollama_url.lower():
            fallback_url = self.ollama_url.replace("ollama", "host.docker.internal")
            logger.warning(f"⚠️ Ollama container unreachable at {self.ollama_url}. Trying developer host fallback: {fallback_url}...")
            try:
                r = requests.get(f"{fallback_url}/api/tags", timeout=1.0)
                if r.status_code == 200:
                    logger.info(f"✅ Success! Connected to local Ollama running on Windows host: {fallback_url}")
                    self._resolved_url = fallback_url
                    return self._resolved_url
            except Exception:
                pass
                
        # Fall back to original config
        self._resolved_url = self.ollama_url
        return self._resolved_url

    def _build_messages(self, user_text: str, history: List[Dict]) -> list:
        """Convert to Ollama chat format"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        for entry in history:
            messages.append({
                "role": entry["role"],
                "content": entry["text"],
            })
        
        messages.append({"role": "user", "content": user_text})
        return messages
    
    def get_response(self, user_text: str, history: List[Dict]) -> str:
        import requests
        
        self.total_requests += 1
        start_time = time.time()
        
        try:
            messages = self._build_messages(user_text, history)
            effective_url = self._get_effective_url()
            
            response = requests.post(
                f"{effective_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                },
                timeout=30,
            )
            
            response.raise_for_status()
            result = response.json()
            text = result.get("message", {}).get("content", "").strip()
            
            response_time = time.time() - start_time
            self.total_response_time += response_time
            
            logger.info(
                f"🤖 LLM [ollama/{self.model}]: '{text[:80]}{'...' if len(text) > 80 else ''}' "
                f"({response_time*1000:.0f}ms)"
            )
            return text if text else "I'm sorry, could you repeat that?"
            
        except Exception as e:
            self.total_errors += 1
            logger.error(f"❌ Ollama error: {e}", exc_info=True)
            return "I'm sorry, I'm having trouble right now. Could you say that again?"
    
    def get_response_streaming(self, user_text: str, history: List[Dict]) -> Generator[str, None, None]:
        """Stream sentences from Ollama"""
        import requests
        
        self.total_requests += 1
        start_time = time.time()
        
        try:
            messages = self._build_messages(user_text, history)
            effective_url = self._get_effective_url()
            
            response = requests.post(
                f"{effective_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                },
                timeout=30,
                stream=True,
            )
            
            response.raise_for_status()
            
            buffer = ""
            first_sentence = True
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            buffer += token
                            
                            sentences, buffer = _extract_sentences(buffer)
                            for sentence in sentences:
                                if first_sentence:
                                    first_time = time.time() - start_time
                                    logger.info(f"🤖 LLM [ollama/{self.model}] first sentence in {first_time*1000:.0f}ms")
                                    first_sentence = False
                                yield sentence
                    except json.JSONDecodeError:
                        continue
            
            # Yield remaining buffer
            if buffer.strip():
                yield buffer.strip()
            
            total_time = time.time() - start_time
            self.total_response_time += total_time
            
        except Exception as e:
            self.total_errors += 1
            logger.error(f"❌ Ollama streaming error: {e}", exc_info=True)
            yield "I'm sorry, I'm having trouble right now. Could you say that again?"


# =============================================================================
# GROQ PROVIDER (Fast cloud LLM — free tier, ~200ms latency)
# =============================================================================

class GroqProvider(BaseLLMProvider):
    """
    Groq cloud LLM provider — runs Llama 3.3 70B on custom inference hardware.
    ~200-400ms latency vs Ollama's ~2000ms on CPU.
    Free tier: no credit card needed, ~30 RPM / 6000 tokens per minute.
    Uses OpenAI-compatible REST API.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        system_prompt: str = "",
        timeout: int = 10,
    ):
        super().__init__("groq", system_prompt)
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        
        if not api_key:
            logger.warning("⚠️ Groq API key not set — GroqProvider will fail on requests")
        
        logger.info(f"✓ GroqProvider initialized (model: {model})")
    
    def get_response(self, user_text: str, history: List[Dict]) -> str:
        self.total_requests += 1
        start_time = time.time()
        
        try:
            import requests
            
            messages = [{"role": "system", "content": self.system_prompt}]
            for h in history[-20:]:  # Keep last 20 turns
                role = "user" if h.get("role") == "user" else "assistant"
                messages.append({"role": role, "content": h.get("text", "")})
            messages.append({"role": "user", "content": user_text})
            
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 150,  # Keep responses short for voice
                    "temperature": 0.7,
                },
                timeout=self.timeout,
            )
            
            if response.status_code == 429:
                logger.warning("⛔ Groq rate limited")
                self.total_errors += 1
                return "I'm sorry, I'm having a moment. Could you say that again?"
            
            response.raise_for_status()
            data = response.json()
            
            result = data["choices"][0]["message"]["content"].strip()
            total_time = time.time() - start_time
            self.total_response_time += total_time
            logger.info(f"🤖 LLM [groq/{self.model}] response in {total_time*1000:.0f}ms")
            return result
            
        except Exception as e:
            self.total_errors += 1
            logger.error(f"❌ Groq error: {e}")
            return "I'm sorry, I'm having a moment. Could you say that again?"
    
    def get_response_streaming(self, user_text: str, history: List[Dict]) -> Generator[str, None, None]:
        self.total_requests += 1
        start_time = time.time()
        first_sentence = True
        
        try:
            import requests
            
            messages = [{"role": "system", "content": self.system_prompt}]
            for h in history[-20:]:
                role = "user" if h.get("role") == "user" else "assistant"
                messages.append({"role": role, "content": h.get("text", "")})
            messages.append({"role": "user", "content": user_text})
            
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 150,
                    "temperature": 0.7,
                    "stream": True,
                },
                timeout=self.timeout,
                stream=True,
            )
            
            if response.status_code == 429:
                logger.warning("⛔ Groq rate limited")
                self.total_errors += 1
                yield "I'm sorry, I'm having a moment. Could you say that again?"
                return
            
            response.raise_for_status()
            
            buffer = ""
            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        buffer += token
                        # Check for sentence boundary
                        if re.search(r'[.!?]\s*$', buffer) or re.search(r'[.!?]"?\s*$', buffer):
                            sentence = buffer.strip()
                            if sentence:
                                if first_sentence:
                                    elapsed = time.time() - start_time
                                    logger.info(f"🤖 LLM [groq/{self.model}] first sentence in {elapsed*1000:.0f}ms")
                                    first_sentence = False
                                yield sentence
                                buffer = ""
                except json.JSONDecodeError:
                    continue
            
            # Yield remaining buffer
            if buffer.strip():
                if first_sentence:
                    elapsed = time.time() - start_time
                    logger.info(f"🤖 LLM [groq/{self.model}] first sentence in {elapsed*1000:.0f}ms")
                yield buffer.strip()
            
            total_time = time.time() - start_time
            self.total_response_time += total_time
            
        except Exception as e:
            self.total_errors += 1
            logger.error(f"❌ Groq streaming error: {e}", exc_info=True)
            yield "I'm sorry, I'm having trouble right now. Could you say that again?"


# =============================================================================
# FALLBACK LLM PROVIDER
# =============================================================================

class FallbackLLM(BaseLLMProvider):
    """
    Fallback LLM provider that wraps a primary LLM (e.g., Gemini)
    and falls back to a secondary local LLM (e.g., Ollama) on error.
    """
    
    def __init__(self, primary: BaseLLMProvider, fallback: BaseLLMProvider, threshold: int = 1):
        super().__init__(f"fallback({primary.name}→{fallback.name})", primary.system_prompt)
        self.primary = primary
        self.fallback = fallback
        self.threshold = threshold
        self.consecutive_failures = 0
        self.using_fallback = False
        self._retry_interval = 120  # seconds before trying primary again
        self._last_primary_attempt = 0.0
        
        logger.info(
            f"✓ FallbackLLM initialized: primary={primary.name}, "
            f"fallback={fallback.name}, threshold={threshold}"
        )
        
    def get_response(self, user_text: str, history: List[Dict]) -> str:
        self.total_requests += 1
        
        # NOTE: Primary retry is handled by the background health check in server_multicall.py
        # Do NOT retry primary here — it wastes call time on failed Gemini attempts
        
        if not self.using_fallback:
            try:
                response = self.primary.get_response(user_text, history)
                if response and "I'm sorry, I'm having a moment" in response:
                    raise RuntimeError("Primary LLM returned error fallback message")
                
                self.consecutive_failures = 0
                return response
            except Exception as e:
                self.consecutive_failures += 1
                logger.warning(f"⚠️ Primary LLM ({self.primary.name}) failed (consecutive: {self.consecutive_failures}/{self.threshold}): {e}")
                
                if self.consecutive_failures >= self.threshold:
                    logger.error(f"❌ Primary LLM ({self.primary.name}) failed {self.consecutive_failures} times. SWITCHING TO FALLBACK ({self.fallback.name}).")
                    self.using_fallback = True
                    self._last_primary_attempt = time.time()
                    
                    # If daily quota is permanently exhausted, don't retry for a long time
                    if hasattr(self.primary, 'daily_quota_exhausted') and self.primary.daily_quota_exhausted:
                        self._retry_interval = 3600  # 1 hour
                        logger.warning(f"⛔ Daily quota exhausted — will retry primary in {self._retry_interval}s")
                
                return self.fallback.get_response(user_text, history)
        else:
            logger.info(f"🤖 Using fallback LLM ({self.fallback.name})")
            return self.fallback.get_response(user_text, history)
            
    def get_response_streaming(self, user_text: str, history: List[Dict]) -> Generator[str, None, None]:
        self.total_requests += 1
        
        
        # NOTE: Primary retry is handled by the background health check in server_multicall.py
        # Do NOT retry primary here — it wastes call time on failed Gemini attempts
        
        if not self.using_fallback:
            try:
                got_content = False
                stream_generator = self.primary.get_response_streaming(user_text, history)
                for sentence in stream_generator:
                    if "I'm sorry, I'm having a moment" in sentence:
                        raise RuntimeError("Primary LLM streaming failed with fallback message")
                    got_content = True
                    yield sentence
                
                if not got_content:
                    raise RuntimeError("Primary LLM streaming yielded no content")
                
                self.consecutive_failures = 0
                return
            except Exception as e:
                self.consecutive_failures += 1
                logger.warning(f"⚠️ Primary LLM streaming ({self.primary.name}) failed (consecutive: {self.consecutive_failures}/{self.threshold}): {e}")
                
                if self.consecutive_failures >= self.threshold:
                    logger.error(f"❌ Primary LLM streaming ({self.primary.name}) failed {self.consecutive_failures} times. SWITCHING TO FALLBACK ({self.fallback.name}).")
                    self.using_fallback = True
                    self._last_primary_attempt = time.time()
                    
                    # If daily quota is permanently exhausted, don't retry for a long time
                    if hasattr(self.primary, 'daily_quota_exhausted') and self.primary.daily_quota_exhausted:
                        self._retry_interval = 3600  # 1 hour — no point retrying all day
                        logger.warning(f"⛔ Daily quota exhausted — will retry primary in {self._retry_interval}s")
                
                for sentence in self.fallback.get_response_streaming(user_text, history):
                    yield sentence
        else:
            logger.info(f"🤖 Using fallback LLM streaming ({self.fallback.name})")
            for sentence in self.fallback.get_response_streaming(user_text, history):
                yield sentence
                
    def get_stats(self) -> dict:
        return {
            'provider': self.name,
            'using_fallback': self.using_fallback,
            'consecutive_failures': self.consecutive_failures,
            'primary_stats': self.primary.get_stats(),
            'fallback_stats': self.fallback.get_stats(),
        }


# =============================================================================
# FACTORY — creates the right LLM provider based on config
# =============================================================================

def create_llm_agent(
    provider: str = "gemini",
    # Gemini config
    gemini_api_key: str = "",
    gemini_model: str = "gemini-2.0-flash",
    # Ollama config
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "qwen2.5:0.5b",
    # Groq config
    groq_api_key: str = "",
    groq_model: str = "llama-3.3-70b-versatile",
    # Shared config
    system_prompt: str = "",
    system_prompt_file: str = "",
    # Fallback config
    fallback_enabled: bool = True,
    fallback_threshold: int = 2,
) -> BaseLLMProvider:
    """
    Factory function to create the appropriate LLM provider.
    
    Args:
        provider: "gemini", "ollama", or "groq"
        gemini_api_key: API key for Gemini
        gemini_model: Gemini model name
        ollama_url: Ollama server URL
        ollama_model: Ollama model name
        groq_api_key: API key for Groq
        groq_model: Groq model name
        system_prompt: Custom system prompt (overrides default)
        system_prompt_file: Path to file containing system prompt (overrides system_prompt)
        fallback_enabled: Enable fallback when primary fails
        fallback_threshold: Threshold before switching to fallback
    
    Returns:
        A BaseLLMProvider instance
    """
    # Load system prompt from file if specified
    if system_prompt_file and os.path.exists(system_prompt_file):
        with open(system_prompt_file, "r") as f:
            system_prompt = f.read().strip()
        logger.info(f"📋 System prompt loaded from: {system_prompt_file}")
    
    if provider == "gemini":
        if not gemini_api_key:
            if fallback_enabled:
                logger.warning("⚠️ GEMINI_API_KEY is missing but fallback is enabled. Swapping default provider to Ollama!")
                return OllamaProvider(
                    ollama_url=ollama_url,
                    model=ollama_model,
                    system_prompt=system_prompt,
                )
            else:
                raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        
        if fallback_enabled:
            primary = GeminiProvider(
                api_key=gemini_api_key,
                model=gemini_model,
                system_prompt=system_prompt,
            )
            # Use Groq as fallback if API key is available (much faster than Ollama)
            if groq_api_key:
                fallback = GroqProvider(
                    api_key=groq_api_key,
                    model=groq_model,
                    system_prompt=system_prompt,
                )
                logger.info("📋 Fallback: Groq (fast cloud LLM)")
            else:
                fallback = OllamaProvider(
                    ollama_url=ollama_url,
                    model=ollama_model,
                    system_prompt=system_prompt,
                )
            return FallbackLLM(primary, fallback, threshold=fallback_threshold)
        else:
            logger.info(f"📋 LLM Provider: Gemini ({gemini_model})")
            return GeminiProvider(
                api_key=gemini_api_key,
                model=gemini_model,
                system_prompt=system_prompt,
            )
    
    elif provider == "groq":
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        logger.info(f"📋 LLM Provider: Groq ({groq_model})")
        return GroqProvider(
            api_key=groq_api_key,
            model=groq_model,
            system_prompt=system_prompt,
        )
    
    elif provider == "ollama":
        logger.info(f"📋 LLM Provider: Ollama ({ollama_model} @ {ollama_url})")
        return OllamaProvider(
            ollama_url=ollama_url,
            model=ollama_model,
            system_prompt=system_prompt,
        )
    
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Use 'gemini', 'groq', or 'ollama'.")
