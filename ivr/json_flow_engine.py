import json
import os
import logging
import threading
from typing import Dict, Any, Optional, Tuple
from fuzzywuzzy import process
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-safe semantic model singleton
# ---------------------------------------------------------------------------
_semantic_model = None
_semantic_model_lock = threading.Lock()
_semantic_model_failed = False  # set True permanently after first load failure


def get_semantic_model():
    """
    Thread-safe, lazy-load of SentenceTransformer model.
    Returns the model, or None if disabled / failed to load.
    """
    global _semantic_model, _semantic_model_failed

    # Fast path — no lock needed once loaded or permanently failed
    if _semantic_model is not None or _semantic_model_failed:
        return _semantic_model

    with _semantic_model_lock:
        # Double-checked locking: check again inside the lock
        if _semantic_model is not None or _semantic_model_failed:
            return _semantic_model

        if not config.USE_SEMANTIC_MATCHING:
            logger.info("Semantic matching disabled via config (USE_SEMANTIC_MATCHING=False)")
            _semantic_model_failed = True
            return None

        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading semantic model (all-MiniLM-L6-v2) — this may take a few seconds...")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            # Warm up the model with a dummy encode to catch meta-tensor errors here
            model.encode("test", convert_to_tensor=True)
            _semantic_model = model
            logger.info("✓ Semantic model loaded and warmed up successfully.")
        except Exception as e:
            logger.error(f"Failed to load sentence_transformers: {e}")
            logger.warning("Semantic matching permanently disabled for this session. Fuzzy-only mode.")
            _semantic_model_failed = True

    return _semantic_model


# ---------------------------------------------------------------------------
# FlowEngine
# ---------------------------------------------------------------------------

class FlowEngine:
    """
    JSON-based IVR flow engine.

    Matching strategy (in order):
      1. Exact substring check  — near-zero cost
      2. Synonym expansion + fuzzy (fuzzywuzzy)  — ~1ms
      3. Semantic cosine similarity (sentence-transformers)  — ~50ms (optional)

    Choice key embeddings are pre-computed when the flow loads so per-query
    semantic inference only needs to encode the user utterance once.
    """

    def __init__(self,
                 flow_dir: str = config.IVR_FLOW_DIR,
                 default_lang: str = config.IVR_DEFAULT_LANG):
        self.flow_dir = flow_dir
        self.default_lang = default_lang
        self.flows: Dict[str, Any] = {}
        # Pre-computed embeddings: {lang: {step_key: {choice_key: tensor}}}
        self._embeddings: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.load_all_flows()

    # ------------------------------------------------------------------
    # Flow loading
    # ------------------------------------------------------------------

    def load_all_flows(self):
        if not os.path.exists(self.flow_dir):
            os.makedirs(self.flow_dir, exist_ok=True)
            logger.warning(f"Flow directory '{self.flow_dir}' created — no flows loaded.")
            return

        for filename in os.listdir(self.flow_dir):
            if not filename.endswith(".json"):
                continue
            lang = filename.replace(".json", "")
            path = os.path.join(self.flow_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                if not raw:
                    logger.warning(f"Skipping empty flow file: {filename}")
                    continue
                self.flows[lang] = json.loads(raw)
                logger.info(f"Loaded flow for language: {lang}")
            except Exception as e:
                logger.error(f"Error loading flow '{filename}': {e}")

        if config.USE_SEMANTIC_MATCHING:
            self._precompute_embeddings()

    def _precompute_embeddings(self):
        """Pre-encode all choice keys at startup so per-query cost is minimal."""
        model = get_semantic_model()
        if model is None:
            return
        try:
            for lang, flow in self.flows.items():
                self._embeddings[lang] = {}
                for step_key, step in flow.get("steps", {}).items():
                    choices = step.get("choices", {})
                    if not choices:
                        continue
                    self._embeddings[lang][step_key] = {
                        k: model.encode(k, convert_to_tensor=True)
                        for k in choices.keys()
                    }
            logger.info("✓ Choice embeddings pre-computed for all flows.")
        except Exception as e:
            logger.error(f"Failed to pre-compute embeddings: {e}")

    def get_flow(self, lang: str) -> Optional[Dict[str, Any]]:
        return self.flows.get(lang)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    # Common synonym expansions for quick yes/no capture
    _SYNONYMS: Dict[str, list] = {
        "yes": ["yes", "yeah", "sure", "correct", "ok", "okay", "yup", "yep",
                "affirmative", "of course", "absolutely"],
        "no":  ["no", "nope", "nah", "incorrect", "negative", "not really"],
    }

    def match_choice(self,
                     step_data: Dict[str, Any],
                     user_text: str,
                     lang: str = "en",
                     step_key: str = "") -> Optional[str]:
        """
        Return the value (next step) for the best-matching choice key,
        or None if confidence is below all thresholds.
        """
        if not user_text:
            return None

        choices = step_data.get("choices", {})
        if not choices:
            return None

        user_lower = user_text.lower().strip()
        choice_keys = list(choices.keys())

        # ---- Pass 0: exact substring match (fastest, zero cost) -----------
        for key in choice_keys:
            if key.lower() in user_lower:
                logger.info(f"Exact match '{user_lower}' -> '{key}'")
                return choices[key]

        # ---- Pass 1a: synonym expansion for yes/no -------------------------
        for key in choice_keys:
            synonyms = self._SYNONYMS.get(key.lower())
            if synonyms:
                best, score = process.extractOne(user_lower, synonyms)
                if score >= config.FUZZY_MATCH_THRESHOLD:
                    logger.info(f"Synonym matched '{user_lower}' -> '{key}' (score: {score})")
                    return choices[key]

        # ---- Pass 1b: fuzzy match on raw choice keys -----------------------
        best_key, fuzzy_score = process.extractOne(user_lower, choice_keys)
        if fuzzy_score >= config.FUZZY_MATCH_THRESHOLD:
            logger.info(f"Fuzzy matched '{user_lower}' -> '{best_key}' (score: {fuzzy_score})")
            return choices[best_key]

        # ---- Pass 2: semantic fallback (optional, pre-computed embeddings) --
        if config.USE_SEMANTIC_MATCHING:
            model = get_semantic_model()
            if model:
                try:
                    from sentence_transformers import util
                    user_emb = model.encode(user_lower, convert_to_tensor=True)

                    # Use pre-computed embeddings if available
                    step_embs = self._embeddings.get(lang, {}).get(step_key, {})

                    best_match = None
                    best_score = -1.0
                    for key in choice_keys:
                        key_emb = step_embs.get(key) or model.encode(key, convert_to_tensor=True)
                        score = util.cos_sim(user_emb, key_emb).item()
                        if score > best_score:
                            best_score = score
                            best_match = key

                    if best_score >= config.SEMANTIC_MATCH_THRESHOLD:
                        logger.info(
                            f"Semantic matched '{user_lower}' -> '{best_match}' "
                            f"(score: {best_score:.2f})"
                        )
                        return choices[best_match]
                except Exception as e:
                    logger.warning(f"Semantic matching error (skipping): {e}")

        logger.info(f"No match found for '{user_lower}'.")
        return None

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process_input(self,
                      session_data: Dict[str, Any],
                      user_text: str) -> Tuple[str, Optional[str], bool, bool]:
        """
        Process user input and advance the flow state.

        Args:
            session_data: Mutable dict tracking 'lang', 'step', 'retry_count'.
                          Updated in-place so the caller can persist it.
            user_text: Transcribed text from the caller.

        Returns:
            (answer_text, audio_filename, should_end_call, is_fallback)
        """
        lang = session_data.get("lang", self.default_lang)
        flow = self.get_flow(lang)

        if not flow:
            return "Sorry, I am having technical difficulties.", None, True, True

        current_step_key = session_data.get("step", flow.get("start"))
        if not current_step_key:
            return "Sorry, the flow is misconfigured.", None, True, True

        step = flow.get("steps", {}).get(current_step_key, {})
        step_type = step.get("type", "normal")
        next_step_key = None

        if step_type == "choice":
            next_step_key = self.match_choice(step, user_text, lang, current_step_key)
            if not next_step_key:
                retry_count = session_data.get("retry_count", 0) + 1
                session_data["retry_count"] = retry_count
                if retry_count > config.IVR_MAX_RETRIES:
                    logger.info("Max retries exceeded — transferring to agent.")
                    next_step_key = "connect_agent"
                    session_data["retry_count"] = 0
                else:
                    logger.info(
                        f"No match for '{user_text}' at step '{current_step_key}' "
                        f"(retry {retry_count}/{config.IVR_MAX_RETRIES})"
                    )
                    return "Sorry, I didn't get that.", "sorry.wav", False, True
            else:
                session_data["retry_count"] = 0

        elif step_type == "input":
            # Accept any utterance as the input value, advance to next
            next_step_key = step.get("next")
            logger.info(f"Captured input at '{current_step_key}': {user_text}")

        elif step.get("action"):
            # Stubbed action — advance to next
            next_step_key = step.get("next", "thank_you")
            logger.info(f"Action stub: {step.get('action')}")

        elif step.get("next"):
            next_step_key = step.get("next")

        else:
            if not step.get("end"):
                logger.warning(f"Step '{current_step_key}' has no next/choices/end.")
            next_step_key = current_step_key  # stay put

        # Advance state
        if next_step_key and next_step_key in flow.get("steps", {}):
            session_data["step"] = next_step_key
            next_step = flow["steps"][next_step_key]
            is_end = next_step.get("end", False) or next_step.get("action") == "transfer_agent"
            return next_step.get("prompt", ""), next_step.get("audio"), is_end, False

        # Stay on current step (end step or missing next)
        is_end = step.get("end", False) or step.get("action") == "transfer_agent"
        return step.get("prompt", ""), step.get("audio"), is_end, False

    def get_initial_step(self, lang: str = None) -> Tuple[str, Optional[str]]:
        """Return the prompt text and audio filename for the very first step."""
        lang = lang or self.default_lang
        flow = self.get_flow(lang)
        if not flow:
            return "Sorry, no flow found.", None
        start_step = flow.get("start")
        step_data = flow.get("steps", {}).get(start_step, {})
        return step_data.get("prompt", ""), step_data.get("audio")
