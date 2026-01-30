"""
LLM Client for local models.

Supports:
- Ollama (REST API)
- llama-cpp-python (direct GGUF loading)

Provides a simple interface to interact with locally-running LLM models.
"""

import json
import time
from typing import Dict, Optional, Any, Generator, Union
from dataclasses import dataclass, field
from pathlib import Path
import requests


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    raw_response: Dict = field(default_factory=dict)
    
    @property
    def total_cost_estimate(self) -> float:
        """Estimate cost (for local models, this is always 0)."""
        return 0.0
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms
        }


class LlamaCppClient:
    """
    Client for llama-cpp-python (direct GGUF model loading).
    
    This loads GGUF models directly without needing a server.
    Better for POC and experimentation.
    
    Usage:
        client = LlamaCppClient(model_path="/path/to/model.gguf")
        response = client.generate("Explain this log anomaly...")
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,  # -1 = all layers on GPU
        temperature: float = 0.1,
        max_tokens: int = 1024,
        verbose: bool = False
    ):
        """
        Initialize llama-cpp client.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            verbose: Print llama.cpp logs
        """
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        
        self._llm = None
        self._loaded = False
    
    def _load_model(self):
        """Lazy load the model."""
        if self._loaded:
            return
        
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "Please install llama-cpp-python: pip install llama-cpp-python"
            )
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"Loading model from {self.model_path}...")
        start = time.time()
        
        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose
        )
        
        print(f"Model loaded in {time.time() - start:.1f}s")
        self._loaded = True
    
    def is_available(self) -> bool:
        """Check if model file exists."""
        return self.model_path.exists()
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_mode: If True, add JSON grammar constraints
            
        Returns:
            LLMResponse object
        """
        self._load_model()
        
        start_time = time.time()
        
        # Build full prompt
        full_prompt = ""
        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}</s>\n"
        full_prompt += f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
        
        # Generation kwargs
        kwargs = {
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "stop": ["</s>", "<|user|>", "<|endoftext|>"],
            "echo": False
        }
        
        # For JSON mode, we can add a grammar constraint if needed
        if json_mode:
            kwargs["stop"].append("\n\n")
        
        result = self._llm(full_prompt, **kwargs)
        
        latency_ms = (time.time() - start_time) * 1000
        
        content = result["choices"][0]["text"].strip()
        usage = result.get("usage", {})
        
        return LLMResponse(
            content=content,
            model=str(self.model_path.name),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
            raw_response=result
        )
    
    def chat(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> LLMResponse:
        """
        Chat completion with message history.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_mode: If True, request JSON output
            
        Returns:
            LLMResponse object
        """
        self._load_model()
        
        start_time = time.time()
        
        result = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            stop=["</s>"]
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        content = result["choices"][0]["message"]["content"].strip()
        usage = result.get("usage", {})
        
        return LLMResponse(
            content=content,
            model=str(self.model_path.name),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
            raw_response=result
        )
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> tuple[Dict, LLMResponse]:
        """
        Generate a JSON response.
        
        Returns:
            Tuple of (parsed_json, LLMResponse)
        """
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True
        )
        
        content = response.content.strip()
        
        # Handle potential markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            end_idx = -1 if lines[-1].strip() in ("```", "```json") else len(lines)
            content = "\n".join(lines[1:end_idx])
        
        # Try to find JSON in the response
        # Sometimes models add text before/after JSON
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            content = content[json_start:json_end]
        
        parsed = json.loads(content)
        
        return parsed, response


class OllamaClient:
    """
    Client for Ollama local LLM server.
    
    Ollama must be running locally (default: http://localhost:11434)
    
    Usage:
        client = OllamaClient(model="llama3.2")
        response = client.generate("Explain this log anomaly...")
    """
    
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 120
    ):
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (e.g., "llama3.2", "llama3.1", "mistral", "codellama")
            base_url: Ollama server URL
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        self._session = requests.Session()
    
    def is_available(self) -> bool:
        """Check if Ollama server is running and model is available."""
        try:
            response = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                return self.model.split(":")[0] in model_names
            return False
        except Exception:
            return False
    
    def list_models(self) -> list:
        """List available models on the Ollama server."""
        try:
            response = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                return response.json().get("models", [])
            return []
        except Exception:
            return []
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_mode: If True, request JSON output format
            
        Returns:
            LLMResponse object
        """
        start_time = time.time()
        
        # Build request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        if json_mode:
            payload["format"] = "json"
        
        try:
            response = self._session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=data.get("response", ""),
                model=self.model,
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                latency_ms=latency_ms,
                raw_response=data
            )
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Ollama request timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}")
    
    def chat(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> LLMResponse:
        """
        Chat completion with message history.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_mode: If True, request JSON output format
            
        Returns:
            LLMResponse object
        """
        start_time = time.time()
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens
            }
        }
        
        if json_mode:
            payload["format"] = "json"
        
        try:
            response = self._session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=self.model,
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                latency_ms=latency_ms,
                raw_response=data
            )
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Ollama request timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> tuple[Dict, LLMResponse]:
        """
        Generate a JSON response from the LLM.
        
        Returns:
            Tuple of (parsed_json, LLMResponse)
            
        Raises:
            json.JSONDecodeError if response is not valid JSON
        """
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True
        )
        
        # Try to parse JSON
        content = response.content.strip()
        
        # Handle potential markdown code blocks
        if content.startswith("```"):
            # Extract content between code blocks
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        
        parsed = json.loads(content)
        
        return parsed, response


class LLMClient:
    """
    Unified LLM client interface.
    
    Supports:
    - Ollama (REST API)
    - llama-cpp-python (direct GGUF model loading)
    """
    
    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.2",
        **kwargs
    ):
        """
        Initialize LLM client.
        
        Args:
            provider: "ollama" or "llama-cpp"
            model: Model name (ollama) or path to GGUF file (llama-cpp)
            **kwargs: Provider-specific arguments
        """
        self.provider = provider
        self.model = model
        
        if provider == "ollama":
            self._client = OllamaClient(model=model, **kwargs)
        elif provider == "llama-cpp":
            self._client = LlamaCppClient(model_path=model, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'ollama' or 'llama-cpp'")
    
    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self._client.is_available()
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text response."""
        return self._client.generate(prompt, **kwargs)
    
    def generate_json(self, prompt: str, **kwargs) -> tuple[Dict, LLMResponse]:
        """Generate JSON response."""
        return self._client.generate_json(prompt, **kwargs)
    
    def chat(self, messages: list, **kwargs) -> LLMResponse:
        """Chat completion."""
        return self._client.chat(messages, **kwargs)


# Quick test
if __name__ == "__main__":
    print("Testing Ollama client...")
    
    client = OllamaClient(model="llama3.2")
    
    # Check availability
    if client.is_available():
        print(f"✓ Ollama is available with model {client.model}")
        
        # List models
        models = client.list_models()
        print(f"\nAvailable models:")
        for m in models:
            print(f"  - {m.get('name')}")
        
        # Test generation
        print("\nTesting generation...")
        response = client.generate(
            prompt="What is log anomaly detection? Answer in one sentence.",
            temperature=0.1
        )
        print(f"Response: {response.content}")
        print(f"Tokens: {response.total_tokens}, Latency: {response.latency_ms:.0f}ms")
        
        # Test JSON generation
        print("\nTesting JSON generation...")
        try:
            parsed, response = client.generate_json(
                prompt='Return a JSON object with keys "status" and "message". Status should be "ok".',
                system_prompt="You are a helpful assistant that always responds in valid JSON."
            )
            print(f"Parsed JSON: {parsed}")
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw response: {response.content}")
    else:
        print("✗ Ollama is not available")
        print("  Start Ollama with: ollama serve")
        print(f"  Pull model with: ollama pull {client.model}")
