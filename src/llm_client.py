"""
Unified LLM Client using OpenAI-compatible API.

Supports:
- OpenAI (GPT-4o, GPT-4, etc.)
- Ollama (local, OpenAI-compatible endpoint)
- Any OpenAI-compatible API

Single client, multiple providers - just change base_url.
"""

import json
import time
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import requests

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Provider presets
PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,  # Ollama doesn't need API key
        "default_model": "llama3.1:8b",
    },
}

# Pricing per 1M tokens (OpenAI only)
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    raw_response: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
        }


class LLMClient:
    """
    Unified LLM client using OpenAI-compatible API.
    
    Usage:
        # OpenAI
        client = LLMClient(provider="openai", model="gpt-4o")
        
        # Ollama (local)
        client = LLMClient(provider="ollama", model="llama3.1:8b")
        
        # Custom endpoint
        client = LLMClient(base_url="http://my-server/v1", model="my-model")
        
        # Generate
        response = client.generate("Explain this anomaly...")
        print(response.content)
    """
    
    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 120,
    ):
        """
        Initialize LLM client.
        
        Args:
            provider: "openai", "ollama", or custom
            model: Model name (uses provider default if not specified)
            base_url: Override provider's base URL
            api_key: API key (or set via environment variable)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        """
        # Get provider config
        if provider in PROVIDERS:
            config = PROVIDERS[provider]
            self.base_url = base_url or config["base_url"]
            self.model = model or config["default_model"]
            env_key = config.get("api_key_env")
            self.api_key = api_key or (os.environ.get(env_key) if env_key else None)
        else:
            # Custom provider
            if not base_url:
                raise ValueError(f"base_url required for custom provider '{provider}'")
            self.base_url = base_url
            self.model = model or "default"
            self.api_key = api_key
        
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Setup session
        self._session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._session.headers.update(headers)
    
    def is_available(self) -> bool:
        """Check if LLM server is accessible."""
        try:
            response = self._session.get(
                f"{self.base_url}/models",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> list:
        """List available models."""
        try:
            response = self._session.get(
                f"{self.base_url}/models",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return [m.get("id", m.get("name", "")) for m in data.get("data", data.get("models", []))]
            return []
        except Exception:
            return []
    
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD (OpenAI only)."""
        for key, pricing in PRICING.items():
            if key in self.model:
                input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
                output_cost = (completion_tokens / 1_000_000) * pricing["output"]
                return input_cost + output_cost
        return 0.0
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Generate a response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_mode: Request JSON output format
            
        Returns:
            LLMResponse object
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages, temperature, max_tokens, json_mode)
    
    def chat(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Chat with message history.
        
        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_mode: Request JSON output format
            
        Returns:
            LLMResponse object
        """
        start_time = time.time()
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            data = response.json()
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract usage (OpenAI format)
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=usage.get("total_tokens", prompt_tokens + completion_tokens),
                latency_ms=latency_ms,
                cost_usd=self._calculate_cost(prompt_tokens, completion_tokens),
                raw_response=data,
            )
            
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("error", {}).get("message", error_msg)
                except:
                    pass
            raise RuntimeError(f"LLM API error: {error_msg}")
        except requests.exceptions.Timeout:
            raise RuntimeError(f"LLM API timeout after {self.timeout}s")
        except Exception as e:
            raise RuntimeError(f"LLM API error: {e}")
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[Dict, LLMResponse]:
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
            json_mode=True,
        )
        
        content = response.content.strip()
        
        # Clean up markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        
        # Find JSON boundaries
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            content = content[json_start:json_end]
        
        parsed = json.loads(content)
        return parsed, response


# Convenience aliases
def get_client(provider: str = "ollama", model: Optional[str] = None, **kwargs) -> LLMClient:
    """Get an LLM client for the specified provider."""
    return LLMClient(provider=provider, model=model, **kwargs)


# Quick test
if __name__ == "__main__":
    print("Testing LLM clients...\n")
    
    # Test Ollama
    print("=== Ollama ===")
    try:
        client = LLMClient(provider="ollama")
        if client.is_available():
            print(f"✓ Available, models: {client.list_models()[:3]}")
            response = client.generate("Say hello in one word.")
            print(f"  Response: {response.content}")
            print(f"  Latency: {response.latency_ms:.0f}ms")
        else:
            print("✗ Not available")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test OpenAI
    print("\n=== OpenAI ===")
    try:
        client = LLMClient(provider="openai")
        if client.is_available():
            print(f"✓ Available")
            response = client.generate("Say hello in one word.")
            print(f"  Response: {response.content}")
            print(f"  Latency: {response.latency_ms:.0f}ms, Cost: ${response.cost_usd:.6f}")
        else:
            print("✗ Not available (check API key)")
    except Exception as e:
        print(f"✗ Error: {e}")
