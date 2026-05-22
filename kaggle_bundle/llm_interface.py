"""
Ollama LLM Interface for LSTM + LLM Ablation Study
Handles communication with Ollama (Gemma3) running on localhost
"""

import requests
import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional
import json


class OllamaInterface:
    """Interface to communicate with Ollama LLM (Gemma3)"""
    
    def __init__(
        self,
        model_name: str = "gemma3:4b",
        host: str = "localhost",
        port: int = 11434,
        timeout: int = 10,
    ):
        self.model_name = model_name
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.embedding_dim = 256  # LLM latent vector dimension
        
        # Check connection
        if not self.check_connection():
            print(f"⚠️  Warning: Could not connect to Ollama at {self.base_url}")
            print("   Make sure Ollama is running: 'ollama serve' or 'ollama run gemma3'")
            self.is_connected = False
        else:
            print(f"✓ Connected to Ollama ({self.model_name}) at {self.base_url}")
            self.is_connected = True
    
    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def generate_strategy_embedding(
        self,
        state: np.ndarray,
        task_description: str = "HalfCheetah locomotion",
        history: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Generate LLM strategy embedding from current state.
        
        Args:
            state: Current observation (qpos, shape: (8,))
            task_description: Description of the task
            history: Optional history of previous prompts
        
        Returns:
            Strategy embedding vector (256-dim)
        """
        
        if not self.is_connected:
            # Fallback: return random embedding if LLM not available
            print("⚠️  LLM not available, returning random embedding")
            return torch.randn(self.embedding_dim)
        
        # Format state info for LLM prompt
        state_info = self._format_state_for_prompt(state)
        
        # Build prompt
        prompt = self._build_prompt(state_info, task_description, history)
        
        # Get LLM response
        strategy_text = self._query_ollama(prompt)
        
        # Convert text response to embedding
        embedding = self._text_to_embedding(strategy_text)
        
        return embedding
    
    def _format_state_for_prompt(self, state: np.ndarray) -> str:
        """Format state vector into readable text for LLM prompt"""
        state_labels = [
            "x_position", "y_position", "z_position",
            "x_velocity", "y_velocity", "z_velocity",
            "torso_rotation", "torso_angular_vel"
        ]
        
        state_info = "Current robot state:\n"
        for i, (label, value) in enumerate(zip(state_labels[:len(state)], state)):
            state_info += f"  - {label}: {value:.3f}\n"
        
        return state_info
    
    def _build_prompt(
        self,
        state_info: str,
        task_description: str,
        history: Optional[List[str]] = None,
    ) -> str:
        """Build prompt for LLM"""
        prompt = f"""Task: {task_description}

{state_info}

Based on the current state, provide a high-level action strategy.
The strategy should describe what action to take next (e.g., "accelerate forward", 
"stabilize balance", "increase speed", "turn left", etc.).

Strategy: """
        
        return prompt
    
    def _query_ollama(self, prompt: str, max_tokens: int = 50) -> str:
        """Query Ollama for strategy text"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"Ollama error: {response.status_code}")
                return ""
        
        except requests.exceptions.Timeout:
            print("⚠️  Ollama timeout - returning empty strategy")
            return ""
        except Exception as e:
            print(f"⚠️  Ollama error: {e}")
            return ""
    
    def _text_to_embedding(self, text: str) -> torch.Tensor:
        """
        Convert LLM strategy text to embedding vector.
        Uses simple hashing + random projection approach.
        """
        if not text:
            # Return zero vector if no response
            return torch.zeros(self.embedding_dim)
        
        # Use word embedding as simple strategy representation
        words = text.lower().split()
        
        # Create embedding from word hash + learned weights
        embedding = np.zeros(self.embedding_dim)
        
        # Strategy keywords and their associated bias directions
        strategy_keywords = {
            "accelerate": np.ones(self.embedding_dim) * 0.5,
            "forward": np.ones(self.embedding_dim) * 0.4,
            "speed": np.ones(self.embedding_dim) * 0.3,
            "left": np.ones(self.embedding_dim) * -0.3,
            "right": np.ones(self.embedding_dim) * 0.3,
            "stabilize": np.ones(self.embedding_dim) * 0.2,
            "balance": np.ones(self.embedding_dim) * 0.2,
            "slow": np.ones(self.embedding_dim) * -0.3,
            "stop": np.ones(self.embedding_dim) * -0.5,
        }
        
        # Add keyword contributions
        for keyword, bias in strategy_keywords.items():
            if keyword in words:
                embedding += bias
        
        # Add randomness for diversity
        np.random.seed(hash(text) % (2**32))  # Deterministic based on text
        embedding += np.random.normal(0, 0.1, self.embedding_dim)
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return torch.from_numpy(embedding).float()
    
    def batch_generate_embeddings(
        self,
        states: np.ndarray,
        task_description: str = "HalfCheetah locomotion",
    ) -> torch.Tensor:
        """
        Generate embeddings for a batch of states.
        
        Args:
            states: Batch of observations (batch_size, seq_len, obs_dim)
            task_description: Task description
        
        Returns:
            Embeddings tensor (batch_size, seq_len, 256)
        """
        embeddings = []
        
        for state_seq in states:
            seq_embeddings = []
            for state in state_seq:
                emb = self.generate_strategy_embedding(state, task_description)
                seq_embeddings.append(emb)
            
            embeddings.append(torch.stack(seq_embeddings))
        
        return torch.stack(embeddings)


class MockOllamaInterface:
    """Mock Ollama interface for testing (when Ollama is not available)"""
    
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.is_connected = False
        print("Using MockOllamaInterface (Ollama not available)")
    
    def generate_strategy_embedding(
        self,
        state: np.ndarray,
        task_description: str = "HalfCheetah locomotion",
        history: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Return random embedding (mock)"""
        # Use state as seed for reproducibility in mock
        seed = int(np.sum(np.abs(state)) * 1000) % (2**31)
        np.random.seed(seed)
        return torch.from_numpy(np.random.randn(self.embedding_dim)).float()
    
    def batch_generate_embeddings(
        self,
        states: np.ndarray,
        task_description: str = "HalfCheetah locomotion",
    ) -> torch.Tensor:
        """Generate embeddings for batch (mock)"""
        batch_size, seq_len = states.shape[0], states.shape[1]
        return torch.randn(batch_size, seq_len, self.embedding_dim)


def get_llm_interface(
    use_ollama: bool = True,
    model_name: str = "gemma3",
    host: str = "localhost",
    port: int = 11434,
) -> OllamaInterface:
    """
    Factory function to get appropriate LLM interface.
    
    Returns OllamaInterface if Ollama is available, else MockOllamaInterface.
    """
    
    if not use_ollama:
        return MockOllamaInterface(embedding_dim=256)
    
    try:
        interface = OllamaInterface(
            model_name=model_name,
            host=host,
            port=port,
            timeout=10
        )
        
        if interface.is_connected:
            return interface
        else:
            print("Ollama not available, falling back to mock interface")
            return MockOllamaInterface(embedding_dim=256)
    
    except Exception as e:
        print(f"Failed to create Ollama interface: {e}")
        return MockOllamaInterface(embedding_dim=256)
