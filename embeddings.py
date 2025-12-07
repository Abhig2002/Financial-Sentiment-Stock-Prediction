"""
Centralized embedding module for Financial Sentiment Stock Prediction.

Supports both MiniLM and FinBERT embeddings with a unified interface.
"""

from typing import List, Optional, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class TextEmbedder:
    """
    Unified interface for text embeddings.
    Supports: 'minilm' (sentence-transformers) and 'finbert' (finance-specific BERT).
    """
    
    def __init__(self, model_name: str = "finbert", device: Optional[str] = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Either 'finbert' or 'minilm'
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name.lower()
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"📦 Loading {self.model_name} embeddings on {self.device}...")
        
        if self.model_name == "finbert":
            self._init_finbert()
        elif self.model_name == "minilm":
            self._init_minilm()
        else:
            raise ValueError(
                f"Unknown model_name: {model_name}. "
                f"Choose 'finbert' or 'minilm'."
            )
        
        print(f"✅ {self.model_name} embeddings loaded successfully.")
    
    def _init_finbert(self):
        """Initialize FinBERT model for finance-specific embeddings."""
        model_id = "yiyanghkust/finbert-tone"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()  # Set to evaluation mode
        self.max_length = 512  # BERT max sequence length
    
    def _init_minilm(self):
        """Initialize MiniLM model using sentence-transformers."""
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = self.model.to(self.device)
    
    def encode(
        self,
        texts: Union[List[str], str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single string or list of strings
            batch_size: Batch size for processing
            show_progress: Show progress bar
            normalize_embeddings: Whether to L2-normalize embeddings
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model_name == "minilm":
            return self._encode_minilm(texts, batch_size, show_progress, normalize_embeddings)
        elif self.model_name == "finbert":
            return self._encode_finbert(texts, batch_size, show_progress, normalize_embeddings)
    
    def _encode_minilm(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool,
        normalize_embeddings: bool
    ) -> np.ndarray:
        """Encode using MiniLM (sentence-transformers)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        )
        return embeddings
    
    def _encode_finbert(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool,
        normalize_embeddings: bool
    ) -> np.ndarray:
        """Encode using FinBERT with mean pooling."""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Use last hidden state
                hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
                
                # Mean pooling with attention mask
                embeddings = self._mean_pooling(hidden_states, attention_mask)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        
        # Normalize if requested
        if normalize_embeddings:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    @staticmethod
    def _mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply mean pooling to hidden states, considering attention mask.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            attention_mask: (batch_size, seq_len)
            
        Returns:
            pooled: (batch_size, hidden_dim)
        """
        # Expand attention mask to match hidden states dimensions
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Sum embeddings
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        
        # Sum mask (count valid tokens)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        # Mean pooling
        return sum_embeddings / sum_mask


def get_embedder(model_name: str = "finbert", device: Optional[str] = None) -> TextEmbedder:
    """
    Factory function to get an embedder instance.
    
    Args:
        model_name: Either 'finbert' or 'minilm'
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
        
    Returns:
        TextEmbedder instance
    """
    return TextEmbedder(model_name=model_name, device=device)






