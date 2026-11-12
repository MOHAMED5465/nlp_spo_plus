import torch
import torch.nn as nn
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType

class DeBERTaCVRP1(nn.Module):
    def __init__(self, num_nodes: int, model_name:str = "microsoft/deberta-v3-base", lora_rank: int=8, lora_alpha :int = 32, lora_dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # base model
        base_model = AutoModel.from_pretrained(model_name)

        # LoRA Config
        lora_config = LoraConfig(
            task_type = TaskType.FEATURE_EXTRACTION,
            r=self.lora_rank,
            lora_alpha = self.lora_alpha,
            lora_dropout = self.lora_dropout,
            target_modules=['query_proj', "value_proj"]
        )

        # encoder
        self.encoder = get_peft_model(
            model=base_model,
            peft_config=lora_config
        )

        # edge cost prediction head
        hidden_size = base_model.config.hidden_size
        self.edge_head = nn.Sequential(
            nn.Linear(2*hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Softplus()
        )

    def forward(self, input_ids, attention_mask):
        # capture dimensions for batch, nodes, and sequence length
        B, N, L = input_ids.shape
        
        # flatten tensors to two dimensions for hugging face compatibility
        input_ids_flat = input_ids.view(B * N, L)
        attention_mask_flat = attention_mask.view(B * N, L)
        
        # forward pass in the encoder with flattened tensors
        outputs = self.encoder(input_ids=input_ids_flat, attention_mask=attention_mask_flat)

        last_hidden_state = outputs.last_hidden_state 
        
        # extract the cls token at position zero
        cls_tokens = last_hidden_state[:, 0, :]  # batch times nodes, hidden siz

        h = cls_tokens.view(B, N, -1) # B, N, hidden_size

        # computing edges costs
        h_src = h.unsqueeze(2).expand(B, N, N, -1) # B, N, N, hidden_dim
    
        h_tgt = h.unsqueeze(1).expand(B, N, N, -1) # B, N, N, hidden_dim
        
        edge_inputs = torch.cat([h_src, h_tgt], dim=-1) # B, N, N, 2*hidden_dim

        edge_inputs = edge_inputs.float()

        self.edge_head = self.edge_head.float()

        c_matrix = self.edge_head(edge_inputs).squeeze(-1) # B, N, N
        
        mask = ~torch.eye(N, dtype=torch.bool, device=c_matrix.device)
        
        c_pred = c_matrix[:, mask] # B, N*(N-1)
        
        return c_pred
