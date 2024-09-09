import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
from typing import Optional, Union, Tuple

from .config import ViTConfig


class ViTPatchEmbedding(nn.Module):
    """ This is the patch embedding layer. 
    It takes an input image and converts it into a sequence of patches. 
    It turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the inital
    `hidden_states` of shape `(batch_size, seq_len, hidden_size)` to be consumed by the Transformer.
    """
    def __init__(self, config: ViTConfig):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        
        self.patch_embedding = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        
        # check input size
        if num_channels != self.num_channels:
            raise ValueError(f"Input channels {num_channels} does not match the expected number of channels {self.num_channels}.")
        
        if height != self.image_size[0] or width != self.image_size[1]:
            if interpolate_pos_encoding:
                raise ValueError(f"Input height {height} and width {width} must be the same "
                                 "as the image size {self.image_size} when interpolate_pos_encoding is True.")
            else:
                print(f"Input height {height} and width {width} are different "
                      "from the image size {self.image_size}. Adding position embedding by interpolation.")
        
        embeddings = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2) #[batch_size, num_patches, hidden_size]
        
        return embeddings
        
        
class ViTEmbedding(nn.Module):
    """ This is the embedding layer. 
    It takes the patch embedding and construct the CLS token, position and patch embeddings. 
    Optionally, also the mask token. """
    def __init__(self, config: ViTConfig, use_mask_token: bool = False):
        """ 
        Args:
            config (ViTConfig): The configuration of the model.
            use_mask_token (bool, optional): Whether to use the mask token. Defaults to False.
        """
        super().__init__()
        self.config = config
        self.use_mask_token = use_mask_token
        self.patch_embedding = ViTPatchEmbedding(config)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        
        self.position_embeddings = nn.Parameter(torch.randn(1, config.num_patches + 1, config.hidden_size))
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        
    def interpolate_pos_encoding(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """ Interpolate the position encoding. 
        
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.
        
        Args: 
            x (torch.Tensor): The input tensor. Shape: (batch_size, num_patches + 1, hidden_size).
            height (int): The height of the image.
            width (int): The width of the image.
            
        Returns: 
            torch.Tensor: The interpolated position encoding.
        """
        num_patches = x.shape[1] - 1 # we have num_patches + 1 positions (including cls token)
        N = self.position_embedding.shape[1] - 1
        if num_patches == N:
            return self.position_embedding
        class_pos_embed = self.position_embedding[:, 0]
        patch_pos_embed = self.position_embedding[:, 1:]
        
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        hidden_size = patch_pos_embed.shape[-1]
        w = width // self.patch_embedding.patch_size[1] + 0.1
        h = height // self.patch_embedding.patch_size[0] + 0.1
        
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), hidden_size).permute(0, 3, 1, 2),
            scale_factor=(w/math.sqrt(N), h/math.sqrt(N)),
            mode='bicubic',
            align_corners=False
        )
        assert int(w) == patch_pos_embed.shape[-2] and int(h) == patch_pos_embed.shape[-1]
        
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, hidden_size)
        
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)
    
    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): The input tensor. Shape: (batch_size, num_channels, height, width).
            
        Returns:
            torch.Tensor: The tokens. Shape: (batch_size, num_patches + 1, hidden_size).
        """
        batch_size, num_channels, height, width = pixel_values.shape
        
        embeddings = self.patch_embedding(pixel_values) # (batch_size, num_patches, hidden_size)
        
        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked positions with the mask token
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = torch.where(mask, mask_tokens, embeddings)
        
        # add [CLS] token to the beginning of the sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        
        if self.config.interpolate_pos_encoding:
            embeddings += self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings += self.position_embeddings
        
        embeddings = self.dropout(embeddings)
        
        return embeddings
        

class ViTSelfAttention(nn.Module):
    """ TODO: This is the self-attention layer. 
    It takes the output of the embedding layer and passes it through a series of linear layers and activations. """
    def __init__(self, config: ViTConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"hidden_size {config.hidden_size} must be divisible by the "
                f"number of attention heads {config.num_attention_heads}"
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """ Transpose the input tensor to get the scores for the attention heads. 
        Args:
            x (torch.Tensor): The input tensor. Shape: (batch_size, seq_length, hidden_size).
            
        Returns:
            torch.Tensor: The transposed tensor. Shape: (batch_size, num_attention_heads, seq_length, attention_head_size).
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape) # (batch_size, seq_length, num_attention_heads, attention_head_size)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, 
                head_mask: Optional[torch.Tensor] = None, 
                output_attentions: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states) # (batch_size, seq_length, self.all_head_size)
        key_layer = self.transpose_for_scores(self.key(hidden_states)) # (batch_size, num_attention_heads, seq_length, attention_head_size)
        value_layer = self.transpose_for_scores(self.value(hidden_states)) # (batch_size, num_attention_heads, seq_length, attention_head_size)
        query_layer = self.transpose_for_scores(mixed_query_layer) # (batch_size, num_attention_heads, seq_length, attention_head_size)
        
        # take the dot product to get the attention scores
        # this is batched matrix multiplication, (batch_size, num_attention_heads, seq_length, seq_length)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # mask out the entire tokens to attend to
        attention_probs = self.dropout(attention_probs)
        
        # mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context_layer = torch.matmul(attention_probs, value_layer) # (batch_size, num_attention_heads, seq_length, attention_head_size)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        
        return outputs
        
    

class ViTSdpaSelfAttention(nn.Module):
    """ TODO: This is the self-attention layer. 
    It takes the output of the embedding layer and passes it through a series of linear layers and activations. """
    

class ViTSelfOutput(nn.Module):
    """ TODO: This is the self-output layer. 
    It takes the output of the self-attention layer and passes it through a series of linear layers and activations. """
    

class ViTAttention(nn.Module):
    """ TODO: This is the attention layer. 
    It takes the output of the self-attention layer and passes it through a series of linear layers and activations. """
    