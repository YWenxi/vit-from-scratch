import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
from typing import Optional


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
    """ TODO: This is the embedding layer. 
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
    

class ViTSdpaSelfAttention(nn.Module):
    """ TODO: This is the self-attention layer. 
    It takes the output of the embedding layer and passes it through a series of linear layers and activations. """
    

class ViTSelfOutput(nn.Module):
    """ TODO: This is the self-output layer. 
    It takes the output of the self-attention layer and passes it through a series of linear layers and activations. """
    

class ViTAttention(nn.Module):
    """ TODO: This is the attention layer. 
    It takes the output of the self-attention layer and passes it through a series of linear layers and activations. """
    