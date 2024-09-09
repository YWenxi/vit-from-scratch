import torch
import torch.nn as nn
import math
import collections


class ViTPatchEmbedding(nn.Module):
    """ TODO: This is the patch embedding layer. 
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
    """ TODO: This is the embedding layer. It takes the patch embedding and adds a class token and position embedding. """
    

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
    