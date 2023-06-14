from .clip import CLIP
from dataclasses import dataclass
from typing import Union
import numpy as np
from tensorflow import keras
from .visual_transformer import VisualTransformer

@dataclass
class ClipConfig(): # Defaults set to match ViT-B/16
    embed_dim: int = 512 
    image_resolution: int = 224
    vision_layers: Union[int, tuple[int, int, int, int]] = 12 
    vision_width: int = 768
    vision_patch_size: int = 16
    context_length: int = 77
    vocab_size: int = 49408
    transformer_width: int = 512
    transformer_layers: int = 12

    @property
    def transformer_heads(self):
        return self.transformer_width // 64


_Models: dict[str, ClipConfig] = { # Each model has a different set of parameters
    'RN50': ClipConfig(embed_dim = 1024, vision_layers = (3, 4, 6, 3), vision_width = 64, vision_patch_size = None),
    'RN101': ClipConfig(vision_layers = (3, 4, 23, 3), vision_width = 64, vision_patch_size = None),
    'RN50x4': ClipConfig(embed_dim = 640, image_resolution = 288, vision_layers = (4, 6, 10, 6), vision_width = 80, vision_patch_size = None, transformer_width = 640),
    'RN50x16': ClipConfig(embed_dim = 768, image_resolution = 384, vision_layers = (6, 8, 18, 8), vision_width = 96, vision_patch_size = None, transformer_width = 768),
    'RN50x64': ClipConfig(embed_dim = 1024, image_resolution = 448, vision_layers = (3, 15, 36, 10), vision_width = 128, vision_patch_size = None, transformer_width = 1024),
    'ViT-B_32': ClipConfig(vision_patch_size = 32),
    'ViT-B_16': ClipConfig(), # Default so no need to pass anything
    'ViT-L_14': ClipConfig(embed_dim = 768, vision_layers = 24, vision_width = 1024, vision_patch_size = 14, transformer_width = 768),
    'ViT-L_14@336px': ClipConfig(embed_dim = 768, image_resolution = 336, vision_layers = 24, vision_width = 1024, vision_patch_size = 14, transformer_width = 768),
    'FaRL-B_16-64': ClipConfig(), # Duplicate of ViT-B/16 just different weights
}


def build_model(model_name: str, visual: bool = True, text: bool = True) -> keras.Model:
    """
    Builds and returns a CLIP model

    Args:
        @param model_name (str): The name of the model configuration to use It takes one of these Model types %s % _Models
        visual (bool): Whether to build the visual CLIP model. Defaults to True.
        text (bool): Whether to build the text CLIP model. Defaults to True.

    Returns:
        keras.Model: The CLIP model with the specified configuration and weights loaded.
    """
    config = _Models[model_name]

    model = CLIP(config.embed_dim, config.image_resolution, config.vision_layers, config.vision_width, config.vision_patch_size,
        config.context_length, config.vocab_size, config.transformer_width, config.transformer_heads, config.transformer_layers)

    #Model must be built to load weights
    empty_image = np.ones((1, config.image_resolution, config.image_resolution, 3), np.float32)
    empty_text = np.ones((1, 4, config.context_length), np.int32)
    model.predict((empty_image, empty_text))

    # model.load_weights(f'/home/nikkelitous/Documents/Projects/CLIP-tf2/models/CLIP_{model_name}.h5') #TODO replace with model download + cache
    model.config = config

    return model

def build_visual_model(model_name: str) -> keras.Model:
    """
    Builds and returns the visual CLIP model.

    Args:
        @param model_name (str): The name of the model configuration to use It takes one of these Model types %s % _Models

    Returns:
        keras.Model: The visual CLIP model with the specified configuration.
    """
    config = _Models[model_name]
    model = VisualTransformer(
                input_resolution=config.image_resolution,
                patch_size=config.vision_patch_size,
                width=config.vision_width,
                layers=config.vision_layers,
                heads=config.vision_layers//64,
                output_dim=config.embed_dim,
                name="visual"
            )
    return model