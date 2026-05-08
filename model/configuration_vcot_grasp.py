"""
Modified from configuration_paligemma.py.
"""

import warnings

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import CONFIG_MAPPING, AutoConfig

from dataclasses import dataclass

logger = logging.get_logger(__name__)


class ArchConfig(PretrainedConfig):

    model_type = "vcot_grasp_architecture"

    def __init__(
        self,
        use_bbox=True,
        action_head="MLP",
        lm_new_extra_token_count=1024,
        token_embedding_pad_multiple=8,
        mlp_action_head=None,
        diffusion_action_head=None,
        grasp_position_dim=4,
        action_position_loss_scale=1.0,
        action_angle_loss_scale=0.1,
        diffusion_repeated_steps=8,
        **kwargs,
    ):
        self.use_bbox = use_bbox
        self.action_head = action_head
        self.lm_new_extra_token_count = lm_new_extra_token_count
        self.token_embedding_pad_multiple = token_embedding_pad_multiple
        self.mlp_action_head = mlp_action_head or {
            "num_blocks": 1,
            "hidden_dim": 256,
            "output_dim": 1,
        }
        self.diffusion_action_head = diffusion_action_head or {
            "token_size": 5,
            "model_type": "DiT-S",
            "in_channels": 5,
            "future_action_window_size": 0,
            "past_action_window_size": 0,
        }
        self.grasp_position_dim = grasp_position_dim
        self.action_position_loss_scale = action_position_loss_scale
        self.action_angle_loss_scale = action_angle_loss_scale
        self.diffusion_repeated_steps = diffusion_repeated_steps
        super().__init__(**kwargs)

    def to_dict(self):
        return {
            "model_type": self.model_type,
            "transformers_version": self.transformers_version,
            "use_bbox": self.use_bbox,
            "action_head": self.action_head,
            "lm_new_extra_token_count": self.lm_new_extra_token_count,
            "token_embedding_pad_multiple": self.token_embedding_pad_multiple,
            "mlp_action_head": self.mlp_action_head,
            "diffusion_action_head": self.diffusion_action_head,
            "grasp_position_dim": self.grasp_position_dim,
            "action_position_loss_scale": self.action_position_loss_scale,
            "action_angle_loss_scale": self.action_angle_loss_scale,
            "diffusion_repeated_steps": self.diffusion_repeated_steps,
        }


class VCoTGraspConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PaliGemmaForConditionalGeneration`]. It is used to instantiate an
    PaliGemmamodel according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the PaliGemma-2B.

    e.g. [paligemma-hf/paligemma-2b](https://huggingface.co/paligemma-hf/paligemma-2b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`PaliGemmaVisionConfig`,  *optional*):
            Custom vision config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 256000):
            The image token index to encode the image prompt.
        vocab_size (`int`, *optional*, defaults to 257152):
            Vocabulary size of the PaliGemmamodel. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~PaliGemmaForConditionalGeneration`]
        projection_dim (`int`, *optional*, defaults to 2048):
            Dimension of the multimodal projection space.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden layer of the Language model.

    Example:

    ```python
    >>> from transformers import PaliGemmaForConditionalGeneration, PaliGemmaConfig, SiglipVisionConfig, GemmaConfig

    >>> # Initializing a Siglip-like vision config
    >>> vision_config = SiglipVisionConfig()

    >>> # Initializing a PaliGemma config
    >>> text_config = GemmaConfig()

    >>> # Initializing a PaliGemma paligemma-3b-224 style configuration
    >>> configuration = PaliGemmaConfig(vision_config, text_config)

    >>> # Initializing a model from the paligemma-3b-224 style configuration
    >>> model = PaliGemmaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vcot_grasp"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig, "arch_config": ArchConfig}

    @staticmethod
    def build_attn_implementation(text_config="sdpa", vision_config="sdpa"):
        return {
            "text_config": text_config,
            "vision_config": vision_config,
        }

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=257152,
        dummy_action_token_index=257153,
        vocab_size=257216,
        projection_dim=2304,
        hidden_size=2304,
        arch_config=None,
        **kwargs,
    ):
        self._ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.dummy_action_token_index = dummy_action_token_index
        self._vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False

        if isinstance(self.vision_config, dict):
            vision_config["model_type"] = vision_config["model_type"] if "model_type" in vision_config else "siglip_vision_model"
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                intermediate_size=4096,
                hidden_size=1152,
                patch_size=14,
                image_size=224,
                num_hidden_layers=27,
                num_attention_heads=16,
                vocab_size=257152,
                vision_use_head=False,
            )

        self.text_config = text_config
        if isinstance(self.text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "gemma"
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["gemma"](
                hidden_size=2048,
                num_hidden_layers=18,
                intermediate_size=16384,
                num_attention_heads=8,
                num_key_value_heads=1,
                is_encoder_decoder=False,
                vocab_size=vocab_size,
            )
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

        if isinstance(arch_config, dict):
            self.arch_config = ArchConfig(**arch_config)
        elif arch_config is None:
            self.arch_config = ArchConfig(False, "None")

        super().__init__(**kwargs)

    def set_attn_implementation(self, text_config="sdpa", vision_config="sdpa"):
        self._attn_implementation = self.build_attn_implementation(
            text_config=text_config,
            vision_config=vision_config,
        )
        return self

    @property
    def ignore_index(self):
        warnings.warn(
            "The `ignore_index` attribute is deprecated and will be removed in v4.47.",
            FutureWarning,
        )
        return self._ignore_index

    @ignore_index.setter
    def ignore_index(self, value):
        self._ignore_index = value

    def to_dict(self):
        output = super().to_dict()
        output.pop("_ignore_index", None)
        return output
