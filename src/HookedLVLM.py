import requests
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
)
from IPython.display import display
import torch
from PIL import Image
from contextlib import contextmanager
from typing import Callable, Union, Dict, Any
import os
import yaml

file_path = os.path.dirname(__file__)
model_cache_dir = os.path.join(file_path, "..", "models")


class HookedLVLM:
    """Hooked LVLM."""

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        hook_loc: str = "text_model_in",
        device: str = "cuda:0",
        quantize: bool = False,
        quantize_type: str = "fp16",
    ):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device,
            cache_dir=model_cache_dir,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.hook_loc = hook_loc
        self.data = None

    def forward(
        self,
        image_path_or_image: Union[str, Image.Image],
        prompt: str,
        output_hidden_states=False,
        output_attentions=False,
    ):

        # Open image if needed
        if isinstance(image_path_or_image, str):
            image = Image.open(image_path_or_image)
        else:
            image = image_path_or_image

        # Prepare inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs.to(self.model.device)

        # Run forward pass
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions
            )

        return outputs
