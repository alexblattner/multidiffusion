# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
from typing import Callable, List, Optional, Tuple, Union, Dict, Any
from packaging import version
import numpy as np
import torch.nn.functional as F
import PIL
import re
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
import torchvision
from diffusers import DiffusionPipeline, ControlNetModel
from diffusers.configuration_utils import FrozenDict
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin,FromCkptMixin
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.controlnet import ControlNetOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler,KarrasDiffusionSchedulers
from diffusers.utils import deprecate, is_accelerate_available, is_accelerate_version, logging, randn_tensor, replace_example_docstring,PIL_INTERPOLATION
from diffusers.models import ModelMixin
import kohya_lora_loader

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""
class MultiControlNetModel(ModelMixin):
    r"""
    Multiple `ControlNetModel` wrapper class for Multi-ControlNet
    This module is a wrapper for multiple instances of the `ControlNetModel`. The `forward()` API is designed to be
    compatible with `ControlNetModel`.
    Args:
        controlnets (`List[ControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. You must set multiple
            `ControlNetModel` as a list.
    """

    def __init__(self, controlnets: Union[List[ControlNetModel], Tuple[ControlNetModel]]):
        super().__init__()
        self.nets = nn.ModuleList(controlnets)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: List[torch.tensor],
        conditioning_scale: List[float],
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple]:
        for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, self.nets)):
            down_samples, mid_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states,
                image,
                scale,
                class_labels,
                timestep_cond,
                attention_mask,
                cross_attention_kwargs,
                guess_mode,
                return_dict,
            )

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        return down_block_res_samples, mid_block_res_sample # type: ignore[return-value]

class MultiStableDiffusion(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromCkptMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *Textual-Inversion*: [`loaders.TextualInversionLoaderMixin.load_textual_inversion`]
        - *LoRA*: [`loaders.LoraLoaderMixin.load_lora_weights`]
        - *Ckpt*: [`loaders.FromCkptMixin.from_ckpt`]

    as well as the following saving methods:
        - *LoRA*: [`loaders.LoraLoaderMixin.save_lora_weights`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        self.controlnet=None
        kohya_lora_loader.install_lora_hook(self)
    def loadControlnet(self, controlnet):
        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)
        self.controlnet=controlnet
    def loadImage(self, image):
        if not hasattr(self, 'image_processor'):
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.image=self.image_processor.preprocess(image)
    def prepare_controlnet_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError("`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        strength,
        height,
        width,
        callback_steps=None,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pos=None,
        mask_types=None,
        controlnet_conditioning_scale=1.0,
        image=None,
    ):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")
        chosenPrompt = prompt
        if prompt_embeds is not None:
            chosenPrompt = prompt_embeds
        chosenNegativePrompt= negative_prompt
        if negative_prompt is not None:
            chosenNegativePrompt = negative_prompt_embeds
        if len(chosenPrompt)!=len(pos):
            raise ValueError(f"`prompt` and `pos` have to be the same length but are {len(chosenPrompt)} and {len(pos)}.")
        if len(chosenPrompt)!=len(mask_types):
            raise ValueError(f"`prompt` and `mask_types` have to be the same length but are {len(chosenPrompt)} and {len(mask_types)}.")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        if not isinstance(pos, list):
            raise ValueError(f"`pos` has to be a list of strings but is {pos}.")
        else:
            for i in pos:
                if not isinstance(i, str) and not isinstance(i, PIL.Image.Image):
                    raise ValueError(f"`pos` has to be a list of strings or PIL.Image.Image but is {pos}.")
                elif isinstance(i, str):
                    pos_base = i.split("-")
                    if len(pos_base) != 2:
                        raise ValueError(
                            f"`pos` has to be a list of strings with the format `layer:row:col` but is {pos}."
                        )
                    else:
                        pos_dev = pos_base[0].split(":")
                        if len(pos_dev)!=2:
                            raise ValueError(
                                f"`pos` has to be a list of strings with the format `layer:row:col` but is {pos}."
                            )
                        else:
                            if not pos_dev[0].isdigit() or not pos_dev[1].isdigit():
                                raise ValueError(
                                    f"`pos` has to be a list of strings with the format `layer:row:col` but is {pos}."
                                )
                        pos_pos = pos_base[1].split(":")
                        if len(pos_base) != 2:
                            raise ValueError(
                                f"`pos` has to be a list of strings with the format `layer:row:col` but is {pos}."
                            )
                        else:
                            if not pos_dev[0].isdigit() or not pos_dev[1].isdigit():
                                raise ValueError(
                                    f"`pos` has to be a list of strings with the format `layer:row:col` but is {pos}."
                                )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        if self.controlnet is not None:
            
            if isinstance(self.controlnet, MultiControlNetModel):
                if isinstance(prompt, list):
                    logger.warning(
                        f"You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)}"
                        " prompts. The conditionings will be fixed across the prompts."
                    )
            if self.controlnet is not None:
                is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
                    self.controlnet, torch._dynamo.eval_frame.OptimizedModule
                )
                if (
                    isinstance(self.controlnet, ControlNetModel)
                    or is_compiled
                    and isinstance(self.controlnet._orig_mod, ControlNetModel)
                ):
                    print('')
                elif (
                    isinstance(self.controlnet, MultiControlNetModel)
                    or is_compiled
                    and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
                ):
                    if not isinstance(image, list):
                        raise TypeError("For multiple controlnets: `image` must be type `list`")

                    # When `image` is a nested list:
                    # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
                    elif any(isinstance(i, list) for i in image):
                        raise ValueError("A single batch of multiple conditionings are supported at the moment.")
                    elif len(image) != len(self.controlnet.nets):
                        raise ValueError(
                            "For multiple controlnets: `image` must have the same length as the number of controlnets."
                        )

                else:
                    assert False

                # Check `controlnet_conditioning_scale`
                if (
                    isinstance(self.controlnet, ControlNetModel)
                    or is_compiled
                    and isinstance(self.controlnet._orig_mod, ControlNetModel)
                ):
                    if not isinstance(controlnet_conditioning_scale, float):
                        raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
                elif (
                    isinstance(self.controlnet, MultiControlNetModel)
                    or is_compiled
                    and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
                ):
                    if isinstance(controlnet_conditioning_scale, list):
                        if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                            raise ValueError("A single batch of multiple conditionings are supported at the moment.")
                    elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(
                        self.controlnet.nets
                    ):
                        raise ValueError(
                            "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                            " the same length as the number of controlnets"
                        )
                else:
                    assert False

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    def prepare_latents_image(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        pos: List[Union[str, PIL.Image.Image]],
        mask_types: List[float] = [1],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        strength: float = 0.5,
        loras_apply: Optional[dict]=None,
        controlnet_image: Optional[Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]]] = None,
        controlnet_conditioning_scale: Optional[Union[float, List[float]]] = None,
        image: Optional[Union[torch.FloatTensor, PIL.Image.Image]] = None,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        remaining_height = height % 8
        remaining_width = width % 8
        height+=remaining_height
        width+=remaining_width
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, strength, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds, pos, mask_types, controlnet_conditioning_scale=controlnet_conditioning_scale, image=controlnet_image
        )
        all_loras=None
        if loras_apply is not None:
            all_loras={}
            for i in loras_apply.values():
                for j in i:
                    if ":" in j:
                        parts = j.split(":")  # split the string at the ":" index to seperate the multiplier
                        j = parts[0]  # pick the actual name of the lora
                    all_loras[j]=None
            for i in all_loras.keys():
                all_loras[i]=self.apply_lora(i, 0.0, self.unet.dtype).to('cuda')
        plen= len(prompt) if prompt is not None  else len(prompt_embeds)
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt[0], str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt[0], list):
            batch_size = len(prompt[0])
        else:
            batch_size = prompt_embeds[0].shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # 3. Encode input prompt
        text_embeddings = []
        bte=None
        for i in range(plen):
            pinput=prompt
            if isinstance(prompt, list):
                pinput=prompt[i]
            peinput=prompt_embeds
            if isinstance(prompt_embeds, list):
                peinput=prompt_embeds[i]
            ninput=negative_prompt
            if isinstance(negative_prompt, list):
                ninput=negative_prompt[i]
            neinput=negative_prompt_embeds
            if isinstance(negative_prompt_embeds, list):
                neinput=negative_prompt_embeds[i]
            one_text_embeddings = self._encode_prompt(
                pinput, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt=ninput, prompt_embeds=peinput, negative_prompt_embeds=neinput
            )
            if i==0:
                be=one_text_embeddings
            text_embeddings.append(one_text_embeddings)
        text_embeddings.append(be)
        image=None
        if hasattr(self, 'image'):
            image = self.image
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timestep = None

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents=None
        if image is not None:
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
            latents = self.prepare_latents_image(
                image, latent_timestep, batch_size, num_images_per_prompt, dtype=text_embeddings[0].dtype, device=device,generator=generator,
            )
        else:
            timesteps = self.scheduler.timesteps
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                text_embeddings[0].dtype,
                device,
                generator,
                latents,
            )
        if controlnet_image is not None:
            is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
                self.controlnet, torch._dynamo.eval_frame.OptimizedModule
            )
            if (
                isinstance(self.controlnet, ControlNetModel)
                or is_compiled
                and isinstance(self.controlnet._orig_mod, ControlNetModel)
            ):
                controlnet_image = self.prepare_controlnet_image(
                    image=controlnet_image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=self.controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )
            elif (
                isinstance(self.controlnet, MultiControlNetModel)
                or is_compiled
                and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
            ):
                images = []

                for image_ in controlnet_image:
                    image_ = self.prepare_controlnet_image(
                        image=image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=self.controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                    )

                    images.append(image_)

                controlnet_image = images
            else:
                assert False
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        # 7. Denoising loop
        # num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        def create_rectangular_mask(height, width, y_start, x_start, block_height, block_width, device='cpu'):
            mask = torch.zeros(height, width, device=device)
            mask[y_start:y_start + block_height, x_start:x_start + block_width] = 1
            return mask

        mask_list = []
        for i in range(plen):
            one_filter = None
            if isinstance(pos[i], str):
                pos_base = pos[i].split("-")
                pos_start = pos_base[0].split(":")
                pos_end = pos_base[1].split(":")
                
                block_height = abs(int(pos_start[1]) - int(pos_end[1])) // 8
                block_width = abs(int(pos_start[0]) - int(pos_end[0])) // 8
                y_start = int(pos_start[1]) // 8
                x_start = int(pos_start[0]) // 8
                one_filter = create_rectangular_mask(height // 8, width // 8, y_start, x_start, block_height, block_width, device=device)
                # one_filter=one_filter.unsqueeze(0).expand(batch_size, 4, -1, -1).to(torch.float16)
            else:
                img = pos[i].convert('L').resize((width // 8, height // 8))

                # Convert image data to a numpy array
                np_data = np.array(img)

                # Normalize the data to range between 0 and 1
                np_data = np_data / 255

                np_data = (np_data > 0.5).astype(np.float32)
                # Convert the numpy array to a PyTorch tensor
                mask = torch.from_numpy(np_data)

                # Convert the numpy array to a PyTorch tensor
                one_filter = mask.to('cuda')
                # one_filter = one_filter.unsqueeze(0)
                # one_filter = one_filter.unsqueeze(0).expand(batch_size, 4, -1, -1).to(torch.float16)

            mask_list.append(one_filter)

        # For each pixel
        for x in range(height//8):
            for y in range(width//8):
                # Get the indices of the masks that are applied to this pixel
                applied_mask_indices = [idx for idx, mask in enumerate(mask_list) if mask[x, y] > 0]

                if applied_mask_indices:
                    mask_strengths = [mask_types[idx] for idx in applied_mask_indices]
                    # Calculate the weights for the applied masks
                    totalM=0
                    multi=len(mask_strengths)>2
                    pxvals=dict()
                    for i in applied_mask_indices:
                        val=mask_list[i][x, y].item()
                        val=val*mask_types[i]
                        pxvals[i]=val
                        totalM+=val
                    total_weights=0
                    for i in applied_mask_indices:
                        w=(pxvals[i]/totalM)
                        mask_list[i][x, y] *= w
                        total_weights+=w
                    if total_weights>1:
                        mask_list[applied_mask_indices[0]][x, y] -= total_weights-1
                    elif total_weights<1:
                        mask_list[applied_mask_indices[0]][x, y] += 1-total_weights
                else:
                    raise ValueError(
                            "unoccupied pixel in the mask. {x}, {y}"
                        )
        colorful_images=[]
        for i in range(len(mask_list)):
            rgb_image = np.zeros((mask_list[i].shape[0], mask_list[i].shape[1], 3))

            # Fill the red channel with value 1
            rgb_image[:,:,i] = 1

            # Create an RGBA image by adding the normalized grayscale image as the alpha channel
            rgba_image = np.zeros((mask_list[i].shape[0], mask_list[i].shape[1], 4))
            rgba_image[:,:,:3] = rgb_image
            normalized_image = mask_list[i].cpu().numpy()
            rgba_image[:,:,3] = normalized_image

            # Convert to range 0-255 and type uint8
            rgba_image = (rgba_image * 255).astype(np.uint8)

            # Convert the numpy array to a PIL image and save it
            colorful_images.append(PIL.Image.fromarray(rgba_image))
            mask_list[i] = mask_list[i].unsqueeze(0).expand(batch_size, 4, -1, -1).to(torch.float16)
            torchvision.transforms.functional.to_pil_image(mask_list[i][0]*256).save(str(i)+".png")
        final_color=colorful_images[0].convert('RGBA')
        for i in range(len(colorful_images)):
            if i==0:
                continue
            final_color=PIL.Image.alpha_composite(final_color, colorful_images[i])
        final_color.save("final.png")
        for i, t in enumerate(self.progress_bar(timesteps)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # predict the noise residual
                noise_preds = []
                for i in range(len(mask_list)):
                    if loras_apply is not None and loras_apply.get(i) is not None:
                        # self.load_lora_weights(loras_apply.get(i))
                        for j in loras_apply.get(i):
                            num_apply=1.0
                            if ":" in j:
                                parts = j.split(":")  # split the string at the ":" index to seperate the multiplier
                                j = parts[0]  # pick the actual name of the lora
                                num_apply = float(parts[1])  # pick the multiplier
                            all_loras[j].alpha=num_apply
                    noise_pred=None
                    if controlnet_image is not None:
                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=text_embeddings[i],
                            controlnet_cond=controlnet_image,
                            conditioning_scale=controlnet_conditioning_scale,
                            guess_mode=False,
                            return_dict=False,
                        )
                        noise_pred=self.unet(
                            latent_model_input, 
                            t, 
                            encoder_hidden_states=text_embeddings[i],
                            cross_attention_kwargs=cross_attention_kwargs,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                            return_dict=False,
                        )[0]
                    else:
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings[i]).sample
                    noise_preds.append(noise_pred)
                    if loras_apply is not None and loras_apply.get(i) is not None:
                        for j in loras_apply.get(i):
                            if ":" in j:
                                parts = j.split(":")  # split the string at the ":" index to seperate the multiplier
                                j = parts[0]  # pick the actual name of the lora
                            all_loras[j].alpha=0.0
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_unconds = []
                    noise_pred_texts = []
                    for i in range(len(mask_list)):
                      noise_pred_uncond, noise_pred_text = noise_preds[i].chunk(2)
                      noise_pred_unconds.append(noise_pred_uncond)
                      noise_pred_texts.append(noise_pred_text)
                    
                    result = None
                    noise_preds = []
                    for i in range(len(mask_list)):
                      noise_pred = noise_pred_unconds[i] + guidance_scale * (noise_pred_texts[i] - noise_pred_unconds[i])
                      noise_preds.append(noise_pred)
                    result = noise_preds[0] * mask_list[0]
                    for i in range(1, len(mask_list)):
                      result += noise_preds[i] * mask_list[i]

                    #noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(result, t, latents, **extra_step_kwargs).prev_sample
                    if i==15:
                        tmg=self.decode_latents(latents)
                        tmg.save("tmg.png")
                    # call the callback, if provided
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            # image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings[0].dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            # image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings[0].dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=False)
