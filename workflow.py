import os
import random
import sys
import time
from typing import Sequence, Mapping, Any, Union
import torch
from PIL import Image
import numpy as np

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    try:
        # Try async version first
        loop.run_until_complete(init_extra_nodes())
    except TypeError:
        # Fallback to sync version if it's not async
        init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


class DetailerWorkflow:
    def __init__(self):
        print("Starting model initialization...")
        start_time = time.time()
        import_custom_nodes()
        
        # Initialize node class mappings
        self.fluxloader = NODE_CLASS_MAPPINGS["FluxLoader"]()
        self.pulidfluxmodelloader = NODE_CLASS_MAPPINGS["PulidFluxModelLoader"]()
        self.pulidfluxinsightfaceloader = NODE_CLASS_MAPPINGS["PulidFluxInsightFaceLoader"]()
        self.text_multiline = NODE_CLASS_MAPPINGS["Text Multiline"]()
        self.pulidfluxevacliploader = NODE_CLASS_MAPPINGS["PulidFluxEvaClipLoader"]()
        self.text_concatenate = NODE_CLASS_MAPPINGS["Text Concatenate"]()
        self.checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        self.cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        self.seed_rgthree = NODE_CLASS_MAPPINGS["Seed (rgthree)"]()
        self.clownoptions_cycles_beta = NODE_CLASS_MAPPINGS["ClownOptions_Cycles_Beta"]()
        self.vram_debug = NODE_CLASS_MAPPINGS["VRAM_Debug"]()
        self.supir_model_loader_v2 = NODE_CLASS_MAPPINGS["SUPIR_model_loader_v2"]()
        self.upscale_model_loader = NODE_CLASS_MAPPINGS["Upscale Model Loader"]()
        self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        self.imagescaletototalpixels = NODE_CLASS_MAPPINGS["ImageScaleToTotalPixels"]()
        self.easy_humansegmentation = NODE_CLASS_MAPPINGS["easy humanSegmentation"]()
        self.mask_crop_region = NODE_CLASS_MAPPINGS["Mask Crop Region"]()
        self.image_crop_location = NODE_CLASS_MAPPINGS["Image Crop Location"]()
        self.imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
        self.imagescaleby = NODE_CLASS_MAPPINGS["ImageScaleBy"]()
        self.imageresize = NODE_CLASS_MAPPINGS["ImageResize+"]()
        self.supir_encode = NODE_CLASS_MAPPINGS["SUPIR_encode"]()
        self.controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
        self.stylemodelloader = NODE_CLASS_MAPPINGS["StyleModelLoader"]()
        self.supir_conditioner = NODE_CLASS_MAPPINGS["SUPIR_conditioner"]()
        self.supir_sample = NODE_CLASS_MAPPINGS["SUPIR_sample"]()
        self.supir_decode = NODE_CLASS_MAPPINGS["SUPIR_decode"]()
        self.colormatch = NODE_CLASS_MAPPINGS["ColorMatch"]()
        self.easy_cleangpuused = NODE_CLASS_MAPPINGS["easy cleanGpuUsed"]()
        self.vaeencodeadvanced = NODE_CLASS_MAPPINGS["VAEEncodeAdvanced"]()
        self.applypulidflux = NODE_CLASS_MAPPINGS["ApplyPulidFlux"]()
        self.loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        self.fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        self.conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        self.setshakkerlabsunioncontrolnettype = NODE_CLASS_MAPPINGS["SetShakkerLabsUnionControlNetType"]()
        self.midas_depthmappreprocessor = NODE_CLASS_MAPPINGS["MiDaS-DepthMapPreprocessor"]()
        self.facedepthmapnode = NODE_CLASS_MAPPINGS["FaceDepthMapNode"]()
        self.image_overlay = NODE_CLASS_MAPPINGS["Image Overlay"]()
        self.controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
        self.lineartpreprocessor = NODE_CLASS_MAPPINGS["LineArtPreprocessor"]()
        self.clipvisionstyleloader = NODE_CLASS_MAPPINGS["ClipVisionStyleLoader"]()
        self.reduxfinetune = NODE_CLASS_MAPPINGS["ReduxFineTune"]()
        self.clownsharksampler_beta = NODE_CLASS_MAPPINGS["ClownsharKSampler_Beta"]()
        self.clownsharkchainsampler_beta = NODE_CLASS_MAPPINGS["ClownsharkChainsampler_Beta"]()
        self.clownguide_style_beta = NODE_CLASS_MAPPINGS["ClownGuide_Style_Beta"]()
        self.vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        self.getimagesize = NODE_CLASS_MAPPINGS["GetImageSize+"]()
        self.imageresizekjv2 = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
        self.growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()
        self.facealignwarpnode = NODE_CLASS_MAPPINGS["FaceAlignWarpNode"]()
        self.basicskinblender = NODE_CLASS_MAPPINGS["BasicSkinBlender"]()
        self.facesegment = NODE_CLASS_MAPPINGS["FaceSegment"]()
        self.layerutility_imageblend = NODE_CLASS_MAPPINGS["LayerUtility: ImageBlend"]()
        self.vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
        self.lora_loader = NODE_CLASS_MAPPINGS["Lora Loader"]()
        self.mxslider = NODE_CLASS_MAPPINGS["mxSlider"]()
        self.image_comparer_rgthree = NODE_CLASS_MAPPINGS["Image Comparer (rgthree)"]()
        self.mathexpressionpysssss = NODE_CLASS_MAPPINGS["MathExpression|pysssss"]()
        self.ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        self.masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
        self.image_blend_by_mask = NODE_CLASS_MAPPINGS["Image Blend by Mask"]()
        self.image_paste_crop = NODE_CLASS_MAPPINGS["Image Paste Crop"]()
        self.layerutility_purgevram = NODE_CLASS_MAPPINGS["LayerUtility: PurgeVRAM"]()

        # Load models and configurations
        self._load_models()
        
        end_time = time.time()
        load_time = end_time - start_time
        print(f"Model initialization completed in {load_time:.2f} seconds")

    def _load_models(self):
        """Load all models and configurations"""
        # Flux loader
        self.fluxloader_98 = self.fluxloader.main(
            model_name="FluxFace_00001_.safetensors",
            weight_dtype="default",
            clip_name1=".use_ckpt_clip",
            clip_name2_opt=".none",
            vae_name=".use_ckpt_vae",
            clip_vision_name="sigclip_vision_patch14_384.safetensors",
            style_model_name="flux1-redux-dev.safetensors",
        )

        # Pulid flux model loader
        self.pulidfluxmodelloader_108 = self.pulidfluxmodelloader.load_model(
            pulid_file="pulid_flux_v0.9.1.safetensors"
        )

        # Pulid flux insight face loader
        self.pulidfluxinsightfaceloader_109 = self.pulidfluxinsightfaceloader.load_insightface(
            provider="CPU"
        )

        # Pulid flux eva clip loader
        self.pulidfluxevacliploader_139 = self.pulidfluxevacliploader.load_eva_clip()

        # Checkpoint loader simple
        self.checkpointloadersimple_183 = self.checkpointloadersimple.load_checkpoint(
            ckpt_name="FluxFace_00001_.safetensors"
        )

        # Clown options cycles beta
        self.clownoptions_cycles_beta_238 = self.clownoptions_cycles_beta.main(
            cycles=5,
            eta_decay_scale=1,
            unsample_eta=0.5,
            unsampler_override="none",
            unsample_steps_to_run=-1,
            unsample_cfg=1,
            unsample_bongmath=False,
        )

        # Checkpoint loader simple for SDXL
        self.checkpointloadersimple_267 = self.checkpointloadersimple.load_checkpoint(
            ckpt_name="juggernaut_ragnarok_sdxl.safetensors"
        )

        # VRAM debug
        self.vram_debug_276 = self.vram_debug.VRAMdebug(
            empty_cache=True,
            gc_collect=True,
            unload_all_models=True,
            model_pass=get_value_at_index(self.checkpointloadersimple_267, 0),
        )

        # SUPIR model loader v2
        self.supir_model_loader_v2_262 = self.supir_model_loader_v2.process(
            supir_model="SUPIR/SUPIR-v0F.ckpt",
            fp8_unet=False,
            diffusion_dtype="auto",
            high_vram=False,
            model=get_value_at_index(self.vram_debug_276, 2),
            clip=get_value_at_index(self.checkpointloadersimple_267, 1),
            vae=get_value_at_index(self.checkpointloadersimple_267, 2),
        )

        # Upscale model loader
        self.upscale_model_loader_273 = self.upscale_model_loader.load_model(
            model_name="RealESRGAN_x4.pth"
        )

        # Control net loader
        self.controlnetloader_312 = self.controlnetloader.load_controlnet(
            control_net_name="flux_shakker_labs_union_pro-fp8_e4m3fn.safetensors"
        )

        # Style model loader
        self.stylemodelloader_391 = self.stylemodelloader.load_style_model(
            style_model_name="flux1-redux-dev.safetensors"
        )

        # SDXL checkpoint loader for final processing
        self.checkpointloadersimple_409 = self.checkpointloadersimple.load_checkpoint(
            ckpt_name="juggernaut_ragnarok_sdxl.safetensors"
        )

    def _load_lora_models(self, applypulidflux_112=None, checkpointloadersimple_409=None):
        """Load LoRA models after base models are ready"""
        # LoRA loaders for Flux model
        if applypulidflux_112 is not None:
            self.loraloadermodelonly_437 = self.loraloadermodelonly.load_lora_model_only(
                lora_name="closeupface-v1.safetensors",
                strength_model=0.5,
                model=get_value_at_index(applypulidflux_112, 0),
            )

            self.loraloadermodelonly_438 = self.loraloadermodelonly.load_lora_model_only(
                lora_name="Flux_pro_extreme_details.safetensors",
                strength_model=0.5,
                model=get_value_at_index(self.loraloadermodelonly_437, 0),
            )

            self.loraloadermodelonly_439 = self.loraloadermodelonly.load_lora_model_only(
                lora_name="Hyper-FLUX.1-dev-16steps-lora.safetensors",
                strength_model=0.1,
                model=get_value_at_index(self.loraloadermodelonly_438, 0),
            )

        # LoRA loaders for SDXL model
        if checkpointloadersimple_409 is not None:
            self.lora_loader_414 = self.lora_loader.load_lora(
                lora_name="skin_4-000015.safetensors",
                strength_model=1,
                strength_clip=1,
                model=get_value_at_index(checkpointloadersimple_409, 0),
                clip=get_value_at_index(checkpointloadersimple_409, 1),
            )

            self.loraloadermodelonly_440 = self.loraloadermodelonly.load_lora_model_only(
                lora_name="skin_1.safetensors",
                strength_model=1,
                model=get_value_at_index(self.lora_loader_414, 0),
            )

    @torch.inference_mode()
    def __call__(self, *args, **kwargs):
        print("Starting inference...")
        inference_start_time = time.time()
        user_prompt = kwargs.get("user_prompt", "")
        fixed_positive = kwargs.get("fixed_positive", "a face with realistic skin texture, blemishes\n\n\nraw photo, (realistic:1.5), (skin details, skin texture:0.45), (skin pores:0.45), (skin imperfections:0.19), (eyes details:0.44), selfie-style amateur photography, film grain")
        user_negative = kwargs.get("user_negative", "")
        fixed_negative = kwargs.get("fixed_negative", "blurry, out of focus, cartoon, poor quality,  bokkeh, depth of field, tan, sunburned, professional photo, studio lighting, NSFW, nudity")
        seed = kwargs.get("seed", random.randint(1, 2**64))
        slider_value = kwargs.get("denoise", 0.21)
        image_path = kwargs.get("image_path", "input_image.png")
        
        with torch.inference_mode():
            # Generate text multiline inputs
            text_multiline_127 = self.text_multiline.text_multiline(text=user_prompt)
            text_multiline_138 = self.text_multiline.text_multiline(text=fixed_positive)

            text_multiline_230 = self.text_multiline.text_multiline(text=user_negative)
            text_multiline_232 = self.text_multiline.text_multiline(text=fixed_negative)

            # Concatenate texts
            text_concatenate_231 = self.text_concatenate.text_concatenate(
                delimiter=", ",
                clean_whitespace="true",
                text_a=get_value_at_index(text_multiline_230, 0),
                text_b=get_value_at_index(text_multiline_232, 0),
            )

            text_concatenate_137 = self.text_concatenate.text_concatenate(
                delimiter=", ",
                clean_whitespace="true",
                text_a=get_value_at_index(text_multiline_127, 0),
                text_b=get_value_at_index(text_multiline_138, 0),
            )

            # Encode texts
            cliptextencode_178 = self.cliptextencode.encode(
                text=get_value_at_index(text_concatenate_231, 0),
                clip=get_value_at_index(self.checkpointloadersimple_183, 1),
            )

            cliptextencode_188 = self.cliptextencode.encode(
                text=get_value_at_index(text_concatenate_137, 0),
                clip=get_value_at_index(self.checkpointloadersimple_183, 1),
            )

            # Generate random seed
            seed_rgthree_185 = self.seed_rgthree.main(
                seed=seed, unique_id=13049319002350842661
            )

            # Load image
            loadimage_301 = self.loadimage.load_image(image=image_path)

            # Image processing pipeline
            imagescaletototalpixels_299 = self.imagescaletototalpixels.upscale(
                upscale_method="nearest-exact",
                megapixels=4,
                image=get_value_at_index(loadimage_301, 0),
            )

            easy_humansegmentation_292 = self.easy_humansegmentation.parsing(
                method="selfie_multiclass_256x256",
                confidence=0.4,
                crop_multi=0,
                mask_components=[3],
                image=get_value_at_index(imagescaletototalpixels_299, 0),
            )

            mask_crop_region_293 = self.mask_crop_region.mask_crop_region(
                padding=50,
                region_type="dominant",
                mask=get_value_at_index(easy_humansegmentation_292, 1),
            )

            image_crop_location_300 = self.image_crop_location.image_crop_location(
                top=get_value_at_index(mask_crop_region_293, 2),
                left=get_value_at_index(mask_crop_region_293, 3),
                right=get_value_at_index(mask_crop_region_293, 4),
                bottom=get_value_at_index(mask_crop_region_293, 5),
                image=get_value_at_index(imagescaletototalpixels_299, 0),
            )

            imageupscalewithmodel_272 = self.imageupscalewithmodel.upscale(
                upscale_model=get_value_at_index(self.upscale_model_loader_273, 0),
                image=get_value_at_index(image_crop_location_300, 0),
            )

            imagescaleby_271 = self.imagescaleby.upscale(
                upscale_method="nearest-exact",
                scale_by=1.5,
                image=get_value_at_index(imageupscalewithmodel_272, 0),
            )

            imageresize_259 = self.imageresize.execute(
                width=1536,
                height=1536,
                interpolation="lanczos",
                method="keep proportion",
                condition="always",
                multiple_of=16,
                image=get_value_at_index(imagescaleby_271, 0),
            )

            supir_encode_248 = self.supir_encode.encode(
                use_tiled_vae=True,
                encoder_tile_size=512,
                encoder_dtype="auto",
                SUPIR_VAE=get_value_at_index(self.supir_model_loader_v2_262, 1),
                image=get_value_at_index(imageresize_259, 0),
            )

            # SUPIR processing
            cliptextencode_396 = self.cliptextencode.encode(
                text="", clip=get_value_at_index(self.checkpointloadersimple_183, 1)
            )

            supir_conditioner_263 = self.supir_conditioner.condition(
                positive_prompt="high quality, detailed, photograph of a person",
                negative_prompt="bad quality, blurry, messy",
                SUPIR_model=get_value_at_index(self.supir_model_loader_v2_262, 0),
                latents=get_value_at_index(supir_encode_248, 0),
            )

            supir_sample_257 = self.supir_sample.sample(
                seed=seed,
                steps=10,
                cfg_scale_start=2,
                cfg_scale_end=1.5,
                EDM_s_churn=5,
                s_noise=1.003,
                DPMPP_eta=1,
                control_scale_start=1,
                control_scale_end=0.9,
                restore_cfg=1,
                keep_model_loaded=False,
                sampler="RestoreDPMPP2MSampler",
                sampler_tile_size=1024,
                sampler_tile_stride=512,
                SUPIR_model=get_value_at_index(self.supir_model_loader_v2_262, 0),
                latents=get_value_at_index(supir_encode_248, 0),
                positive=get_value_at_index(supir_conditioner_263, 0),
                negative=get_value_at_index(supir_conditioner_263, 1),
            )

            supir_decode_254 = self.supir_decode.decode(
                use_tiled_vae=True,
                decoder_tile_size=512,
                SUPIR_VAE=get_value_at_index(self.supir_model_loader_v2_262, 1),
                latents=get_value_at_index(supir_sample_257, 0),
            )

            colormatch_253 = self.colormatch.colormatch(
                method="mkl",
                strength=1,
                image_ref=get_value_at_index(image_crop_location_300, 0),
                image_target=get_value_at_index(supir_decode_254, 0),
            )

            easy_cleangpuused_395 = self.easy_cleangpuused.empty_cache(
                anything=get_value_at_index(colormatch_253, 0), unique_id=108042808736749469
            )

            # VAE encoding and PulidFlux processing
            vaeencodeadvanced_91 = self.vaeencodeadvanced.main(
                resize_to_input="false",
                width=1024,
                height=1024,
                mask_channel="red",
                invert_mask=False,
                latent_type="16_channels",
                image_1=get_value_at_index(easy_cleangpuused_395, 0),
                image_2=get_value_at_index(easy_cleangpuused_395, 0),
                vae=get_value_at_index(self.checkpointloadersimple_183, 2),
            )

            applypulidflux_112 = self.applypulidflux.apply_pulid_flux(
                weight=0.4,
                start_at=0.2,
                end_at=0.8,
                model=get_value_at_index(self.fluxloader_98, 0),
                pulid_flux=get_value_at_index(self.pulidfluxmodelloader_108, 0),
                eva_clip=get_value_at_index(self.pulidfluxevacliploader_139, 0),
                face_analysis=get_value_at_index(self.pulidfluxinsightfaceloader_109, 0),
                image=get_value_at_index(image_crop_location_300, 0),
                unique_id=2664753588248060434,
            )

            # Load LoRA models for Flux
            self._load_lora_models(applypulidflux_112, None)

            vram_debug_104 = self.vram_debug.VRAMdebug(
                empty_cache=True,
                gc_collect=True,
                unload_all_models=True,
                any_input=get_value_at_index(vaeencodeadvanced_91, 0),
                model_pass=get_value_at_index(self.loraloadermodelonly_439, 0),
            )

            # Control net processing
            fluxguidance_241 = self.fluxguidance.append(
                guidance=3.5, conditioning=get_value_at_index(cliptextencode_188, 0)
            )

            conditioningzeroout_237 = self.conditioningzeroout.zero_out(
                conditioning=get_value_at_index(cliptextencode_188, 0)
            )

            setshakkerlabsunioncontrolnettype_311 = (
                self.setshakkerlabsunioncontrolnettype.set_controlnet_type(
                    type="depth", control_net=get_value_at_index(self.controlnetloader_312, 0)
                )
            )

            midas_depthmappreprocessor_394 = self.midas_depthmappreprocessor.execute(
                a=6.28,
                bg_threshold=0.1,
                resolution=512,
                image=get_value_at_index(image_crop_location_300, 0),
            )

            facedepthmapnode_400 = self.facedepthmapnode.generate_face_depth_map(
                image=get_value_at_index(image_crop_location_300, 0)
            )

            image_overlay_284 = self.image_overlay.apply_overlay_image(
                overlay_resize="Fit",
                resize_method="nearest-exact",
                rescale_factor=1,
                width=512,
                height=512,
                x_offset=0,
                y_offset=0,
                rotation=9.992007221626409e-15,
                opacity=50,
                base_image=get_value_at_index(midas_depthmappreprocessor_394, 0),
                overlay_image=get_value_at_index(facedepthmapnode_400, 0),
            )

            controlnetapplyadvanced_313 = self.controlnetapplyadvanced.apply_controlnet(
                strength=0.8,
                start_percent=0,
                end_percent=0.2,
                positive=get_value_at_index(fluxguidance_241, 0),
                negative=get_value_at_index(conditioningzeroout_237, 0),
                control_net=get_value_at_index(setshakkerlabsunioncontrolnettype_311, 0),
                image=get_value_at_index(image_overlay_284, 0),
                vae=get_value_at_index(self.checkpointloadersimple_183, 2),
            )

            # Continue with canny control net
            setshakkerlabsunioncontrolnettype_318 = (
                self.setshakkerlabsunioncontrolnettype.set_controlnet_type(
                    type="canny", control_net=get_value_at_index(self.controlnetloader_312, 0)
                )
            )

            easy_humansegmentation_286 = self.easy_humansegmentation.parsing(
                method="selfie_multiclass_256x256",
                confidence=0.4,
                crop_multi=0,
                mask_components=[3],
                image=get_value_at_index(image_crop_location_300, 0),
            )

            lineartpreprocessor_283 = self.lineartpreprocessor.execute(
                coarse="disable",
                resolution=512,
                image=get_value_at_index(easy_humansegmentation_286, 0),
            )

            controlnetapplyadvanced_319 = self.controlnetapplyadvanced.apply_controlnet(
                strength=0.6,
                start_percent=0,
                end_percent=0.4,
                positive=get_value_at_index(controlnetapplyadvanced_313, 0),
                negative=get_value_at_index(controlnetapplyadvanced_313, 1),
                control_net=get_value_at_index(setshakkerlabsunioncontrolnettype_318, 0),
                image=get_value_at_index(lineartpreprocessor_283, 0),
                vae=get_value_at_index(self.checkpointloadersimple_183, 2),
            )

            clipvisionstyleloader_399 = self.clipvisionstyleloader.process_image(
                clip_vision="sigclip_vision_patch14_384.safetensors",
                style_model="flux1-redux-dev.safetensors",
                crop_method="none",
                image=get_value_at_index(image_crop_location_300, 0),
            )

            reduxfinetune_397 = self.reduxfinetune.apply_style(
                fusion_mode="Mix",
                fusion_strength=1,
                SUPER_REDUX=False,
                conditioning=get_value_at_index(controlnetapplyadvanced_319, 0),
                style_model=get_value_at_index(self.stylemodelloader_391, 0),
                clip_vision_output=get_value_at_index(clipvisionstyleloader_399, 2),
            )

            # Sampling process
            clownsharksampler_beta_135 = self.clownsharksampler_beta.main(
                eta=0.5,
                sampler_name="multistep/res_2m",
                scheduler="beta57",
                steps=30,
                steps_to_run=15,
                denoise=1,
                cfg=1,
                seed=seed,
                sampler_mode="unsample",
                bongmath=True,
                model=get_value_at_index(self.fluxloader_98, 0),
                positive=get_value_at_index(cliptextencode_396, 0),
                negative=get_value_at_index(cliptextencode_396, 0),
                latent_image=get_value_at_index(vram_debug_104, 0),
            )

            clownsharkchainsampler_beta_239 = self.clownsharkchainsampler_beta.main(
                eta=0.5,
                sampler_name="multistep/res_2m",
                steps_to_run=1,
                cfg=1,
                sampler_mode="resample",
                bongmath=True,
                positive=get_value_at_index(reduxfinetune_397, 0),
                negative=get_value_at_index(cliptextencode_396, 0),
                latent_image=get_value_at_index(clownsharksampler_beta_135, 0),
                options=get_value_at_index(self.clownoptions_cycles_beta_238, 0),
            )

            clownguide_style_beta_80 = self.clownguide_style_beta.main(
                apply_to="positive",
                method="WCT",
                weight=1,
                synweight=1,
                weight_scheduler="constant",
                start_step=0,
                end_step=-1,
                invert_mask=False,
                guide=get_value_at_index(vaeencodeadvanced_91, 1),
            )

            clownsharkchainsampler_beta_240 = self.clownsharkchainsampler_beta.main(
                eta=0.5,
                sampler_name="multistep/res_2m",
                steps_to_run=-1,
                cfg=1,
                sampler_mode="resample",
                bongmath=True,
                model=get_value_at_index(vram_debug_104, 2),
                positive=get_value_at_index(reduxfinetune_397, 0),
                negative=get_value_at_index(cliptextencode_396, 0),
                latent_image=get_value_at_index(clownsharkchainsampler_beta_239, 0),
                guides=get_value_at_index(clownguide_style_beta_80, 0),
            )

            vaedecode_94 = self.vaedecode.decode(
                samples=get_value_at_index(clownsharkchainsampler_beta_240, 0),
                vae=get_value_at_index(self.checkpointloadersimple_183, 2),
            )

            # Post-processing pipeline
            getimagesize_402 = self.getimagesize.execute(
                image=get_value_at_index(vaedecode_94, 0)
            )

            imageresizekjv2_401 = self.imageresizekjv2.resize(
                width=get_value_at_index(getimagesize_402, 0),
                height=get_value_at_index(getimagesize_402, 1),
                upscale_method="nearest-exact",
                keep_proportion="stretch",
                pad_color="0, 0, 0",
                crop_position="center",
                divisible_by=2,
                device="cpu",
                unique_id=548787987,
                image=get_value_at_index(easy_cleangpuused_395, 0),
            )

            easy_humansegmentation_383 = self.easy_humansegmentation.parsing(
                method="selfie_multiclass_256x256",
                confidence=0.4,
                crop_multi=0,
                mask_components=[3],
                image=get_value_at_index(imageresizekjv2_401, 0),
            )

            growmaskwithblur_384 = self.growmaskwithblur.expand_mask(
                expand=15,
                incremental_expandrate=0,
                tapered_corners=True,
                flip_input=False,
                blur_radius=5,
                lerp_alpha=1,
                decay_factor=1,
                fill_holes=False,
                mask=get_value_at_index(easy_humansegmentation_383, 1),
            )

            facealignwarpnode_385 = self.facealignwarpnode.run(
                original_image=get_value_at_index(imageresizekjv2_401, 0),
                donor_image=get_value_at_index(vaedecode_94, 0),
                donor_mask=get_value_at_index(growmaskwithblur_384, 0),
            )

            basicskinblender_387 = self.basicskinblender.run(
                original_image=get_value_at_index(imageresizekjv2_401, 0),
                enhanced_image=get_value_at_index(facealignwarpnode_385, 0),
                mask=get_value_at_index(facealignwarpnode_385, 1),
            )

            # Face segmentation and final processing
            face_segment_params = {
                "Skin": False,
                "Nose": False,
                "Eyeglasses": False,
                "Left-eye": True,
                "Right-eye": True,
                "Left-eyebrow": False,
                "Right-eyebrow": False,
                "Left-ear": False,
                "Right-ear": False,
                "Mouth": True,
                "Upper-lip": True,
                "Lower-lip": True,
                "Hair": False,
                "Earring": False,
                "Neck": False,
            }
            
            facesegment_344 = self.facesegment.segment_face(
                process_res=512,
                mask_blur=0,
                mask_offset=0,
                invert_output=False,
                background="Alpha",
                background_color="#222222",
                images=get_value_at_index(basicskinblender_387, 0),
                **face_segment_params
            )

            growmaskwithblur_355 = self.growmaskwithblur.expand_mask(
                expand=5,
                incremental_expandrate=0,
                tapered_corners=True,
                flip_input=False,
                blur_radius=5,
                lerp_alpha=1,
                decay_factor=1,
                fill_holes=False,
                mask=get_value_at_index(facesegment_344, 1),
            )

            layerutility_imageblend_348 = self.layerutility_imageblend.image_blend(
                invert_mask=False,
                blend_mode="normal",
                opacity=100,
                background_image=get_value_at_index(basicskinblender_387, 0),
                layer_image=get_value_at_index(imageresizekjv2_401, 0),
                layer_mask=get_value_at_index(growmaskwithblur_355, 0),
            )

            # Load LoRA models for SDXL
            self._load_lora_models(None, self.checkpointloadersimple_409)

            vaeencode_407 = self.vaeencode.encode(
                pixels=get_value_at_index(layerutility_imageblend_348, 0),
                vae=get_value_at_index(self.checkpointloadersimple_409, 2),
            )

            cliptextencode_419 = self.cliptextencode.encode(
                text="textured skin, pores, blemishes, realistic detailed face",
                clip=get_value_at_index(self.checkpointloadersimple_409, 1),
            )

            cliptextencode_420 = self.cliptextencode.encode(
                text="cartoon, smooth",
                clip=get_value_at_index(self.checkpointloadersimple_409, 1),
            )

            mxslider_441 = self.mxslider.main(Xi=0, Xf=slider_value, isfloatX=1)

            # Image comparison and final sampling
            image_comparer_rgthree_101 = self.image_comparer_rgthree.compare_images(
                image_a=get_value_at_index(image_crop_location_300, 0),
                image_b=get_value_at_index(vaedecode_94, 0),
            )

            image_comparer_rgthree_255 = self.image_comparer_rgthree.compare_images(
                image_a=get_value_at_index(image_crop_location_300, 0),
                image_b=get_value_at_index(colormatch_253, 0),
            )

            mathexpressionpysssss_431 = self.mathexpressionpysssss.evaluate(
                expression="a*2", a=get_value_at_index(mxslider_441, 0), prompt="PROMPT"
            )

            mathexpressionpysssss_422 = self.mathexpressionpysssss.evaluate(
                expression="a / 2", a=get_value_at_index(mxslider_441, 0), prompt="PROMPT"
            )

            mathexpressionpysssss_431 = get_value_at_index(mxslider_441, 0)*2
            mathexpressionpysssss_422 = get_value_at_index(mxslider_441, 0)/2
            print('mathexp431: ',mathexpressionpysssss_431)
            print('mathexp422: ',mathexpressionpysssss_422)
            ksampler_413 = self.ksampler.sample(
                seed=seed,
                steps=40,
                cfg=5.5,
                sampler_name="res_2m",
                scheduler="beta57",
                denoise=mathexpressionpysssss_422,
                model=get_value_at_index(self.loraloadermodelonly_440, 0),
                positive=get_value_at_index(cliptextencode_419, 0),
                negative=get_value_at_index(cliptextencode_420, 0),
                latent_image=get_value_at_index(vaeencode_407, 0),
            )

            vaedecode_412 = self.vaedecode.decode(
                samples=get_value_at_index(ksampler_413, 0),
                vae=get_value_at_index(self.checkpointloadersimple_409, 2),
            )

            easy_humansegmentation_428 = self.easy_humansegmentation.parsing(
                method="selfie_multiclass_256x256",
                confidence=0.4,
                crop_multi=0,
                mask_components=[3],
                image=get_value_at_index(layerutility_imageblend_348, 0),
            )

            masktoimage_435 = self.masktoimage.mask_to_image(
                mask=get_value_at_index(easy_humansegmentation_428, 1)
            )

            image_blend_by_mask_434 = self.image_blend_by_mask.image_blend_mask(
                blend_percentage=mathexpressionpysssss_431,
                image_a=get_value_at_index(layerutility_imageblend_348, 0),
                image_b=get_value_at_index(vaedecode_412, 0),
                mask=get_value_at_index(masktoimage_435, 0),
            )

            image_paste_crop_305 = self.image_paste_crop.image_paste_crop(
                crop_blending=0.1,
                crop_sharpening=0,
                image=get_value_at_index(imagescaletototalpixels_299, 0),
                crop_image=get_value_at_index(image_blend_by_mask_434, 0),
                crop_data=get_value_at_index(image_crop_location_300, 1),
            )

            image_comparer_rgthree_357 = self.image_comparer_rgthree.compare_images(
                image_a=get_value_at_index(imageresizekjv2_401, 0),
                image_b=get_value_at_index(layerutility_imageblend_348, 0),
            )

            for res in image_paste_crop_305[0]:
                img = Image.fromarray(np.clip(255. * res.detach().cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
            
            # Clear VRAM cache and garbage collect
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time
            print(f"Inference completed in {inference_time:.2f} seconds")
            return img


if __name__ == "__main__":
    workflow = DetailerWorkflow()
    result = workflow()
    result.save('result.png')
    print("Workflow completed successfully")

