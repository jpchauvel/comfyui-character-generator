import argparse
import asyncio
import os
import pathlib
import random
import sys
from typing import Any, Mapping, Sequence, Union

from comfyui_character_generator.util.manager import AppManager


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


def add_comfyui_directory_to_sys_path(
    comfyui_path: pathlib.Path | None,
) -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    if comfyui_path is None:
        raise ValueError("comfyui_path is None")
    if os.path.isdir(comfyui_path):
        sys.path.append(str(comfyui_path))
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths(comfyui_path: pathlib.Path | None) -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    if comfyui_path is None:
        raise ValueError("comfyui_path is None")
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = comfyui_path / "extra_model_paths.yaml"

    if os.path.isfile(extra_model_paths):
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio

    import execution
    import server
    from nodes import init_extra_nodes

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    loop = asyncio.get_event_loop()
    loop.run_until_complete(init_extra_nodes())


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Automated ComfyUI Character Generator.",
    )
    parser.add_argument(
        "--config_idx",
        type=int,
        required=True,
        help="Config index.",
    )
    parser.add_argument(
        "--prompt_idx",
        type=int,
        required=True,
        help="Prompt index.",
    )
    return parser.parse_args()


def main() -> None:
    manager: AppManager = AppManager(sys.stdin.read())
    args: argparse.Namespace = get_args()

    import torch

    add_comfyui_directory_to_sys_path(manager.config.comfyui_path)
    add_extra_model_paths(manager.config.comfyui_path)

    from nodes import NODE_CLASS_MAPPINGS

    def generate() -> None:
        import_custom_nodes()
        with torch.inference_mode():
            emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
            emptylatentimage_5 = emptylatentimage.generate(
                width=manager.config.sub_configs[args.config_idx].width,
                height=manager.config.sub_configs[args.config_idx].height,
                batch_size=manager.config.sub_configs[args.config_idx].batch,
            )

            checkpointloadersimple = NODE_CLASS_MAPPINGS[
                "CheckpointLoaderSimple"
            ]()
            checkpointloadersimple_12 = checkpointloadersimple.load_checkpoint(
                ckpt_name=manager.config.sub_configs[args.config_idx].ckpt,
            )

            loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
            loraloaders: list[Any] = []
            for idx, lora in enumerate(
                manager.config.sub_configs[args.config_idx].loras
            ):
                if idx == 0:
                    loraloaders.append(
                        loraloader.load_lora(
                            lora_name=lora,
                            strength_model=manager.config.sub_configs[
                                args.config_idx
                            ].lora_strengths[idx],
                            strength_clip=manager.config.sub_configs[
                                args.config_idx
                            ].lora_strengths[idx],
                            model=get_value_at_index(
                                checkpointloadersimple_12, 0
                            ),
                            clip=get_value_at_index(
                                checkpointloadersimple_12, 1
                            ),
                        )
                    )
                else:
                    loraloaders.append(
                        loraloader.load_lora(
                            lora_name=lora,
                            strength_model=manager.config.sub_configs[
                                args.config_idx
                            ].lora_strengths[idx],
                            strength_clip=manager.config.sub_configs[
                                args.config_idx
                            ].lora_strengths[idx],
                            model=get_value_at_index(loraloaders[idx - 1], 0),
                            clip=get_value_at_index(loraloaders[idx - 1], 1),
                        )
                    )

            last_lora: Any = loraloaders[-1]

            cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            cliptextencode_6 = cliptextencode.encode(
                text=manager.config.sub_configs[args.config_idx].system_prompt,
                clip=get_value_at_index(last_lora, 1),
            )

            cliptextencode_15 = cliptextencode.encode(
                text=manager.config.sub_configs[args.config_idx].sub_prompts[
                    args.prompt_idx
                ],
                clip=get_value_at_index(last_lora, 1),
            )

            cliptextencode_7 = cliptextencode.encode(
                text=manager.config.sub_configs[
                    args.config_idx
                ].system_neg_prompt,
                clip=get_value_at_index(last_lora, 1),
            )

            cliptextencode_31 = cliptextencode.encode(
                text=manager.config.sub_configs[args.config_idx].neg_prompts[
                    args.prompt_idx
                ],
                clip=get_value_at_index(last_lora, 1),
            )

            controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
            controlnetloader_34 = controlnetloader.load_controlnet(
                control_net_name=manager.config.sub_configs[
                    args.config_idx
                ].controlnet,
            )

            loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
            loadimage_51 = loadimage.load_image(
                image=manager.config.sub_configs[args.config_idx].pose_images[
                    args.prompt_idx
                ]
            )

            loadimage_56 = loadimage.load_image(
                image=manager.config.sub_configs[
                    args.config_idx
                ].face_swap_images[args.prompt_idx]
            )

            upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
            upscalemodelloader_76 = upscalemodelloader.load_model(
                model_name=manager.config.sub_configs[args.config_idx].upscaler
            )

            conditioningconcat = NODE_CLASS_MAPPINGS["ConditioningConcat"]()
            openposepreprocessor = NODE_CLASS_MAPPINGS[
                "OpenposePreprocessor"
            ]()
            lineartpreprocessor = NODE_CLASS_MAPPINGS["LineArtPreprocessor"]()
            depthanythingpreprocessor = NODE_CLASS_MAPPINGS[
                "DepthAnythingPreprocessor"
            ]()
            easy_anythingindexswitch = NODE_CLASS_MAPPINGS[
                "easy anythingIndexSwitch"
            ]()
            controlnetapplyadvanced = NODE_CLASS_MAPPINGS[
                "ControlNetApplyAdvanced"
            ]()
            impactswitch = NODE_CLASS_MAPPINGS["ImpactSwitch"]()
            ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            easy_cleangpuused = NODE_CLASS_MAPPINGS["easy cleanGpuUsed"]()
            easy_clearcacheall = NODE_CLASS_MAPPINGS["easy clearCacheAll"]()
            ultimatesdupscale = NODE_CLASS_MAPPINGS["UltimateSDUpscale"]()
            reactorfaceswap = NODE_CLASS_MAPPINGS["ReActorFaceSwap"]()
            saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

            conditioningconcat_18 = conditioningconcat.concat(
                conditioning_to=get_value_at_index(cliptextencode_6, 0),
                conditioning_from=get_value_at_index(cliptextencode_15, 0),
            )

            conditioningconcat_33 = conditioningconcat.concat(
                conditioning_to=get_value_at_index(cliptextencode_7, 0),
                conditioning_from=get_value_at_index(cliptextencode_31, 0),
            )

            openposepreprocessor_66 = openposepreprocessor.estimate_pose(
                detect_hand="enable",
                detect_body="enable",
                detect_face="enable",
                resolution=512,
                scale_stick_for_xinsr_cn="disable",
                image=get_value_at_index(loadimage_51, 0),
            )

            lineartpreprocessor_67 = lineartpreprocessor.execute(
                coarse="disable",
                resolution=512,
                image=get_value_at_index(loadimage_51, 0),
            )

            depthanythingpreprocessor_68 = depthanythingpreprocessor.execute(
                ckpt_name="depth_anything_vitb14.pth",
                resolution=512,
                image=get_value_at_index(loadimage_51, 0),
            )

            easy_anythingindexswitch_71 = (
                easy_anythingindexswitch.index_switch(
                    index=manager.config.sub_configs[
                        args.config_idx
                    ].pose_detection_type.value,
                    value0=get_value_at_index(openposepreprocessor_66, 0),
                    value1=get_value_at_index(lineartpreprocessor_67, 0),
                    value2=get_value_at_index(depthanythingpreprocessor_68, 0),
                )
            )

            controlnetapplyadvanced_35 = (
                controlnetapplyadvanced.apply_controlnet(
                    strength=0.6000000000000001,
                    start_percent=0,
                    end_percent=0.4000000000000001,
                    positive=get_value_at_index(conditioningconcat_18, 0),
                    negative=get_value_at_index(conditioningconcat_33, 0),
                    control_net=get_value_at_index(controlnetloader_34, 0),
                    image=get_value_at_index(easy_anythingindexswitch_71, 0),
                    vae=get_value_at_index(checkpointloadersimple_12, 2),
                )
            )

            impactswitch_62 = impactswitch.doit(
                select=(
                    1
                    if manager.config.sub_configs[
                        args.config_idx
                    ].disable_controlnet
                    else 2
                ),
                sel_mode=False,
                input1=get_value_at_index(conditioningconcat_18, 0),
                input2=get_value_at_index(controlnetapplyadvanced_35, 0),
                unique_id=62,
            )

            impactswitch_63 = impactswitch.doit(
                select=(
                    1
                    if manager.config.sub_configs[
                        args.config_idx
                    ].disable_controlnet
                    else 2
                ),
                sel_mode=False,
                input1=get_value_at_index(conditioningconcat_33, 0),
                input2=get_value_at_index(controlnetapplyadvanced_35, 1),
                unique_id=63,
            )

            ksampler_3 = ksampler.sample(
                seed=manager.config.sub_configs[args.config_idx].seed,
                steps=manager.config.sub_configs[args.config_idx].steps,
                cfg=manager.config.sub_configs[args.config_idx].guidance_scale,
                sampler_name="dpmpp_sde",
                scheduler="karras",
                denoise=1,
                model=get_value_at_index(last_lora, 0),
                positive=get_value_at_index(impactswitch_62, 0),
                negative=get_value_at_index(impactswitch_63, 0),
                latent_image=get_value_at_index(emptylatentimage_5, 0),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(checkpointloadersimple_12, 2),
            )

            easy_cleangpuused_43 = easy_cleangpuused.empty_cache(
                anything=get_value_at_index(last_lora, 0),
                unique_id=10755185619704895890,
            )

            easy_clearcacheall_44 = easy_clearcacheall.empty_cache(
                anything=get_value_at_index(easy_cleangpuused_43, 0),
                unique_id=10390157557927668883,
            )

            easy_cleangpuused_45 = easy_cleangpuused.empty_cache(
                anything=get_value_at_index(vaedecode_8, 0),
                unique_id=5840249241297495656,
            )

            easy_clearcacheall_46 = easy_clearcacheall.empty_cache(
                anything=get_value_at_index(easy_cleangpuused_45, 0),
                unique_id=12735645180315196723,
            )

            ultimatesdupscale_77 = ultimatesdupscale.upscale(
                upscale_by=1.5000000000000002,
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=8,
                sampler_name="euler",
                scheduler="simple",
                denoise=0.20000000000000004,
                mode_type="Linear",
                tile_width=1024,
                tile_height=1024,
                mask_blur=8,
                tile_padding=32,
                seam_fix_mode="None",
                seam_fix_denoise=1,
                seam_fix_width=64,
                seam_fix_mask_blur=8,
                seam_fix_padding=16,
                force_uniform_tiles=True,
                tiled_decode=False,
                image=get_value_at_index(vaedecode_8, 0),
                model=get_value_at_index(last_lora, 0),
                positive=get_value_at_index(impactswitch_62, 0),
                negative=get_value_at_index(impactswitch_63, 0),
                vae=get_value_at_index(checkpointloadersimple_12, 2),
                upscale_model=get_value_at_index(upscalemodelloader_76, 0),
            )

            reactorfaceswap_57 = reactorfaceswap.execute(
                enabled=True,
                swap_model="inswapper_128.onnx",
                facedetection="retinaface_resnet50",
                face_restore_model="GPEN-BFR-512.onnx",
                face_restore_visibility=1,
                codeformer_weight=0.5,
                detect_gender_input="no",
                detect_gender_source="no",
                input_faces_index="0",
                source_faces_index="0",
                console_log_level=1,
                # input_image=get_value_at_index(easy_imagebatchtoimagelist_79, 0),
                input_image=get_value_at_index(ultimatesdupscale_77, 0),
                source_image=get_value_at_index(loadimage_56, 0),
            )

            if manager.config.sub_configs[
                args.config_idx
            ].output_path != pathlib.Path(""):
                saveimage.output_dir = (
                    manager.config.sub_configs[args.config_idx]
                    .output_path.expanduser()
                    .as_posix()
                )
            saveimage_59 = saveimage.save_images(
                filename_prefix="ComfyUI",
                images=get_value_at_index(reactorfaceswap_57, 0),
            )

    generate()


if __name__ == "__main__":
    main()
