import argparse
import os
import pathlib
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
    init_extra_nodes()


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Automated ComfyUI Character Generator.",
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
                width=manager.config.width,
                height=manager.config.height,
                batch_size=manager.config.batch,
            )

            checkpointloadersimple = NODE_CLASS_MAPPINGS[
                "CheckpointLoaderSimple"
            ]()
            checkpointloadersimple_12 = checkpointloadersimple.load_checkpoint(
                ckpt_name=manager.config.ckpt,
            )

            loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
            loraloaders: list[Any] = []
            for idx, lora in enumerate(manager.config.loras):
                if idx == 0:
                    loraloaders.append(
                        loraloader.load_lora(
                            lora_name=lora,
                            strength_model=manager.config.lora_strengths[idx],
                            strength_clip=manager.config.lora_strengths[idx],
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
                            strength_model=manager.config.lora_strengths[idx],
                            strength_clip=manager.config.lora_strengths[idx],
                            model=get_value_at_index(loraloaders[idx - 1], 0),
                            clip=get_value_at_index(loraloaders[idx - 1], 1),
                        )
                    )

            last_lora: Any = loraloaders[-1]

            cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            cliptextencode_6 = cliptextencode.encode(
                text=manager.config.system_prompt,
                clip=get_value_at_index(last_lora, 1),
            )

            cliptextencode_15 = cliptextencode.encode(
                text=manager.config.sub_prompts[args.prompt_idx],
                clip=get_value_at_index(last_lora, 1),
            )

            cliptextencode_7 = cliptextencode.encode(
                text=manager.config.system_neg_prompt,
                clip=get_value_at_index(last_lora, 1),
            )

            cliptextencode_31 = cliptextencode.encode(
                text=manager.config.neg_prompts[args.prompt_idx],
                clip=get_value_at_index(last_lora, 1),
            )

            controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
            controlnetloader_34 = controlnetloader.load_controlnet(
                control_net_name=manager.config.controlnet,
            )

            loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
            loadimage_51 = loadimage.load_image(
                image=manager.config.pose_images[args.prompt_idx]
            )

            loadimage_56 = loadimage.load_image(
                image=manager.config.face_swap_images[args.prompt_idx]
            )

            conditioningconcat = NODE_CLASS_MAPPINGS["ConditioningConcat"]()
            dwpreprocessor = NODE_CLASS_MAPPINGS["DWPreprocessor"]()
            controlnetapplyadvanced = NODE_CLASS_MAPPINGS[
                "ControlNetApplyAdvanced"
            ]()
            impactswitch = NODE_CLASS_MAPPINGS["ImpactSwitch"]()
            ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            easy_cleangpuused = NODE_CLASS_MAPPINGS["easy cleanGpuUsed"]()
            easy_clearcacheall = NODE_CLASS_MAPPINGS["easy clearCacheAll"]()
            reactorfaceswap = NODE_CLASS_MAPPINGS["ReActorFaceSwap"]()
            image_comparer_rgthree = NODE_CLASS_MAPPINGS[
                "Image Comparer (rgthree)"
            ]()
            saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

            conditioningconcat_18 = conditioningconcat.concat(
                conditioning_to=get_value_at_index(cliptextencode_6, 0),
                conditioning_from=get_value_at_index(cliptextencode_15, 0),
            )

            conditioningconcat_33 = conditioningconcat.concat(
                conditioning_to=get_value_at_index(cliptextencode_7, 0),
                conditioning_from=get_value_at_index(cliptextencode_31, 0),
            )

            dwpreprocessor_65 = dwpreprocessor.estimate_pose(
                detect_hand="disable",
                detect_body="enable",
                detect_face="enable",
                resolution=512,
                bbox_detector="yolox_l.onnx",
                pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt",
                scale_stick_for_xinsr_cn="disable",
                image=get_value_at_index(loadimage_51, 0),
            )

            controlnetapplyadvanced_35 = (
                controlnetapplyadvanced.apply_controlnet(
                    strength=0.6000000000000001,
                    start_percent=0,
                    end_percent=0.4000000000000001,
                    positive=get_value_at_index(conditioningconcat_18, 0),
                    negative=get_value_at_index(conditioningconcat_33, 0),
                    control_net=get_value_at_index(controlnetloader_34, 0),
                    image=get_value_at_index(dwpreprocessor_65, 0),
                    vae=get_value_at_index(checkpointloadersimple_12, 2),
                )
            )

            impactswitch_62 = impactswitch.doit(
                select=1 if manager.config.disable_controlnet else 2,
                sel_mode=False,
                input1=get_value_at_index(conditioningconcat_18, 0),
                input2=get_value_at_index(controlnetapplyadvanced_35, 0),
                unique_id=62,
            )

            impactswitch_63 = impactswitch.doit(
                select=1 if manager.config.disable_controlnet else 2,
                sel_mode=False,
                input1=get_value_at_index(conditioningconcat_33, 0),
                input2=get_value_at_index(controlnetapplyadvanced_35, 1),
                unique_id=63,
            )

            ksampler_3 = ksampler.sample(
                seed=manager.config.seed,
                steps=manager.config.steps,
                cfg=manager.config.guidance_scale,
                sampler_name="euler",
                scheduler="normal",
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
                input_image=get_value_at_index(vaedecode_8, 0),
                source_image=get_value_at_index(loadimage_56, 0),
            )

            image_comparer_rgthree_58 = image_comparer_rgthree.compare_images(
                image_a=get_value_at_index(vaedecode_8, 0),
                image_b=get_value_at_index(reactorfaceswap_57, 0),
            )

            if manager.config.output_path != pathlib.Path(""):
                saveimage.output_dir = (
                    manager.config.output_path.expanduser().as_posix()
                )
            saveimage_59 = saveimage.save_images(
                filename_prefix="ComfyUI",
                images=get_value_at_index(reactorfaceswap_57, 0),
            )

    generate()


if __name__ == "__main__":
    main()
