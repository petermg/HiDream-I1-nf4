import torch
import gradio as gr
import logging
import os
import tempfile
import glob
from datetime import datetime
from PIL import Image
from .nf4 import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directory for saving images
OUTPUT_DIR = os.path.join("outputs")

# Resolution options
RESOLUTION_OPTIONS = [
    "1024 × 1024 (Square)",
    "768 × 1360 (Portrait)",
    "1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)",
    "1168 × 880 (Landscape)",
    "1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]

# Scheduler options (flow-matching only)
SCHEDULER_OPTIONS = [
    "FlashFlowMatchEulerDiscreteScheduler",
    "FlowUniPCMultistepScheduler"
]

# Image format options
IMAGE_FORMAT_OPTIONS = ["PNG", "JPEG", "WEBP"]

# Parse resolution string to get height and width
def parse_resolution(resolution_str):
    try:
        return tuple(map(int, resolution_str.split("(")[0].strip().split(" × ")))
    except (ValueError, IndexError) as e:
        raise ValueError("Invalid resolution format") from e

def clean_previous_temp_files():
    temp_dir = tempfile.gettempdir()
    patterns = [os.path.join(temp_dir, f"hdi1_*.{ext}") for ext in ["png", "jpeg", "webp"]]
    deleted_files = []

    for pattern in patterns:
        for temp_file in glob.glob(pattern):
            try:
                os.remove(temp_file)
                deleted_files.append(temp_file)
                logger.info(f"Deleted temporary file: {temp_file}")
            except OSError as e:
                logger.warning(f"Failed to delete temporary file {temp_file}: {str(e)}")

    gradio_temp_dir = os.path.join(temp_dir, "gradio")
    if os.path.exists(gradio_temp_dir):
        for root, _, files in os.walk(gradio_temp_dir):
            for file in files:
                if file.endswith((".png", ".jpeg", ".webp")):
                    gradio_file = os.path.join(root, file)
                    logger.info(f"Found Gradio temporary file: {gradio_file}")

    return deleted_files

def clean_all_temp_files():
    status_message = "Starting temporary file cleanup..."
    logger.info(status_message)

    try:
        deleted_files = clean_previous_temp_files()

        temp_dir = tempfile.gettempdir()
        gradio_temp_dir = os.path.join(temp_dir, "gradio")
        if os.path.exists(gradio_temp_dir):
            for root, _, files in os.walk(gradio_temp_dir):
                for file in files:
                    if file.endswith((".png", ".jpeg", ".webp")):
                        gradio_file = os.path.join(root, file)
                        try:
                            os.remove(gradio_file)
                            deleted_files.append(gradio_file)
                            logger.info(f"Deleted Gradio temporary file: {gradio_file}")
                        except OSError as e:
                            logger.warning(f"Failed to delete Gradio temporary file {gradio_file}: {str(e)}")

        status_message = f"Cleanup complete. Deleted {len(deleted_files)} files."
        logger.info(status_message)
        return status_message
    except Exception as e:
        error_message = f"Cleanup error: {str(e)}"
        logger.error(error_message)
        return error_message

def generate_single_image(pipe, model_type, prompt, resolution, seed, guidance_scale, num_inference_steps):
    try:
        width, height = resolution

        if seed == -1:
            seed = torch.randint(0, 1000000, (1,)).item()

        generator = torch.Generator("cuda").manual_seed(seed)

        params = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator
        }

        image = pipe(**params).images[0]
        return image, seed
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {str(e)}") from e

def gen_img_helper(model, prompt, res, seed, scheduler, guidance_scale, num_inference_steps, shift, image_format, num_images):
    global pipe, current_model
    try:
        clean_previous_temp_files()

        if not prompt or len(prompt.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        if not isinstance(seed, (int, float)) or seed < -1:
            raise ValueError("Seed must be -1 or a non-negative integer")

        if model != current_model:
            if pipe is not None:
                del pipe
                torch.cuda.empty_cache()
            pipe, _ = load_models(model)
            current_model = model

        scheduler_map = {
            "FlashFlowMatchEulerDiscreteScheduler": FlashFlowMatchEulerDiscreteScheduler,
            "FlowUniPCMultistepScheduler": FlowUniPCMultistepScheduler
        }
        scheduler_class = scheduler_map[scheduler]
        pipe.scheduler = scheduler_class(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)

        res = parse_resolution(res)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_extension = image_format.lower()
        images = []
        image_paths = []
        temp_paths = []

        for i in range(int(num_images)):
            img, used_seed = generate_single_image(pipe, model, prompt, res, seed if i == 0 else -1, guidance_scale, num_inference_steps)
            images.append(img)

            fname = f"output_{timestamp}_{i+1}.{file_extension}"
            path = os.path.join(OUTPUT_DIR, fname)
            if image_format == "JPEG":
                img = img.convert("RGB")
            img.save(path, format=image_format)
            image_paths.append(path)

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}", prefix="hdi1_") as temp_file:
                img.save(temp_file, format=image_format)
                temp_paths.append(temp_file.name)

        return images, used_seed, f"Saved {len(images)} images to: {OUTPUT_DIR}", temp_paths[0], f"Generated {len(images)} image(s)."

    except Exception as e:
        error_message = f"Error: {str(e)}"
        logger.error(error_message)
        return None, None, None, None, error_message

if __name__ == "__main__":
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    current_model = None
    pipe = None

    with gr.Blocks(title="HiDream-I1-nf4 Dashboard") as demo:
        gr.Markdown("# HiDream-I1-nf4 Dashboard")

        with gr.Row():
            with gr.Column():
                model_type = gr.Radio(choices=list(MODEL_CONFIGS.keys()), value="fast", label="Model Type")
                prompt = gr.Textbox(label="Prompt", lines=3)
                resolution = gr.Radio(choices=RESOLUTION_OPTIONS, value=RESOLUTION_OPTIONS[0], label="Resolution")
                seed = gr.Number(label="Seed (use -1 for random)", value=-1, precision=0)
                scheduler = gr.Radio(choices=SCHEDULER_OPTIONS, value="FlashFlowMatchEulerDiscreteScheduler", label="Scheduler")
                guidance_scale = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=2.0, label="Guidance Scale")
                num_inference_steps = gr.Slider(minimum=1, maximum=100, step=1, value=25, label="Number of Inference Steps")
                shift = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=3.0, label="Shift")
                image_format = gr.Radio(choices=IMAGE_FORMAT_OPTIONS, value="PNG", label="Image Format")
                num_images = gr.Number(label="Number of Images", value=1, precision=0)
                generate_btn = gr.Button("Generate Image(s)")
                cleanup_btn = gr.Button("Clean Temporary Files")

            with gr.Column():
                status_message = gr.Textbox(label="Status", value="Ready", interactive=False)
                output_image = gr.Gallery(label="Generated Image(s)", columns=2, height="auto")
                seed_used = gr.Number(label="Seed Used", interactive=False)
                save_path = gr.Textbox(label="Saved Image Path", interactive=False)
                download_file = gr.File(label="Download First Image", interactive=False, file_types=[".png", ".jpeg", ".webp"])

        generate_btn.click(
            fn=gen_img_helper,
            inputs=[model_type, prompt, resolution, seed, scheduler, guidance_scale, num_inference_steps, shift, image_format, num_images],
            outputs=[output_image, seed_used, save_path, download_file, status_message]
        )

        cleanup_btn.click(fn=clean_all_temp_files, inputs=[], outputs=[status_message])

    demo.launch(share=True, pwa=True)
