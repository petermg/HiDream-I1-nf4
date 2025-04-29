import torch
import gradio as gr
import logging
import os
from datetime import datetime
from .nf4 import *

# Output directory for saving images
OUTPUT_DIR = ".\outputs"

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

# Parse resolution string to get height and width
def parse_resolution(resolution_str):
    return tuple(map(int, resolution_str.split("(")[0].strip().split(" × ")))

def gen_img_helper(model, prompt, res, seed, scheduler, guidance_scale, num_inference_steps, shift):
    global pipe, current_model

    # 1. Check if the model matches loaded model, load the model if not
    if model != current_model:
        print(f"Unloading model {current_model}...")
        if pipe is not None:
            del pipe
            torch.cuda.empty_cache()
        
        print(f"Loading model {model}...")
        pipe, _ = load_models(model)
        current_model = model
        print("Model loaded successfully!")

    # 2. Update scheduler
    config = MODEL_CONFIGS[model]
    scheduler_map = {
        "FlashFlowMatchEulerDiscreteScheduler": FlashFlowMatchEulerDiscreteScheduler,
        "FlowUniPCMultistepScheduler": FlowUniPCMultistepScheduler
    }
    scheduler_class = scheduler_map[scheduler]
    device = pipe._execution_device

    # Set scheduler with shift for flow-matching schedulers
    pipe.scheduler = scheduler_class(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)

    # 3. Generate image
    res = parse_resolution(res)
    image, seed = generate_image(pipe, model, prompt, res, seed, guidance_scale, num_inference_steps)
    
    # 4. Save image locally
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(OUTPUT_DIR, f"output_{timestamp}.png")
    image.save(output_path)
    
    return image, seed, f"Image saved to: {output_path}"

def generate_image(pipe, model_type, prompt, resolution, seed, guidance_scale, num_inference_steps):
    # Parse resolution
    width, height = resolution

    # Handle seed
    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    # Common parameters
    params = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "num_images_per_prompt": 1,
        "generator": generator
    }

    images = pipe(**params).images
    return images[0], seed

if __name__ == "__main__":
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    # Initialize globals without loading model
    current_model = None
    pipe = None

    # Create Gradio interface
    with gr.Blocks(title="HiDream-I1-nf4 Dashboard") as demo:
        gr.Markdown("# HiDream-I1-nf4 Dashboard")
        
        with gr.Row():
            with gr.Column():
                model_type = gr.Radio(
                    choices=list(MODEL_CONFIGS.keys()),
                    value="fast",
                    label="Model Type",
                    info="Select model variant (e.g., 'fast' for quick generation)"
                )
                
                prompt = gr.Textbox(
                    label="Prompt", 
                    placeholder="A cat holding a sign that says \"Hi-Dreams.ai\".", 
                    lines=3
                )
                
                resolution = gr.Radio(
                    choices=RESOLUTION_OPTIONS,
                    value=RESOLUTION_OPTIONS[0],
                    label="Resolution",
                    info="Select image resolution"
                )
                
                seed = gr.Number(
                    label="Seed (use -1 for random)", 
                    value=-1, 
                    precision=0
                )
                
                scheduler = gr.Radio(
                    choices=SCHEDULER_OPTIONS,
                    value="FlashFlowMatchEulerDiscreteScheduler",
                    label="Scheduler",
                    info="Select scheduler type. Flow-matching schedulers are optimized for HiDream, providing stable, high-quality, prompt-relevant images."
                )
                
                guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=2.0,
                    label="Guidance Scale",
                    info="Controls prompt adherence. Use 2.0–5.0; increase to 4.0–5.0 for stronger prompt following."
                )
                
                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=25,
                    label="Number of Inference Steps",
                    info="Controls denoising steps. Use 25–50; increase to 40–50 for sharper images."
                )
                
                shift = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    step=0.1,
                    value=3.0,
                    label="Shift",
                    info="Scheduler shift parameter for flow-matching schedulers. Use 1.0–5.0; 3.0 is a good default."
                )
                
                generate_btn = gr.Button("Generate Image")
                
            with gr.Column():
                output_image = gr.Image(label="Generated Image", type="pil")
                seed_used = gr.Number(label="Seed Used", interactive=False)
                save_path = gr.Textbox(label="Saved Image Path", interactive=False)
        
        generate_btn.click(
            fn=gen_img_helper,
            inputs=[model_type, prompt, resolution, seed, scheduler, guidance_scale, num_inference_steps, shift],
            outputs=[output_image, seed_used, save_path]
        )

    demo.launch(share=True, pwa=True)
