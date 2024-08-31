# ---
# output-directory: "/tmp/stable-diffusion-xl"
# args: ["--prompt", "An astronaut riding a green horse"]
# runtimes: ["runc", "gvisor"]
# ---
# # Stable Diffusion XL 1.0
#
# This example is similar to the [Stable Diffusion CLI](/docs/examples/stable_diffusion_cli)
# example, but it generates images from the larger SDXL 1.0 model. Specifically, it runs the
# first set of steps with the base model, followed by the refiner model.
#
# [Try out the live demo here!](https://modal-labs--stable-diffusion-xl-app.modal.run/) The first
# generation may include a cold-start, which takes around 20 seconds. The inference speed depends on the GPU
# and step count (for reference, an A100 runs 40 steps in 8 seconds).

# ## Basic setup

import io
from pathlib import Path
import os
from pathlib import Path
import base64
from typing import List

from modal import (
    Stub,
    Image,
    build,
    enter,
    gpu,
    method,
    web_endpoint,
    asgi_app,

)
from fastapi import Response, Query, FastAPI, Request
from fastapi.responses import JSONResponse,StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# ## Define a container image
#
# To take advantage of Modal's blazing fast cold-start times, we'll need to download our model weights
# inside our container image with a download function. We ignore binaries, ONNX weights and 32-bit weights.
#
# Tip: avoid using global variables in this function to ensure the download step detects model changes and
# triggers a rebuild.


sdxl_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "diffusers==0.26.3",
        "invisible_watermark==0.2.0",
        "transformers~=4.38.2",
        "accelerate==0.27.2",
        "safetensors==0.4.2",
    )
)

stub = Stub("stable-diffusion-xl")  # Note: prior to April 2024, "app" was called "stub"

with sdxl_image.imports():
    import torch
    from diffusers import DiffusionPipeline


# ## Load model and run inference
#
# The container lifecycle [`@enter` decorator](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta)
# loads the model at startup. Then, we evaluate it in the `run_inference` function.
#
# To avoid excessive cold-starts, we set the idle timeout to 240 seconds, meaning once a GPU has loaded the model it will stay
# online for 4 minutes before spinning down. This can be adjusted for cost/experience trade-offs.

web_app = FastAPI()
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@stub.cls(gpu=gpu.A10G(), container_idle_timeout=240, image=sdxl_image)
class Model:
    @build()
    def build(self):
        from huggingface_hub import snapshot_download

        ignore = [
            "*.bin",
            "*.onnx_data",
            "*/diffusion_pytorch_model.safetensors",
        ]
        snapshot_download(
            "stabilityai/stable-diffusion-xl-base-1.0", ignore_patterns=ignore
        )
        snapshot_download(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            ignore_patterns=ignore,
        )

    @enter()
    def enter(self):
        load_options = dict(
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
        )

        # Load base model
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", **load_options
        )

        # Load refiner model
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            **load_options,
        )

    def _inference(self, prompt, body_part="shoulder", n_steps=50, high_noise_frac=0.8):
        negative_prompt = "disfigured, ugly, deformed"
        detailed_prompt = (
            f"A photo-realistic image of a {body_part} with a {prompt} tattoo, "
            "showing the tattoo clearly on the human model's {body_part}. "
            "The human model should be visible, and the tattoo should be intricate, "
            "highly detailed, and visually appealing."
        )

        image = self.base(
            prompt=detailed_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images

        image = self.refiner(
            prompt=detailed_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="JPEG")

        return byte_stream

    @method()
    def inference(self, prompt, body_part="shoulder", n_steps=24, high_noise_frac=0.8):
        return self._inference(
            prompt,
            body_part=body_part,
            n_steps=n_steps,
            high_noise_frac=high_noise_frac,
        ).getvalue()

    @web_endpoint()
    async def web_inference(self, prompt: str, body_part: str = "back", n_steps: int = 24, high_noise_frac: float = 0.8):
        image_bytes = self._inference(prompt, body_part=body_part, n_steps=n_steps, high_noise_frac=high_noise_frac).getvalue()
        image_bytes_list = base64.b64encode(image_bytes).decode('utf-8')
        return image_bytes_list


# FastAPI endpoint
@web_app.post("/tattoo")
async def tattoo(request: Request):
    body = await request.json()
    prompt = body.get("prompt")
    body_part = body.get("body_part", "back")
    n_images = body.get("n_images", 2)

    print("prompt is: ", prompt)
    if not prompt:
        return JSONResponse({"error": "Please provide a prompt"}, status_code=400)
    
    images_base64: List[str] = []

    for _ in range(n_images):
        image_bytes = Model().inference.remote(prompt=prompt, body_part=body_part)
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        images_base64.append(image_base64)
    return JSONResponse({"images": images_base64})

@stub.function(allow_concurrent_inputs=4)
@asgi_app()
def entrypoint():
    return web_app


@stub.local_entrypoint()
def main(
    prompt: str = "Traditional tattoo of a skull with roses", body_part: str = "forearm"
):
    # Create the "images" directory if it doesn't exist
    images_dir = Path(os.getcwd()) / "images"
    images_dir.mkdir(exist_ok=True)

    for i in range(4):  # Increase to 4 images for more options
        image_bytes = Model().inference.remote(prompt=prompt, body_part=body_part)
        print(image_bytes)
        output_path = images_dir / f"output_{i+1}.png"
        print(f"Saving it to {output_path}")
        with open(output_path, "wb") as f:
            f.write(image_bytes)


if __name__ == "__main__":
    main()

web_app = FastAPI()
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import  FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# FastAPI endpoint
@web_app.post("/hello")
async def hello(request: Request):
    body = await request.json()
    prompts = body.get("prompts")
    
    print("prompt is: ", prompts)
    if not prompts:
        return JSONResponse({"error": "Please provide a prompt"}, status_code=400)
    
    # response = Model.generate.remote(prompts, settings=None)
    return JSONResponse({"data": "Hi"})

@app.function(allow_concurrent_inputs=4)
@modal.asgi_app()
def entrypoint():
    return web_app
