import torch
import spaces
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
import ipown
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
import gradio as gr
import cv2

#base_model_path = "SG161222/RealVisXL_V3.0"
#base_model_path = "cagliostrolab/animagine-xl-3.0"
#base_model_path = "playgroundai/playground-v2-1024px-aesthetic"
base_model_path = "frankjoshua/juggernautXL_v8Rundiffusion"  

try: from pip._internal.operations import freeze
except ImportError: # pip < 10.0
    from pip.operations import freeze

pkgs = freeze.freeze()
for pkg in pkgs: print(pkg)

ip_ckpt = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid_sdxl.bin", repo_type="model")
device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
# vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    add_watermarker=False,
    use_safetensors=True,
    variant="fp16"
    # vae=vae,
    #feature_extractor=safety_feature_extractor,
    #safety_checker=safety_checker
)

ip_model = ipown.IPAdapterFaceIDXL(pipe, ip_ckpt, device)

@spaces.GPU(enable_queue=True)
def generate_image(images, prompt, negative_prompt, face_strength, likeness_strength, progress=gr.Progress(track_tqdm=True)):
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Start the process
    pipe.to(device)
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(512, 512))
    
    faceid_all_embeds = []
    for image in images:
        face = cv2.imread(image)
        faces = app.get(face)
        faceid_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        faceid_all_embeds.append(faceid_embed)

    average_embedding = torch.mean(torch.stack(faceid_all_embeds, dim=0), dim=0)
    
    total_negative_prompt = negative_prompt
    
    print("Generating SDXL")
    image = ip_model.generate(
        prompt=prompt, negative_prompt=total_negative_prompt, faceid_embeds=average_embedding,
        scale=likeness_strength, width=864, height=1152, guidance_scale=face_strength, num_inference_steps=30
    )

    print(image)
    try:
        image[0].save('save_image.jpg')
        print(type(image[0]),"no error")

    except:
        print("error")
    return image

def swap_to_gallery(images):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

def remove_back_to_files():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
css = '''
h1{margin-bottom: 0 !important}
'''
with gr.Blocks(css=css) as demo:
    #gr.Markdown("# IP-Adapter-FaceID SDXL demo")
    #gr.Markdown("A simple Demo for the [h94/IP-Adapter-FaceID SDXL model](https://huggingface.co/h94/IP-Adapter-FaceID) together with [Juggernaut XL v7](https://huggingface.co/stablediffusionapi/juggernaut-xl-v7). You should run this on at least 24 GB of VRAM.")
    with gr.Row():
        with gr.Column():
            files = gr.Files(
                        label="Drag 1 or more photos of your face",
                        file_types=["image"]
                    )
            uploaded_files = gr.Gallery(label="Your images", visible=False, columns=5, rows=1, height=250)
            with gr.Column(visible=False) as clear_button:
                remove_and_reupload = gr.ClearButton(value="Remove files and upload new ones", components=files, size="sm")
            prompt = gr.Textbox(label="Prompt",
                        info="Try something like 'a photo of a man/woman/person'",
                        placeholder="A photo of a man/woman/person ...",
                        value="")
            negative_prompt = gr.Textbox(label="Negative Prompt", info="What the model should NOT produce.",placeholder="low quality", value="(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth")
            style = "Photorealistic"
            face_strength = gr.Slider(label="Prompt Strength", info="How much the written prompt weighs into the generated images.", value=7.5, step=0.1, minimum=0, maximum=15)
            likeness_strength = gr.Slider(label="Photo Embedding Strength", info="How much your uploaded files weigh into the generated images.", value=0.4, step=0.1, minimum=0, maximum=1.5)
            submit = gr.Button("Submit", variant="primary")
        with gr.Column():
            gallery = gr.Gallery(label="Generated Images")
        files.upload(fn=swap_to_gallery, inputs=files, outputs=[uploaded_files, clear_button, files])
        remove_and_reupload.click(fn=remove_back_to_files, outputs=[uploaded_files, clear_button, files])
        submit.click(fn=generate_image,
                    inputs=[files,prompt,negative_prompt, face_strength, likeness_strength],
                    outputs=gallery)
    
    # gr.Markdown("This demo includes extra features to mitigate the implicit bias of the model and prevent explicit usage of it to generate content with faces of people, including third parties, that is not safe for all audiences, including naked or semi-naked people.")
    
demo.launch()