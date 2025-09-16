import os
import imageio
import numpy as np
import torch
from diffusers import CogVideoXPipeline

def save_output_list(save_list, folder_name, sample_folder_name, save_folder, is_latent=False):

    s_idx = 0
    e_idx = len(save_list)

    frame_number = save_list[0].shape[1]

    for denoise_step in range(s_idx, e_idx):
        for frame_idx in range(frame_number):
            save_latent = save_list[denoise_step][0, frame_idx]
            
            if is_latent:
                save_path = os.path.join(save_folder, sample_folder_name, folder_name, "%03d_denoise_step" % denoise_step, "%03d_frame.npy" % frame_idx)
            else:
                save_path = os.path.join(save_folder, sample_folder_name, folder_name, "%03d_denoise_step" % (denoise_step+1), "%03d_frame.npy" % frame_idx)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            np.save(save_path, save_latent)

# text_motion.txt에서 한 줄씩 불러오기
with open("text_motion.txt", "r") as f:
    prompts = [line.strip() for line in f if line.strip()]

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

fps = 8
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

for idx, prompt in enumerate(prompts):
    print(f"[{idx}] Generating video for prompt: {prompt}")
    video, text_original_list, uncond_original_list, diff_original_list, final_original_list = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42),
    )
    video = video[0]

    sample_num = len(os.listdir(output_folder))
    sample_folder_name = "%04d_sample" % sample_num

    save_folder = os.path.join(output_folder, sample_folder_name)
    os.makedirs(save_folder, exist_ok=True)

    output_path = os.path.join(save_folder, "output.mp4")

    with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
        for frame in video:  # video는 PIL.Image 리스트
            writer.append_data(np.array(frame))  # PIL → numpy array 변환

    print(f"Saved video to {output_path}")

    save_output_list(text_original_list, "text_x0", sample_folder_name, output_folder, is_latent=False)
    save_output_list(uncond_original_list, "uncond_x0", sample_folder_name, output_folder, is_latent=False)
    save_output_list(diff_original_list, "diff_x0", sample_folder_name, output_folder, is_latent=False)
    save_output_list(final_original_list, "final_x0", sample_folder_name, output_folder, is_latent=False)

    prompt_output_path = os.path.join(output_folder, sample_folder_name, "prompt.txt")

    with open(prompt_output_path, "w", encoding="utf-8") as f:
        f.write(prompt)

