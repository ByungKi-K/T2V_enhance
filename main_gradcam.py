import os
import imageio
import numpy as np

# === SaveCrossAttnProcessor: cross-attn 확률 저장용 ===
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import CogVideoXPipeline

from diffusers.models.attention_processor import Attention, CogVideoXAttnProcessor2_0_attnW, CogVideoXAttnProcessor2_0_attnW_Grad
from library.utils import to_gif_and_save

def normalize(array):
    output = (array - np.min(array)) / (np.max(array) - np.min(array))
    return output


def save_tokendrop_list(save_list, folder_name, sample_folder_name, save_folder, prompt_list):

    s_idx = 0
    e_idx = len(save_list)
    frame_number = save_list[0].shape[0]

    for denoise_step in range(s_idx, e_idx):
        for frame_idx in range(frame_number):

            diff_feature = save_list[denoise_step].float()

            H, W = diff_feature.shape[1:3]

            diff_feature = F.interpolate(diff_feature.unsqueeze(1), (H*2, W*2))[:, 0, ...] # shape: [frame num, H, W, 226]

            diff_feature = diff_feature.detach().cpu().numpy()
            diff_feature = normalize(diff_feature)

            save_path = os.path.join(save_folder, sample_folder_name, folder_name, "%03d_denoise_step" % denoise_step, "otherwise.gif")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            to_gif_and_save(diff_feature, save_path)

def save_attnW_list(save_list, folder_name, sample_folder_name, save_folder, prompt_list):

    s_idx = 0
    e_idx = len(save_list)
    frame_number = save_list[0][0].shape[1]

    uncond_list = []
    text_list = []

    for save_ in save_list:
        uncond_list.append(save_[0])
        text_list.append(save_[1])

    for denoise_step in range(s_idx, e_idx):
        for frame_idx in range(frame_number):

            uncond_latent_meanhead = uncond_list[denoise_step].float()
            text_latent_meanhead = text_list[denoise_step].float()

            uncond_latent_meanhead_meanblocks = uncond_latent_meanhead.mean(0) # shape: [frame num, H//2, W//2, 226]
            text_latent_meanhead_meanblocks = text_latent_meanhead.mean(0) # shape: [frame num, H//2, W//2, 226]

            H, W = uncond_latent_meanhead_meanblocks.shape[1:3]

            uncond_latent_meanhead_meanblocks = F.interpolate(uncond_latent_meanhead_meanblocks.permute(0, 3, 1, 2), (H*2, W*2)) # shape: [frame num, H, W, 226]
            text_latent_meanhead_meanblocks = F.interpolate(text_latent_meanhead_meanblocks.permute(0, 3, 1, 2), (H*2, W*2)) # shape: [frame num, H, W, 226]

            for idx, prompt_slice in enumerate(prompt_list):

                text_latent_slice = text_latent_meanhead_meanblocks[:, idx].numpy()
                text_latent_slice_norm = normalize(text_latent_slice)
                
                save_path = os.path.join(save_folder, sample_folder_name, folder_name, "%03d_denoise_step" % denoise_step, "%03d_text_%s.gif" % (idx,prompt_slice))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                to_gif_and_save(text_latent_slice_norm, save_path)

            text_latent_otherwise = text_latent_meanhead_meanblocks[:, idx+1:].mean(1).numpy()
            text_latent_otherwise = normalize(text_latent_otherwise)

            save_path = os.path.join(save_folder, sample_folder_name, folder_name, "%03d_denoise_step" % denoise_step, "otherwise.gif")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            to_gif_and_save(text_latent_slice_norm, save_path)

def save_attnW_list2(save_list, folder_name, sample_folder_name, save_folder, prompt_list):

    s_idx = 0
    e_idx = len(save_list)
    frame_number = save_list[0][0].shape[1]

    uncond_list = []
    text_list = []

    for save_ in save_list:
        uncond_list.append(save_[2])
        text_list.append(save_[3])

    for denoise_step in range(s_idx, e_idx):
        for frame_idx in range(frame_number):

            uncond_latent_meanhead = uncond_list[denoise_step].float()
            text_latent_meanhead = text_list[denoise_step].float()

            uncond_latent_meanhead_meanblocks = uncond_latent_meanhead.mean(0) # shape: [frame num, H//2, W//2, 226]
            text_latent_meanhead_meanblocks = text_latent_meanhead.mean(0) # shape: [frame num, H//2, W//2, 226]

            H, W = uncond_latent_meanhead_meanblocks.shape[1:3]

            uncond_latent_meanhead_meanblocks = F.interpolate(uncond_latent_meanhead_meanblocks.permute(0, 3, 1, 2), (H*2, W*2)) # shape: [frame num, H, W, 226]
            text_latent_meanhead_meanblocks = F.interpolate(text_latent_meanhead_meanblocks.permute(0, 3, 1, 2), (H*2, W*2)) # shape: [frame num, H, W, 226]

            for idx, prompt_slice in enumerate(prompt_list):

                text_latent_slice = text_latent_meanhead_meanblocks[:, idx].numpy()
                text_latent_slice_norm = normalize(text_latent_slice)
                
                save_path = os.path.join(save_folder, sample_folder_name, folder_name, "%03d_denoise_step" % denoise_step, "%03d_text2_%s.gif" % (idx,prompt_slice))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                to_gif_and_save(text_latent_slice_norm, save_path)

            text_latent_otherwise = text_latent_meanhead_meanblocks[:, idx+1:].mean(1).numpy()
            text_latent_otherwise = normalize(text_latent_otherwise)

            save_path = os.path.join(save_folder, sample_folder_name, folder_name, "%03d_denoise_step" % denoise_step, "otherwise2.gif")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            to_gif_and_save(text_latent_slice_norm, save_path)


def save_attnW_list3(save_list, folder_name, sample_folder_name, save_folder, prompt_list):

    s_idx = 0
    e_idx = len(save_list)
    frame_number = save_list[0][0].shape[1]

    text_list = []

    for save_ in save_list:
        text_list.append(save_[4])

    for denoise_step in range(s_idx, e_idx):
        for frame_idx in range(frame_number):
            text_latent_meanhead = text_list[denoise_step].float()

            text_latent_meanhead_meanblocks = text_latent_meanhead.mean(0) # shape: [frame num, H//2, W//2, 226]

            H, W = text_latent_meanhead_meanblocks.shape[1:3]

            text_latent_meanhead_meanblocks = F.interpolate(text_latent_meanhead_meanblocks.unsqueeze(1), (H*2, W*2))[:, 0, ...] # shape: [frame num, H, W, 226]

            text_latent_otherwise = text_latent_meanhead_meanblocks.numpy()
            text_latent_otherwise = normalize(text_latent_otherwise)

            save_path = os.path.join(save_folder, sample_folder_name, folder_name, "%03d_denoise_step" % denoise_step, "difference.gif")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            to_gif_and_save(text_latent_otherwise, save_path)

            


def save_attnWG_list(save_list, folder_name, sample_folder_name, save_folder, prompt_list):

    s_idx = 0
    e_idx = len(save_list)
    frame_number = save_list[0].shape[0]
    
    text_list = save_list

    for denoise_step in range(s_idx, e_idx):
        for frame_idx in range(frame_number):

            text_latent_meanhead = text_list[denoise_step].float()

            text_latent_meanhead_meanblocks = text_latent_meanhead # shape: [frame num, H//2, W//2, 226]

            H, W = text_latent_meanhead_meanblocks.shape[1:3]

            text_latent_meanhead_meanblocks = F.interpolate(text_latent_meanhead_meanblocks.permute(0, 3, 1, 2), (H*2, W*2)) # shape: [frame num, H, W, 226]

            for idx, prompt_slice in enumerate(prompt_list):

                text_latent_slice = text_latent_meanhead_meanblocks[:, idx].detach().cpu().numpy()
                text_latent_slice_norm = normalize(text_latent_slice)
                
                save_path = os.path.join(save_folder, sample_folder_name, folder_name, "%03d_denoise_step" % denoise_step, "%03d_text_%s.gif" % (idx,prompt_slice))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                to_gif_and_save(text_latent_slice_norm, save_path)



def save_output_list(save_list, folder_name, sample_folder_name, save_folder, is_latent=False):

    s_idx = 0
    e_idx = len(save_list)

    frame_number = save_list[0].shape[0]

    for denoise_step in range(s_idx, e_idx):
        for frame_idx in range(frame_number):
            save_latent = save_list[denoise_step][frame_idx].float().numpy()
            
            if is_latent:
                save_path = os.path.join(save_folder, sample_folder_name, folder_name, "%03d_denoise_step" % denoise_step, "%03d_frame.npy" % frame_idx)
            else:
                save_path = os.path.join(save_folder, sample_folder_name, folder_name, "%03d_denoise_step" % (denoise_step+1), "%03d_frame.npy" % frame_idx)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            np.save(save_path, save_latent)


# 3) Processor 장착 (모듈마다 새 인스턴스 생성, 같은 리스트 공유)
# def install_cogvideox_processor(transformer, store_list1, store_list2):
#     n = 0
#     for _, m in transformer.named_modules():
#         if isinstance(m, Attention):
#             try:
#                 m.set_processor(CogVideoXAttnProcessor2_0_attnW(store_list1=store_list1, store_list2=store_list2, head_average=True))
#                 n += 1
#             except Exception:
#                 pass
#     print(f"[xattn] installed on {n} Attention modules")


def install_cogvideox_processor(transformer, store_list1, store_list2, store_list3):
    proc = CogVideoXAttnProcessor2_0_attnW(store_list1=store_list1, store_list2=store_list2, store_list3=store_list3, head_average=True)
    n = 0
    for _, m in transformer.named_modules():
        if isinstance(m, Attention):
            try:
                m.set_processor(proc)
                n += 1
            except Exception:
                pass
    print(f"[xattn] installed on {n} Attention modules")
    

def install_cogvideox_processor_grad(transformer, store_list):
    n = 0
    for _, m in transformer.named_modules():
        if isinstance(m, Attention):
            # 모듈마다 새 인스턴스(권장). 같은 store_list 공유.
            m.set_processor(CogVideoXAttnProcessor2_0_attnW_Grad(store_list=store_list, head_average=True))
            n += 1
    print(f"[xattn-grad] installed on {n} Attention modules")
    

# text_motion.txt에서 한 줄씩 불러오기
with open("text_fails.txt", "r") as f:
    prompts = [line.strip() for line in f if line.strip()]

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)
# 2) 파이프라인 객체에 저장 리스트를 달기
gradcam_type = "attention_weight"

if gradcam_type == "attention_weight" or gradcam_type == "token_drop":
    pipe._saved_attns1 = []   # ← 중요!
    pipe._saved_attns2 = []   # ← 중요!
    pipe._saved_attns3 = []   # ← 중요!
    install_cogvideox_processor(pipe.transformer, pipe._saved_attns1, pipe._saved_attns2, pipe._saved_attns3)

if gradcam_type == "attention_grad_weight":
    # 1) 저장 리스트를 파이프라인 객체에 붙여 공유
    pipe._saved_attns = []          # (옵션) 평균 확률 스냅샷용
    pipe._gw_handles = []           # grad-weighted용 내부 핸들 모음(선택)
    install_cogvideox_processor_grad(pipe.transformer, pipe._saved_attns)

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

fps = 8
output_folder = gradcam_type + "_experiment"
os.makedirs(output_folder, exist_ok=True)

for prompt in prompts:

    video, _, _, _, _, additional_list, prompt_tokens = pipe(
        prompt=[prompt, "A gymnast standing still in uniform"],
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        generator=torch.Generator(device="cuda").manual_seed(42),
        guidance_scale=6,
        gradcam_type=gradcam_type,
        gradcam_keywords="dog"
    )
    video = video[0]

    if gradcam_type == "attention_weight":
        sample_num = len(os.listdir(output_folder))
        sample_folder_name = "%04d_sample" % sample_num

        save_folder = os.path.join(output_folder, sample_folder_name)
        os.makedirs(save_folder, exist_ok=True)

        output_path = os.path.join(save_folder, "output.mp4")

        with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
            for frame in video:  # video는 PIL.Image 리스트
                writer.append_data(np.array(frame))  # PIL → numpy array 변환
                
        save_attnW_list3(additional_list, "attention_weight", sample_folder_name, output_folder, prompt_tokens)
        save_attnW_list(additional_list, "attention_weight", sample_folder_name, output_folder, prompt_tokens)
        save_attnW_list2(additional_list, "attention_weight", sample_folder_name, output_folder, prompt_tokens)

    if gradcam_type == "attention_grad_weight":
        sample_num = len(os.listdir(output_folder))
        sample_folder_name = "%04d_sample" % sample_num
        save_attnWG_list(additional_list, "attention_grad_weight", sample_folder_name, output_folder, prompt_tokens)

        save_folder = os.path.join(output_folder, sample_folder_name)
        os.makedirs(save_folder, exist_ok=True)

        output_path = os.path.join(save_folder, "output.mp4")

        with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
            for frame in video:  # video는 PIL.Image 리스트
                writer.append_data(np.array(frame))  # PIL → numpy array 변환

    if gradcam_type == "token_drop":

        sample_num = len(os.listdir(output_folder))
        sample_folder_name = "%04d_sample" % sample_num

        save_folder = os.path.join(output_folder, sample_folder_name)
        os.makedirs(save_folder, exist_ok=True)

        output_path = os.path.join(save_folder, "output.mp4")

        with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
            for frame in video:  # video는 PIL.Image 리스트
                writer.append_data(np.array(frame))  # PIL → numpy array 변환

        print(f"Saved video to {output_path}")

        save_tokendrop_list(additional_list, "token_drop", sample_folder_name, output_folder, prompt_tokens)
