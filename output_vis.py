import os
from glob import glob
import argparse
import numpy as np
from library.utils import normalize, to_gif_and_save

def run(args):

    output_dir = args.folder_name + "_vis_" + args.order_type

    basedir = args.folder_name

    scene_list = [d for d in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, d))]
    scene_list = sorted(scene_list)
    # scene_list = scene_list[17:]

    if args.order_type == "denoise_step":

        for scene in scene_list:

            scene_dir = os.path.join(basedir, scene)
            feature_type_list = [d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d))]

            for feature_type in feature_type_list:

                denoise_step_list = sorted(os.listdir(os.path.join(basedir, scene, feature_type)))

                for denoise_step in denoise_step_list:
                    frames_features = sorted(glob(os.path.join(basedir, scene, feature_type, denoise_step, "*.npy")))

                    stacked_fatures = []
                    for frames_feature in frames_features:
                        stacked_fatures.append(np.load(frames_feature))
                    stacked_fatures = np.stack(stacked_fatures, 0)

                    """ Channel이 16이라 우선 mean값을 관찰함"""
                    # stacked_fatures = np.mean(stacked_fatures, 1)

                    norm_stacked_features = normalize(stacked_fatures, norm_type="min_max")
                    
                    print(feature_type)
                    print("diff" in feature_type)

                    save_denoise_step = denoise_step

                    save_path = os.path.join(output_dir, scene, save_denoise_step, feature_type + ".gif")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    to_gif_and_save(norm_stacked_features, save_path)
    
    elif args.order_type == "frame_number":

        for scene in scene_list:

            scene_dir = os.path.join(basedir, scene)
            feature_type_list = [d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d))]

            for feature_type in feature_type_list:
                denoise_step_list = sorted(os.listdir(os.path.join(basedir, scene, feature_type)))
                frame_number = len(sorted(glob(os.path.join(basedir, scene, feature_type, denoise_step_list[0], "*.npy"))))

                for frame_idx in range(frame_number):
                    stacked_fatures = []
                    for denoise_step in denoise_step_list:
                        frames_features = sorted(glob(os.path.join(basedir, scene, feature_type, denoise_step, "*.npy")))
                        stacked_fatures.append(np.load(frames_features[frame_idx]))
                    
                    stacked_fatures = np.stack(stacked_fatures, 0)
                    """ Channel이 16이라 우선 mean값을 관찰함"""
                    stacked_fatures = np.mean(stacked_fatures, 1)

                    norm_stacked_features = normalize(stacked_fatures, norm_type="min_max")
                    
                    save_path = os.path.join(output_dir, scene, feature_type, "%03d_frame" % frame_idx, "vis.gif")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    to_gif_and_save(norm_stacked_features, save_path)

    else:
        print("order type error")
        exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--order_type', default='denoise_step', help="어떤 순서로 visualization? [denoise_step or frame_number]")
    parser.add_argument('--folder_name', default='token_drop', help="")
    args = parser.parse_args()

    run(args)
