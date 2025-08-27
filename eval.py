import argparse
import os
import json
from einops import rearrange
from omegaconf import OmegaConf
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import read_video_from_path
import torch
import numpy as np
from decord import VideoReader, cpu
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from vbench import VBench

##############################################################
# Aranged in categories for better readability
# Directory structure assumption:
# ref_videos_dir
# base_path/       # base path of a specific category
#   prompts.json
#   video1.mp4
#   ...
# gen_videos_dir
# base_path/       # base path of a specific category
#   videos/      # contains subdirs prompt1..prompt5
#    prompt1/
#      video1.mp4
#      ...
#    ...
#   ...
###############################################################

class Video_Metrics_Processor:
    def __init__(
        self,
        base_ref_path: str, # Path to the directory that contains the prompt file and videos,
        base_gen_path: str, # Path to the directory that contains the generated videos
        cotracker_ckpt: str, # Path to the CoTracker checkpoint, we use the scaled_offline.pth
        clip_model="openai/clip-vit-base-patch32",
        device="cuda",
    ):
        self.base_ref_path = base_ref_path
        self.base_gen_path = base_gen_path
        if not torch.cuda.is_available():
            device = "cpu"
            print("CUDA is not available, using CPU instead.")
        self.device = device
        cotracker_model = CoTrackerPredictor(checkpoint=cotracker_ckpt)
        self.cotracker_model = cotracker_model.to(self.device)
        self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model)
        
        self.vbench = VBench(
            device=device,
            full_info_dir='VBench_full_info.json',  # Path to downloaded JSON
            output_path='evaluation_results'    # Directory to save results
        )
        
    def read_prompt_file(self):
        '''
        Read the prompt file from the reference video directory.
        And return a dictionary mapping video names to their prompts.
        '''
        prompt_file = os.path.join(self.base_ref_path, "prompts.json")
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt file {prompt_file} does not exist.")
        with open(prompt_file, "r") as f:
            prompts = json.load(f)
        return prompts  # {video_name: [prompt1, prompt2..prompt5] ...}
    
    def read_ref_videos(self, prompts: dict):
        '''
        Utilize the video name to read the reference video, and return a dictionary
        mapping video name to its ref video path.
        '''
        ref_video_paths = {}
        for key in prompts.keys():
            ref_video_path = os.path.join(self.base_ref_path, key + ".mp4")
            if not os.path.exists(ref_video_path):
                raise FileNotFoundError(f"Reference video {ref_video_path} does not exist.")
            ref_video_paths[key] = ref_video_path
        return ref_video_paths # {video_name: ref_video_path}
    
    def compute_text_frame_sim(self, images, prompt):  # operate on a single video
        with torch.no_grad():

            max_len = getattr(getattr(self.clip_model.config, "text_config", None), "max_position_embeddings", 77)
            text_inputs = self.processor(
                text=[prompt],                # 显式列表更稳
                return_tensors="pt",
                padding=True,
                truncation=True,                   # 开启截断
                max_length=max_len,                # 限制到模型最大长度（CLIP 一般为 77）
            ).to(self.device)
            text_feats = self.clip_model.get_text_features(**text_inputs)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            image_inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            image_feats = self.clip_model.get_image_features(**image_inputs)
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)  # (N, D)

        text_frame_sim_score = (image_feats @ text_feats.T).squeeze(-1).detach().cpu().mean().item()
        sims = []
        for i in range(len(image_feats) - 1):
            a = image_feats[i]
            b = image_feats[i + 1]
            sims.append((a @ b.T) / (a.norm() * b.norm()))
        temporal_sim_score = torch.tensor(sims).detach().cpu().mean().item()
        return  text_frame_sim_score, temporal_sim_score 

    def get_mf_score(self, ref_video_path, gen_video_path,):
        
        def get_tracklets(self, video_path, mask=None):
            video = read_video_from_path(video_path)  # numpy: [T, H, W, 3], uint8
            video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(self.device)  # [1, T, 3, H, W]
            with torch.no_grad():
                pred_tracks_small, pred_visibility_small = self.cotracker_model(
                    video, grid_size=55, segm_mask=mask
                )  # [B, T, L, 2]
            pred_tracks_small = rearrange(pred_tracks_small, "b t l c -> (b l) t c")  # [N, T, 2]
            return pred_tracks_small
        
        def get_similarity_matrix(tracklets1, tracklets2):
            # tracklets*: [N, T, 2] and [M, T, 2]
            eps = 1e-8
            displacements1 = tracklets1[:, 1:] - tracklets1[:, :-1]  # [N, T-1, 2]
            displacements1 = displacements1 / (displacements1.norm(dim=-1, keepdim=True) + eps)

            displacements2 = tracklets2[:, 1:] - tracklets2[:, :-1]  # [M, T-1, 2]
            displacements2 = displacements2 / (displacements2.norm(dim=-1, keepdim=True) + eps)

            similarity_matrix = torch.einsum("ntc,mtc->nmt", displacements1, displacements2).mean(dim=-1)  # [N, M]
            return similarity_matrix
        
        ref_tracklets = get_tracklets(self, ref_video_path)  # (num_tracks, num_frames, 2)
        gen_tracklets = get_tracklets(self, gen_video_path)  # (num_tracks, num_frames, 2)
        similarity_matrix = get_similarity_matrix(ref_tracklets, gen_tracklets)
        # similarity_matrix: [N, M]
        n, m = similarity_matrix.shape
        if n == m:
            eye = torch.eye(n, device=similarity_matrix.device)
            similarity_matrix = similarity_matrix * (1 - eye)
        max_similarity, _ = similarity_matrix.max(dim=1)  # per source track best match in ref
        average_score = max_similarity.mean()
        return average_score.item()
    
    def get_vbench_score(self, gen_video_path, prompt):
        dimensions = [
            'subject_consistency',
            'background_consistency',
            'motion_smoothness',
            'dynamic_degree',
            'aesthetic_quality',
            'imaging_quality'
        ]
        results = self.vbench.evaluate(
            videos_path=gen_video_path, 
            name='results', 
            prompt_list=[prompt], 
            dimension_list=dimensions,
            mode='custom_input'
        )
        return results
    
    def process_all_videos(self):
        prompts = self.read_prompt_file()  # {video_name: [prompt1, prompt2..prompt5]}
        ref_video_paths = self.read_ref_videos(prompts) # {video_name: ref_video_path}
        all_results = {}
        global_scores = []
        for video_name, ref_video_path in ref_video_paths.items():
            video_results = {}
            video_prompt_scores = []
            for i, prompt in enumerate(prompts[video_name]):
                gen_videos_dir = os.path.join(self.base_gen_path, video_name, f"prompt{i+1}")
                videos = os.listdir(gen_videos_dir)
                if len(videos) == 0:
                    print(f"[Warn] No generated videos found in {gen_videos_dir}, skip.")
                    continue
                prompt_results = []
                for video in videos:
                    gen_video_path = os.path.join(gen_videos_dir, video)
                    if not os.path.exists(gen_video_path):
                        print(f"[Warn] Generated video not found for {video_name} prompt {i+1}: {gen_video_path}, skip.")
                        continue
                    print(f"Processing {gen_video_path} with reference {ref_video_path} and prompt: {prompt}")
                    
                    # Read generated video frames
                    vr = VideoReader(gen_video_path, ctx=cpu(0))
                    frames = vr.get_batch(range(len(vr)))
                    images = list(frames)
                    
                    text_sim, temporal_sim = self.compute_text_frame_sim(images, prompt)
                    mf_score = self.get_mf_score(ref_video_path, gen_video_path)
                    vbench_scores = self.get_vbench_score(gen_video_path, prompt)
                    vbench_scores['text_frame_similarity'] = text_sim
                    vbench_scores['temporal_consistency'] = temporal_sim
                    vbench_scores['motion_fidelity'] = mf_score
                    vbench_scores['video_file'] = video
                    prompt_results.append(vbench_scores)
                    global_scores.append(vbench_scores)
                # 计算该prompt下所有视频的平均
                avg_scores = {}
                if len(prompt_results) > 0:
                    for key in prompt_results[0].keys():
                        if key == 'video_file': continue
                        avg_scores[key] = sum([res[key] for res in prompt_results]) / len(prompt_results)
                    avg_scores['video_file'] = 'Average'
                    prompt_results.append(avg_scores)
                video_results[f'prompt{i+1}'] = prompt_results
                video_prompt_scores.extend(prompt_results[:-1])  # 不包括平均行
            # 计算该video所有prompt的平均
            video_avg_scores = {}
            if len(video_prompt_scores) > 0:
                for key in video_prompt_scores[0].keys():
                    if key == 'video_file': continue
                    video_avg_scores[key] = sum([res[key] for res in video_prompt_scores]) / len(video_prompt_scores)
                video_avg_scores['video_file'] = 'Video_Average'
                video_results['video_average'] = video_avg_scores
            all_results[video_name] = video_results
        # 计算全局平均
        if len(global_scores) > 0:
            global_avg = {}
            for key in global_scores[0].keys():
                if key == 'video_file': continue
                global_avg[key] = sum([res[key] for res in global_scores]) / len(global_scores)
            global_avg['video_file'] = 'Global_Average'
            all_results['global_average'] = global_avg
        # 保存结果
        os.makedirs('results', exist_ok=True)
        with open('results/video_metrics_hierarchical.json', 'w') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print("Hierarchical results saved to results/video_metrics_hierarchical.json")
        return all_results
    
##############################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cotracker_path", type=str, default="/home/Wind645/checkpoints/scaled_offline.pth")
    parser.add_argument("--gen_videos_path", type=str, required=True)
    parser.add_argument("--ref_videos_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, default="results")
    args = parser.parse_args()

    prompt_file = os.path.join(args.ref_videos_path, "prompts.json")
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file {prompt_file} does not exist.")

    with open(prompt_file, "r") as f:
        prompts = json.load(f)

    all_results = []
    for video_name, prompt_3 in prompts.items():
        ref_video_path = os.path.join(args.ref_videos_path, video_name + ".mp4")
        gen_video_path = os.path.join(args.gen_videos_path, video_name)
        for i in range(3):
            prompt = prompt_3[i]
            gen_video_path_t = os.path.join(gen_video_path, "prompt" + str(i + 1))
            processor = Video_Metrics_Processor(
                prompt_video_path=gen_video_path_t,
                ref_video_path=ref_video_path,
                prompt=prompt,
                cotracker_path=args.cotracker_path,
            )
            results = processor.process_all_videos()  # list[dict]
            all_results.extend(results)  # flatten

    # Aggregate global averages over per-video entries (exclude the "Average" rows)
    entries = [r for r in all_results if isinstance(r, dict) and r.get("video") not in (None, "Average")]
    final_mf = [r["mf_score"] for r in entries]
    final_text_frame_sim = [r["text_frame_similarity"] for r in entries]
    final_temporal_consistency = [r["temporal_consistency"] for r in entries]

    avg_mf = sum(final_mf) / len(final_mf) if final_mf else 0
    avg_text_frame_sim = sum(final_text_frame_sim) / len(final_text_frame_sim) if final_text_frame_sim else 0
    avg_temporal_consistency = (
        sum(final_temporal_consistency) / len(final_temporal_consistency) if final_temporal_consistency else 0
    )

    print(f"Average MF Score: {avg_mf}")
    print(f"Average Text-Frame Similarity: {avg_text_frame_sim}")
    print(f"Average Temporal Consistency: {avg_temporal_consistency}")

    os.makedirs(args.result_path, exist_ok=True)
    result_file = os.path.join(args.result_path, "video_metrics_results.json")
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {result_file}")
    print("All videos processed.")