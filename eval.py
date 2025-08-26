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


class Video_Metrics_Processor:
    def __init__(
        self,
        prompt_video_path,
        ref_video_path,
        prompt,
        cotracker_path,
        clip_model="openai/clip-vit-base-patch32",
        device="cuda",
    ):
        self.prompt_video_path = prompt_video_path
        self.prompt = prompt
        self.ref_video_file = ref_video_path
        if not torch.cuda.is_available():
            device = "cpu"
            print("CUDA is not available, using CPU instead.")
        self.device = device
        cotracker_model = CoTrackerPredictor(checkpoint=cotracker_path)
        self.cotracker_model = cotracker_model.to(self.device)
        self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model)

    def compute_text_frame_sim(self, images):  # operate on a single video
        with torch.no_grad():

            max_len = getattr(getattr(self.clip_model.config, "text_config", None), "max_position_embeddings", 77)
            text_inputs = self.processor(
                text=[self.prompt],                # 显式列表更稳
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

        sim_scores = (image_feats @ text_feats.T).squeeze(-1).detach().cpu().mean().item()
        return sim_scores, image_feats  # image_feats: [N, D] on self.device

    @staticmethod
    def compute_temporal_consistency(img_feats):  # operate on a single video
        sims = []
        for i in range(len(img_feats) - 1):
            a = img_feats[i]
            b = img_feats[i + 1]
            sims.append((a @ b.T) / (a.norm() * b.norm()))
        return torch.tensor(sims).detach().cpu().mean().item()  # a single number

    @staticmethod
    def get_similarity_matrix(tracklets1, tracklets2):
        # tracklets*: [N, T, 2] and [M, T, 2]
        eps = 1e-8
        displacements1 = tracklets1[:, 1:] - tracklets1[:, :-1]  # [N, T-1, 2]
        displacements1 = displacements1 / (displacements1.norm(dim=-1, keepdim=True) + eps)

        displacements2 = tracklets2[:, 1:] - tracklets2[:, :-1]  # [M, T-1, 2]
        displacements2 = displacements2 / (displacements2.norm(dim=-1, keepdim=True) + eps)

        similarity_matrix = torch.einsum("ntc,mtc->nmt", displacements1, displacements2).mean(dim=-1)  # [N, M]
        return similarity_matrix

    def get_tracklets(self, video_path, mask=None):
        video = read_video_from_path(video_path)  # numpy: [T, H, W, 3], uint8
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(self.device)  # [1, T, 3, H, W]
        with torch.no_grad():
            pred_tracks_small, pred_visibility_small = self.cotracker_model(
                video, grid_size=55, segm_mask=mask
            )  # [B, T, L, 2]
        pred_tracks_small = rearrange(pred_tracks_small, "b t l c -> (b l) t c")  # [N, T, 2]
        return pred_tracks_small

    @staticmethod
    def get_mf_score(similarity_matrix):
        # similarity_matrix: [N, M]
        n, m = similarity_matrix.shape
        if n == m:
            eye = torch.eye(n, device=similarity_matrix.device)
            similarity_matrix = similarity_matrix * (1 - eye)
        max_similarity, _ = similarity_matrix.max(dim=1)  # per source track best match in ref
        average_score = max_similarity.mean()
        return average_score.item()

    def process_all_videos(self):
        files = os.listdir(self.prompt_video_path)
        files = [f for f in files if f.endswith(".mp4")]
        mf_scores = []
        text_frame_sims = []
        temporal_consists = []
        results = []
        for file in files:
            video_path = os.path.join(self.prompt_video_path, file)
            #ref_video_file = os.path.join(self.ref_video_path, file)
            if not os.path.exists(self.ref_video_file):
                print(f"[Warn] Reference video not found for {file}: {self.ref_video_file}, skip.")
                continue

            print(f"Processing {video_path} ...")
            print(f"prompt: {self.prompt}")

            # Motion fidelity via CoTracker
            tracklets_gen = self.get_tracklets(video_path)  # (num_tracks, num_frames, 2)
            tracklets_ref = self.get_tracklets(self.ref_video_file)  # (num_tracks, num_frames, 2)

            similarity_matrix = self.get_similarity_matrix(tracklets_gen, tracklets_ref)
            mf_score = self.get_mf_score(similarity_matrix)

            # Text-frame similarity and temporal consistency via CLIP
            vr = VideoReader(video_path, ctx=cpu(0))
            frames = vr.get_batch(range(len(vr))).asnumpy()  # N, H, W, 3
            images = list(frames)
            text_frame_sim, img_feats = self.compute_text_frame_sim(images)
            temporal_consist = self.compute_temporal_consistency(img_feats)

            mf_scores.append(mf_score)
            text_frame_sims.append(text_frame_sim)
            temporal_consists.append(temporal_consist)
            results.append(
                {
                    "video": file,
                    "text_frame_similarity": text_frame_sim,
                    "temporal_consistency": temporal_consist,
                    "mf_score": mf_score,
                }
            )

        avg_mf = sum(mf_scores) / len(mf_scores) if mf_scores else 0
        avg_text_frame_sim = sum(text_frame_sims) / len(text_frame_sims) if text_frame_sims else 0
        avg_temporal_consist = sum(temporal_consists) / len(temporal_consists) if temporal_consists else 0
        results.append(
            {
                "video": "Average",
                "text_frame_similarity": avg_text_frame_sim,
                "temporal_consistency": avg_temporal_consist,
                "mf_score": avg_mf,
            }
        )
        return results


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