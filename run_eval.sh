export CUDA_VISIBLE_DEVICES=6
python /home/Wind645/code/evaluate/eval.py \
    --cotracker_path /home/Wind645/checkpoints/scaled_offline.pth \
    --gen_videos_path /home/Wind645/code/diffsynth4MT/exp_davis_amf \
    --ref_videos_path /home/Wind645/code/diffsynth4MT/41_ref \
    --result_path only_effi_davis