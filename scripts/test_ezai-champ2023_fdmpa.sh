# sh path.sh

# Evaluate on the full EZAI Championship dataset split.
# FDMPA handcrafted features will be cached to:
#   data/ezai-championship2023/tr_handcrafted_feats.pkl
python3 -m prosody_scorer.test \
    --dataset eoleedi/ezai-championship2023 \
    --split train \
    --model FDMPAScorer \
    --checkpoint "exp/fdmpa_two_stage_codebook/models/best_audio_model.pth" \
    --aspect prosodic
