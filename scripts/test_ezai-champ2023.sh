sh path.sh

python -m prosody_scorer.test \
    --dataset eoleedi/ezai-championship2023 \
    --model ClusterScorer \
    --checkpoint "exp/SpeechOcean762/SSLfeat_fluency+prosodicScore/1e-3-3-25-32-ClusterScorer-br/0/models/best_audio_model.pth" \
    --kmeans_model exp/kmeans/so762/kmeans_model.joblib \
    --aspect fluency prosodic