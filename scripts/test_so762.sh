sh path.sh

python -m prosody_scorer.test \
    --dataset mispeech/speechocean762 \
    --model ClusterScorer \
    --checkpoint "exp/SSLfeat_fluency+prosodicScore/1e-3-3-25-32-ClusterScorer-br/0/models/best_audio_model.pth" \
    --kmeans_model exp/kmeans/kmeans_model.joblib \
    --aspect fluency prosodic