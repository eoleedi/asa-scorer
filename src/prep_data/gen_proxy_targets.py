import os
import sys
import argparse
import pickle
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import VitsModel, AutoTokenizer

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.proxy_features import (
    IntonationProxyFeatures,
    RhythmProxyFeatures,
    ProminenceProxyFeatures,
    extract_f0_contour,
    extract_syllable_features,
)


from scipy.stats import spearmanr


def generate_proxy_targets(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Dataset
    print(f"Loading dataset: {args.dataset_name} ({args.split})")
    dataset = load_dataset(args.dataset_name, split=args.split)

    # 2. Initialize TTS
    print("Initializing TTS model...")
    tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    tts_model.to(device)
    tts_model.eval()

    # 3. Initialize Proxy Feature Extractors
    intonation_extractor = IntonationProxyFeatures(alpha=0.001)
    rhythm_extractor = RhythmProxyFeatures(beta=0.1)
    prominence_extractor = ProminenceProxyFeatures()

    proxy_targets = {}

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Temporary directory for audio files
    temp_dir = os.path.join(args.output_dir, "temp_audio")
    os.makedirs(temp_dir, exist_ok=True)

    print("Processing utterances...")
    for idx, item in enumerate(tqdm(dataset)):
        utt_id = item.get("id", f"sample_{idx}")
        text = item.get("text", "")

        if not text:
            print(f"Skipping {utt_id}: No text found")
            continue

        # --- A. Generate Reference Audio (Teacher) ---
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = tts_model(**inputs)

        ref_waveform = output.waveform.cpu()
        ref_sr = tts_model.config.sampling_rate

        # Save ref audio temporarily
        ref_path = os.path.join(temp_dir, f"{utt_id}_ref.wav")
        torchaudio.save(ref_path, ref_waveform, ref_sr)

        # --- B. Get Learner Audio ---
        learner_audio = item["audio"]
        learner_waveform = torch.tensor(
            learner_audio["array"], dtype=torch.float32
        ).unsqueeze(0)
        learner_sr = learner_audio["sampling_rate"]

        # Save learner audio temporarily
        learner_path = os.path.join(temp_dir, f"{utt_id}_learner.wav")
        torchaudio.save(learner_path, learner_waveform, learner_sr)

        try:
            # --- C. Extract Features ---

            # 1. Intonation (F0)
            f0_learner = extract_f0_contour(learner_path, sr=16000)
            f0_ref = extract_f0_contour(ref_path, sr=16000)

            intonation_score = intonation_extractor.compute_teacher_quality(
                f0_learner, f0_ref
            )

            # 2. Rhythm (Durations)
            # Note: extract_syllable_features uses energy-based segmentation if alignment is missing
            syll_learner = extract_syllable_features(learner_path)
            syll_ref = extract_syllable_features(ref_path)

            # Calculate total duration
            dur_learner = len(learner_waveform.squeeze()) / learner_sr
            dur_ref = len(ref_waveform.squeeze()) / ref_sr

            rhythm_score = rhythm_extractor.compute_teacher_quality(
                syll_learner["durations"], dur_learner, syll_ref["durations"], dur_ref
            )

            # 3. Prominence
            # We need lexical stress for the reference.
            # For now, we'll assume a dummy stress pattern or infer it from the TTS text if possible.
            # Since we don't have a g2p with stress here easily, we'll use a simplified approach:
            # Just use the acoustic prominence scores from the reference as the "teacher" scores?
            # Or better: Use the prominence extractor to get scores for the learner,
            # and use the reference's acoustic prominence as the target?
            # The ProminenceProxyFeatures.compute_prominence_scores requires lexical stress.
            # Let's use a placeholder of all zeros for stress for now if we can't get it.

            # Actually, the ProminenceProxyLoss expects PAIRS of syllables (ranking).
            # We can generate these pairs from the REFERENCE audio's prominence.

            # Compute reference prominence scores (assuming no lexical stress info for now)
            ref_stress = np.zeros_like(syll_ref["durations"])  # Placeholder
            ref_prominence = prominence_extractor.compute_prominence_scores(
                syll_ref["f0_peaks"],
                syll_ref["intensities"],
                syll_ref["durations"],
                ref_stress,
            )

            # Generate pairs from reference
            prominence_pairs = prominence_extractor.generate_pairwise_preferences(
                ref_prominence
            )

            # Also compute learner prominence scores (for debugging/analysis, though model will predict them)
            learner_stress = np.zeros_like(syll_learner["durations"])
            learner_prominence = prominence_extractor.compute_prominence_scores(
                syll_learner["f0_peaks"],
                syll_learner["intensities"],
                syll_learner["durations"],
                learner_stress,
            )

            # Compute Prominence Quality Score (Spearman Correlation)
            # We need to align them first. Since we don't have alignment, we can't easily correlate.
            # But we can use the global statistics or just assume the number of syllables is roughly similar?
            # If counts differ, we can't correlate directly.
            # Fallback: Use the ranking accuracy of the learner scores against the reference pairs?
            # Yes! "How many of the reference pairs are satisfied by the learner scores?"

            concordant_pairs = 0
            total_pairs = len(prominence_pairs)
            if total_pairs > 0:
                # We need to map learner syllables to reference syllables to check pairs.
                # Without alignment, this is impossible.
                # So we can't compute a "Prominence Quality Score" easily without alignment.
                # BUT, the Intonation and Rhythm scores are computed using DTW/Global stats which handle misalignment.
                # Maybe we can skip Prominence Quality Score for now and just use the Ranking Loss as an auxiliary task to shape the embedding?
                # And the final score just uses Intonation and Rhythm + Global Embedding?
                prominence_score = 0.5  # Placeholder
            else:
                prominence_score = 0.5

            # Store everything
            proxy_targets[utt_id] = {
                "intonation_score": float(intonation_score),
                "rhythm_score": float(rhythm_score),
                "prominence_score": float(prominence_score),  # Placeholder
                "prominence_pairs": prominence_pairs,
                "ref_prominence": ref_prominence,  # Optional, for analysis
                "learner_prominence": learner_prominence,  # Optional
                "syllable_segments": syll_learner[
                    "segments"
                ],  # (N, 2) start, end in seconds
            }

        except Exception as e:
            print(f"Error processing {utt_id}: {e}")
            continue
        finally:
            # Cleanup temp files
            if os.path.exists(ref_path):
                os.remove(ref_path)
            if os.path.exists(learner_path):
                os.remove(learner_path)

    # 4. Save Results
    output_path = os.path.join(args.output_dir, f"{args.split}_proxy_targets.pkl")
    print(f"Saving proxy targets to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(proxy_targets, f)

    # Remove temp dir
    os.rmdir(temp_dir)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="eoleedi/ezai-championship2023"
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="data/proxy_targets")
    args = parser.parse_args()

    generate_proxy_targets(args)
