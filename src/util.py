import torch
import argparse


def load_checkpoint_and_upload_to_huggingface(
    model,
    checkpoint_path: str,
    repo_name: str,
    huggingface_token: str,
    device: str = "cpu",
):
    """
    Load model checkpoint and upload to Hugging Face Hub.

    Args:
        model: The model instance to load the checkpoint into.
        checkpoint_path (str): Path to the model checkpoint file.
        repo_name (str): Name of the Hugging Face repository to upload to.
        huggingface_token (str): Hugging Face authentication token.
        device (str): Device to map the model to ('cpu' or 'cuda').
    """
    # Load the model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)

    # Upload the model to Hugging Face Hub
    model.push_to_hub(repo_name, use_auth_token=huggingface_token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load model checkpoint and upload to Hugging Face Hub"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Name of the Hugging Face repository to upload to",
    )
    parser.add_argument(
        "--huggingface-token",
        type=str,
        required=True,
        help="Hugging Face authentication token",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to map the model to ('cpu' or 'cuda')",
    )
    args = parser.parse_args()

    # Example model instantiation (replace with actual model)
    model = torch.nn.Module()  # Replace with actual model class

    load_checkpoint_and_upload_to_huggingface(
        model,
        args.checkpoint_path,
        args.repo_name,
        args.huggingface_token,
        args.device,
    )
