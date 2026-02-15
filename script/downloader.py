import os
import argparse
from modelscope import snapshot_download


def verify_model_files(model_dir):
    """
    Verify model files in the given directory
    
    Args:
        model_dir: Path to model directory
    """
    print("Verifying model files...")
    files = os.listdir(model_dir)
    print(f"Files in directory: {len(files)} items")
    
    config_files = [f for f in files if f.endswith('.json')]
    if config_files:
        print(f"Config files found: {config_files}")
    else:
        print("Warning: No config.json file found!")
    
    model_files = [f for f in files if f.endswith('.bin') or f.endswith('.safetensors')]
    if model_files:
        print(f"Model files found: {len(model_files)} files")


def download_model(model_name, save_dir=None, revision="master"):
    """
    Download model from ModelScope
    
    Args:
        model_name: Model name on ModelScope (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        save_dir: Directory to save the model (default: ~/huggingface)
        revision: Model revision (default: "master")
    """
    if save_dir is None:
        save_dir = os.path.expanduser("~/huggingface")
    
    print(f"Model: {model_name}")
    print(f"Cache directory: {save_dir}")
    print(f"Revision: {revision}")
    print("-" * 60)
    
    try:
        model_dir = snapshot_download(
            model_name,
            cache_dir=save_dir,
            revision=revision,
            local_files_only=False
        )
        
        print(f"Model available at: {model_dir}")
        print()
        verify_model_files(model_dir)
        return model_dir
    except Exception as e:
        print(f"\nError downloading model: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Download models from ModelScope")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model name on ModelScope (e.g., 'Qwen/Qwen2.5-0.5B-Instruct')"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save the model (default: ~/huggingface/model_name)"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="master",
        help="Model revision (default: 'master')"
    )
    
    args = parser.parse_args()
    
    model_dir = download_model(args.model_name, args.save_dir, args.revision)
    
    if model_dir:
        print(f"\n{'='*60}")
        print(f"USAGE:")
        print(f"python infer.py --model_path {model_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
