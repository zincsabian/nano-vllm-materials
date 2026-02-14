import argparse
from modelscope.hub.api import HubApi


def generate_owner_candidates(owner):
    """
    Generate possible owner name variations
    
    Args:
        owner: Original owner name
    
    Returns:
        List of candidate owner names
    """
    candidates = []
    
    candidates.append(owner)
    
    variations = [
        owner.replace("-", "_"),
        owner.replace("_", "-"),
        owner.replace(" ", "-"),
        owner.replace(" ", "_"),
    ]
    candidates.extend(variations)
    
    case_variations = []
    for c in candidates:
        case_variations.extend([
            c,
            c.lower(),
            c.upper(),
            c.capitalize(),
            c.title(),
        ])
    
    candidates = list(dict.fromkeys(case_variations))
    
    if owner.lower() == "qwen":
        candidates.insert(0, "Qwen")
    elif owner.lower() == "deepseek" or owner.lower() == "deepseek-ai":
        candidates.insert(0, "deepseek-ai")
    
    return candidates


def search_models(owner="Qwen", page_size=100, filter_str=None):
    """
    Search and list models from ModelScope
    
    Args:
        owner: Owner or group name (e.g., "Qwen")
        page_size: Number of models per page
        filter_str: Filter string to match model names (case-insensitive)
    """
    api = HubApi()
    
    print(f"Listing models from owner: {owner}")
    if filter_str:
        print(f"Filtering models containing: '{filter_str}'")
    print("-" * 80)
    
    owner_candidates = generate_owner_candidates(owner)
    
    models = None
    used_owner = None
    
    for candidate in owner_candidates:
        try:
            result = api.list_models(
                owner_or_group=candidate,
                page_size=page_size
            )
            
            if result and 'Models' in result and result.get('Models'):
                models = result.get('Models', [])
                used_owner = candidate
                if candidate != owner:
                    print(f"Using owner: '{candidate}' (searched for '{owner}')")
                break
        except Exception:
            continue
    
    if not models:
        print(f"No models found for owner: {owner}")
        print("\nHint: Common owner patterns include:")
        print("  - Qwen (通义千问)")
        print("  - deepseek-ai (DeepSeek)")
        print("  - modelscope (ModelScope官方)")
        print("  - AI-ModelScope")
        return
    
    filtered_models = models
    
    if filter_str:
        filter_str_lower = filter_str.lower()
        filtered_models = [
            model for model in models 
            if filter_str_lower in model.get('Name', '').lower()
        ]
    
    if not filtered_models:
        if filter_str:
            print(f"No models found matching filter: '{filter_str}'")
            print(f"\nShowing all {len(models)} models instead:\n")
            for i, model in enumerate(models[:30], 1):
                print(f"{i:2d}. {model.get('Name', 'N/A')}")
            if len(models) > 30:
                print(f"... and {len(models) - 30} more")
        else:
            print("No models found.")
        return
    
    print(f"Found {len(filtered_models)} models:\n")
    
    for i, model in enumerate(filtered_models, 1):
        print(f"{i:2d}. {model.get('Name', 'N/A')}")
        print(f"    Model ID: {model.get('Path', '')}/{model.get('Name', 'N/A')}")
        print(f"    Downloads: {model.get('Downloads', 0)}")
        print()
    
    print(f"\n{'='*80}")
    print(f"To download any of these models, use:")
    print(f"python downloader.py --model_name \"<Model ID>\"")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Search models from ModelScope")
    parser.add_argument(
        "--owner",
        type=str,
        default="Qwen",
        help="Owner or group name (case-insensitive search, e.g., 'Qwen', 'deepseek-ai')"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Limit number of search results (default: 100)"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter string to match model names (case-insensitive, e.g., 'instruct', 'r1')"
    )
    
    args = parser.parse_args()
    
    search_models(args.owner, args.limit, args.filter)


if __name__ == "__main__":
    main()
