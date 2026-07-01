"""Merge a trained adapter into a standalone model directory."""

from easylora import merge_adapter


def main() -> None:
    merged_dir = merge_adapter(
        base_model_name_or_path="meta-llama/Llama-3.2-1B",
        adapter_dir="./output/adapter",
        output_dir="./merged_model",
    )
    print(f"Merged model saved to: {merged_dir}")


if __name__ == "__main__":
    main()
