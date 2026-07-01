"""Minimal config-driven LoRA training."""

from easylora import DataConfig, ModelConfig, TrainConfig, train


def main() -> None:
    config = TrainConfig(
        model=ModelConfig(base_model="meta-llama/Llama-3.2-1B"),
        data=DataConfig(dataset_name="tatsu-lab/alpaca", format="auto"),
    )
    artifacts = train(config)
    print(f"Adapter saved to: {artifacts.adapter_dir}")


if __name__ == "__main__":
    main()
