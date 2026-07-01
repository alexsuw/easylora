"""Evaluate an adapter and write a markdown report."""

from easylora.config import DataConfig, ModelConfig
from easylora.data.formatting import format_examples
from easylora.data.loaders import load_dataset_any
from easylora.eval import build_eval_report, compute_perplexity, generate_samples, save_eval_report
from easylora.lora.adapter import load_adapter
from easylora.utils.hf import load_tokenizer


def main() -> None:
    base_model = "meta-llama/Llama-3.2-1B"
    adapter_dir = "./output/adapter"
    dataset = "eval.jsonl"

    model_cfg = ModelConfig(base_model=base_model)
    tokenizer = load_tokenizer(model_cfg)
    model = load_adapter(base_model, adapter_dir)
    data_cfg = DataConfig(dataset_path=dataset, format="auto")
    formatted = format_examples(load_dataset_any(data_cfg), data_cfg, tokenizer)
    perplexity = compute_perplexity(model, tokenizer, formatted, max_samples=32)

    prompts = ["Explain LoRA in one sentence."]
    outputs = generate_samples(model, tokenizer, prompts)
    report = build_eval_report(
        base_model=base_model,
        adapter_dir=adapter_dir,
        dataset=dataset,
        perplexity=perplexity,
        generations=[
            {"prompt": prompt, "adapter_output": output}
            for prompt, output in zip(prompts, outputs, strict=True)
        ],
    )
    save_eval_report(report, "./output/eval_report.md")


if __name__ == "__main__":
    main()
