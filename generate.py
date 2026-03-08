import argparse
import torch
from llm_core.model import TransformerLM
from tokenizer.BPETokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text from a pretrained model")
    parser.add_argument("--config", type=str, default="model_config.json",
                        help="Path to model config JSON file (default: model_config.json)")
    parser.add_argument("--checkpoint", type=str, default="ckpt_final.pt",
                        help="Path to model checkpoint file (default: ckpt_final.pt)")
    parser.add_argument("--vocab", type=str, default="vocab.json",
                        help="Path to vocab file (default: vocab.json)")
    parser.add_argument("--merges", type=str, default="merges.json",
                        help="Path to merges file (default: merges.json)")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k sampling")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda)")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = TransformerLM.from_pretrained(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    ).to(device)
    model.eval()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=["<|endoftext|>"])
    eos_token_id = tokenizer.encode("<|endoftext|>")[0] if "<|endoftext|>" in tokenizer.special_tokens else None

    print("Model loaded. Type your prompt (Ctrl+C to quit):\n")
    while True:
        try:
            prompt = input("> ")
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not prompt:
            continue

        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=eos_token_id,
        )
        print(tokenizer.decode(output_ids.squeeze().tolist()))
        print()


if __name__ == "__main__":
    main()
