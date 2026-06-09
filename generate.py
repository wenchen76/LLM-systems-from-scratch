import argparse
import sys
import time

import torch
from llm_core.engine import LLMEngine, SamplingParams
from llm_core.model import TransformerLM
from tokenizer.BPETokenizer import Tokenizer


def run_batch(engine, tokenizer, prompts, sampling):
    """Submit prompts as concurrent requests and stream completions as they finish."""
    id_to_prompt = {}
    for prompt in prompts:
        request_id = engine.add_request(tokenizer.encode(prompt), sampling)
        id_to_prompt[request_id] = prompt

    print(f"submitted {len(prompts)} request(s); generating concurrently...\n")
    start = time.perf_counter()
    steps = generated = 0
    while engine.has_work():
        finished = engine.step()
        steps += 1
        generated += len(engine.running) + len(finished)  # one token per request that ran this step
        elapsed = max(time.perf_counter() - start, 1e-9)
        sys.stdout.write(f"\r  running={len(engine.running):2d}  step={steps:4d}  "
                         f"{generated / elapsed:7.1f} tok/s")
        sys.stdout.flush()
        for req in finished:  # requests finish at different times -> print as they retire
            sys.stdout.write("\r" + " " * 48 + "\r")  # clear the status line
            print(f"  [#{req.request_id}] {id_to_prompt[req.request_id]!r}")
            print(f"      -> {tokenizer.decode(req.output_ids)}")
            print(f"      ({len(req.output_ids)} tokens, {req.finish_reason})\n")
    sys.stdout.write("\r" + " " * 48 + "\r")
    print(f"done: {len(prompts)} requests in {steps} steps, {time.perf_counter() - start:.2f}s")


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
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Disable FlashAttention (PyTorch SDPA); enabled by default")
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

    # FlashAttention (PyTorch SDPA) on by default: faster and lower-memory for
    # both prefill and cached decode, with identical results. --no-flash-attn opts out.
    use_flash_attn = not args.no_flash_attn
    for module in model.modules():
        if hasattr(module, "use_flash_attn"):
            module.use_flash_attn = use_flash_attn

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=["<|endoftext|>"])
    eos_token_id = tokenizer.encode("<|endoftext|>")[0] if "<|endoftext|>" in tokenizer.special_tokens else None

    engine = LLMEngine(model, device=device)
    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=eos_token_id,
    )

    print("Model loaded (continuous-batching engine).")
    print("Enter one prompt per line; submit the batch with a blank line.")
    print("Prompts in a batch generate concurrently. Ctrl+C / Ctrl+D to quit.\n")
    while True:
        prompts = []
        try:
            while True:
                line = input(f"[{len(prompts)}] + " if prompts else "> ")
                if line == "":
                    break
                prompts.append(line)
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not prompts:
            continue

        run_batch(engine, tokenizer, prompts, sampling)
        print()


if __name__ == "__main__":
    main()
