#!/usr/bin/env python3
"""07_codeql_verify.py — CodeQL static analysis for CWE detection in code LLM outputs."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import load_prompts
from src.evaluate import evaluate_refusal_rate


def extract_code_blocks(text: str, language: str) -> list:
    """Extract code blocks from generated text."""
    import re
    pattern = rf"```{language}\n(.*?)\n```"
    return re.findall(pattern, text, re.DOTALL)


def run_codeql_analysis(code: str, language: str, codeql_path: str, query_pack: str) -> dict:
    """Run CodeQL analysis on a code snippet."""
    with tempfile.TemporaryDirectory() as tmpdir:
        code_file = Path(tmpdir) / f"test.{language}"
        code_file.write_text(code)

        db_dir = Path(tmpdir) / "codeql-db"
        result_file = Path(tmpdir) / "results.sarif"

        # Create database
        subprocess.run([
            codeql_path, "database", "create", str(db_dir),
            f"--language={language}", f"--source-root={tmpdir}"
        ], capture_output=True, check=False)

        # Run queries
        subprocess.run([
            codeql_path, "database", "analyze", str(db_dir),
            query_pack, f"--format=sarif-latest", f"--output={result_file}"
        ], capture_output=True, check=False)

        if result_file.exists():
            import json
            with open(result_file) as f:
                return json.load(f)
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--prompts", required=True, help="CWE harmful prompts file")
    parser.add_argument("--language", default="python", choices=["python", "javascript", "go", "java", "cpp"])
    parser.add_argument("--codeql-path", default="codeql")
    parser.add_argument("--query-pack", help="CodeQL query pack path (default: auto-detect)")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", "-o", help="Write JSON results")
    args = parser.parse_args()

    # Auto-detect query pack
    if args.query_pack is None:
        # Try to find standard CodeQL queries
        import os
        codeql_home = os.environ.get("CODEQL_HOME", "")
        if codeql_home:
            args.query_pack = f"{codeql_home}/{args.language}-queries"
        else:
            args.query_pack = f"codeql/{args.language}-queries"

    print(f"Loading model: {args.model}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=dtype, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    prompts = load_prompts(args.prompts, args.max_samples)
    print(f"Evaluating {len(prompts)} CWE prompts...")

    results = {
        "model": args.model,
        "language": args.language,
        "query_pack": args.query_pack,
        "prompts": [],
    }

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt[:80]}...")

        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated[:, input_len:]
        text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

        # Extract code blocks
        code_blocks = extract_code_blocks(text, args.language)

        cwe_findings = []
        for code in code_blocks:
            analysis = run_codeql_analysis(code, args.language, args.codeql_path, args.query_pack)
            if analysis.get("runs"):
                for run in analysis["runs"]:
                    for result in run.get("results", []):
                        cwe_findings.append({
                            "rule_id": result.get("ruleId"),
                            "message": result.get("message", {}).get("text"),
                            "locations": result.get("locations"),
                        })

        results["prompts"].append({
            "prompt": prompt,
            "response": text,
            "code_blocks_found": len(code_blocks),
            "cwe_findings": cwe_findings,
        })

        print(f"  Code blocks: {len(code_blocks)}, CWE findings: {len(cwe_findings)}")

        del inputs, generated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    total_findings = sum(len(p["cwe_findings"]) for p in results["prompts"])
    print(f"\nTotal CWE findings across {len(prompts)} prompts: {total_findings}")


if __name__ == "__main__":
    main()