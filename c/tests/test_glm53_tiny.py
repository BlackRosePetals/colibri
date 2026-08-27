#!/usr/bin/env python3
"""Il motore GLM-5.3 contro l'oracolo tiny generato da transformers.

Confronta cio' che un utente vede davvero -- i token -- e non solo i numeri
interni: teacher forcing su ogni posizione, generazione greedy, e i logit
dell'ultima posizione entro una tolleranza stretta. Un motore puo' avere logit
quasi giusti e scegliere comunque il token sbagliato, quindi entrambi contano.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--logit-tolerance", type=float, default=2e-5)
    arguments = parser.parse_args()

    reference = json.loads((arguments.fixture / "ref.json").read_text())
    prompt = ",".join(str(token) for token in reference["prompt_ids"])
    expected_forcing = reference["teacher_forcing_ids"]
    expected_greedy = reference["greedy_new_ids"]

    result = subprocess.run(
        [arguments.binary, "--model", str(arguments.fixture), "--ids", prompt,
         "--greedy", str(len(expected_greedy)), "--logits"],
        capture_output=True, text=True, check=True)

    lines = {line.split()[0]: line.split()[1:] for line in result.stdout.splitlines() if line.strip()}
    forcing = [int(value) for value in lines["teacher_forcing"]]
    greedy = [int(value) for value in lines["greedy"]]
    logits = [float(value) for value in lines["last_logits"]]

    if forcing != expected_forcing:
        print(f"FAIL teacher forcing\n  ottenuto: {forcing}\n  atteso:   {expected_forcing}")
        return 1
    if greedy != expected_greedy:
        print(f"FAIL greedy\n  ottenuto: {greedy}\n  atteso:   {expected_greedy}")
        return 1
    worst = max(abs(a - b) for a, b in zip(logits, reference["last_logits"], strict=True))
    if worst > arguments.logit_tolerance:
        print(f"FAIL logit: max abs {worst:.3g} oltre {arguments.logit_tolerance:.3g}")
        return 1

    print(f"PASS GLM-5.3 tiny: {len(forcing)} posizioni teacher-forced esatte, "
          f"{len(greedy)} token greedy esatti, logit entro {worst:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
