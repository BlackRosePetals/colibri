#!/usr/bin/env python3
"""Il motore GLM-5.3 con un'immagine, contro l'oracolo di transformers.

Il test sul testo prova la meta' testuale, quello sul tower prova la meta'
vision. Questo prova il punto in cui si incontrano: le patch passano per la
torre, gli embedding che ne escono sostituiscono le posizioni dei token
segnaposto, e il modello risponde. Un motore che prenda quegli embedding
nell'ordine sbagliato, o li allinei di una posizione, resta un modello che
parla: solo il confronto coi token dell'oracolo se ne accorge.

La traccia greedy si confronta fino a `greedy_exact_steps`. Oltre quel punto la
selezione dei pool dell'indexer e' passata da un pareggio: il punteggio passa
per una ReLU, i pool che non piacciono a nessuna testa valgono esattamente 0, e
quale di quelli entri nella selezione lo decide l'ordine interno di torch.topk,
che non e' specificato. La nostra regola (indice piu' basso) e' deterministica e
riproducibile; quella di torch cambia fra backend. Da li' in poi la traccia non
sarebbe un bersaglio legittimo per nessuna implementazione, nemmeno per una
seconda esecuzione di torch su un'altra macchina.
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
    prompt = ",".join(str(token) for token in reference["prompt"])
    grid_h, grid_w = reference["grid"]
    expected_forcing = reference["teacher_forcing"]
    expected_greedy = reference["greedy"]
    comparable = reference.get("greedy_exact_steps", len(expected_greedy))

    result = subprocess.run(
        [arguments.binary, "--model", str(arguments.fixture), "--ids", prompt,
         "--patches", str(arguments.fixture / "patches.f32"),
         "--grid", f"{grid_h}x{grid_w}",
         "--greedy", str(len(expected_greedy)), "--logits"],
        capture_output=True, text=True, check=True)

    lines = {line.split()[0]: line.split()[1:]
             for line in result.stdout.splitlines() if line.strip()}
    tokens = int(lines["vision_tokens"][0])
    forcing = [int(value) for value in lines["teacher_forcing"]]
    greedy = [int(value) for value in lines["greedy"]]
    logits = [float(value) for value in lines["last_logits"]]

    if tokens != reference["image_tokens"]:
        print(f"FAIL token immagine: {tokens}, attesi {reference['image_tokens']}")
        return 1
    if forcing != expected_forcing:
        print(f"FAIL teacher forcing\n  ottenuto: {forcing}\n  atteso:   {expected_forcing}")
        return 1
    if greedy[:comparable] != expected_greedy[:comparable]:
        print(f"FAIL greedy nei primi {comparable} passi\n"
              f"  ottenuto: {greedy[:comparable]}\n"
              f"  atteso:   {expected_greedy[:comparable]}")
        return 1
    worst = max(abs(a - b) for a, b in zip(logits, reference["last_logits"], strict=True))
    if worst > arguments.logit_tolerance:
        print(f"FAIL logit: max abs {worst:.3g} oltre {arguments.logit_tolerance:.3g}")
        return 1

    skipped = ("" if comparable == len(expected_greedy) else
               f", {len(expected_greedy) - comparable} passi greedy non "
               f"confrontabili (pareggio nella selezione dei pool)")
    print(f"PASS GLM-5.3 multimodale: {tokens} token immagine, "
          f"{len(forcing)} posizioni teacher-forced esatte, "
          f"{comparable} token greedy esatti, logit entro {worst:.3g}{skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
