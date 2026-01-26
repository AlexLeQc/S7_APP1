"""
Réalisé avec l'asistance de ChatGPT
Mini script de test pour le système expert "porte".

Usage:
    python -m code_depart.test_doors

Ce script parcourt `Door.STATES`, appelle `SolvingDoors.solve_door`
et affiche la clé attendue (dans `Door.KEYS`) et la clé retournée
par le moteur Prolog.

Remarque: ce script nécessite `swiplserver` et SWI-Prolog installés
si vous voulez tester le solveur Prolog réel.
"""

import os
import sys

# Assurer que le workspace racine est dans le path pour importer modules
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Door import Door

try:
    from ai_player.systeme_expert import SolvingDoors
except Exception as e:
    SolvingDoors = None
    print("Attention: impossible d'importer SolvingDoors:", e)


def run_tests():
    print("Lancement des tests sur Door.STATES\n")

    solver = None
    if SolvingDoors is not None:
        try:
            solver = SolvingDoors()
        except Exception as e:
            print("Erreur lors de l'initialisation de SolvingDoors:", e)
            solver = None
    results = []

    for i, state in enumerate(Door.STATES):
        expected = Door.KEYS[i]
        print(f"Test {i + 1}: state = {state}")

        prolog_result = None
        if solver is not None:
            try:
                prolog_result = solver.solve_door(state)
            except Exception as e:
                prolog_result = f"ERROR: {e}"

        ok = prolog_result == expected

        # Calculer le nombre de cristaux (exclure la serrure)
        crystals_count = sum(1 for x in state[1:] if x != "")

        results.append(
            {
                "test": i + 1,
                "state": state,
                "expected": expected,
                "prolog": prolog_result,
                "ok": ok,
                "crystals": crystals_count,
            }
        )

        print(f" - Expected (Door.KEYS): {expected}")
        print(f" - Prolog result         : {prolog_result}")
        print(f" - OK?                   : {ok}")
        print("-" * 50)

    # Résumé
    total = len(results)
    correct = sum(1 for r in results if r["ok"])
    pct = (correct / total * 100) if total > 0 else 0.0

    print("\n=== Résumé des tests ===")
    print(f"Total tests: {total}")
    print(f"Succès     : {correct} ({pct:.1f}%)")

    # Résumé par nombre de cristaux
    groups = {}
    for r in results:
        groups.setdefault(r["crystals"], []).append(r)

    for n in sorted(groups.keys()):
        grp = groups[n]
        t = len(grp)
        c = sum(1 for r in grp if r["ok"])
        p = (c / t * 100) if t > 0 else 0.0
        print(f"{n} cristaux : {c}/{t} ({p:.1f}%)")

    # Lister les échecs si existants
    failed = [r for r in results if not r["ok"]]
    if failed:
        print("\nÉchecs détaillés :")
        for r in failed:
            print(
                f"Test {r['test']}: crystals={r['crystals']}, expected={r['expected']}, got={r['prolog']}, state={r['state']}"
            )


if __name__ == "__main__":
    run_tests()
