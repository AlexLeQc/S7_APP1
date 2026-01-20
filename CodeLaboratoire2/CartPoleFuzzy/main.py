# Université de Sherbrooke
# Code for Artificial Intelligence module
# Adapted by Audrey Corbeil Therrien, Simon Brodeur

# Source code
# Classic cart-pole system implemented by Rich Sutton et al.
# Copied from http://incompleteideas.net/sutton/book/code/pole.c
# permalink: https://perma.cc/C9ZM-652R

# NOTE : The print_state function of the FuzzyController may need
# to be updated with the latest version, if you encounter the error, the fix is
# available on github
# https://github.com/scikit-fuzzy/scikit-fuzzy/blob/master/skfuzzy/control/controlsystem.py
# Lines 514-572 from github replace lines 493-551 in the 0.4.2 2019 release

import time

import matplotlib.pyplot as plt
import skfuzzy as fuzz
from cartpole import *
from skfuzzy import control as ctrl


def createFuzzyController():
    # Variables avec plages réalistes pour le pendule
    # Angle ∈ [-0.21, +0.21] rad (limite d'échec du CartPole)
    angle = ctrl.Antecedent(np.linspace(-0.21, 0.21, 1000), "angle")

    # Force ∈ [-10, +10] N (valeurs réelles du CartPole)
    force = ctrl.Consequent(
        np.linspace(-10, 10, 1000), "force", defuzzify_method="centroid"
    )
    force.accumulation_method = np.fmax

    # Fonctions d'appartenance optimisées pour l'angle
    angle["negative"] = fuzz.trapmf(angle.universe, [-0.21, -0.21, -0.08, -0.02])
    angle["zero"] = fuzz.trimf(angle.universe, [-0.06, 0.0, 0.06])
    angle["positive"] = fuzz.trapmf(angle.universe, [0.02, 0.08, 0.21, 0.21])

    # Fonctions d'appartenance pour la force (avec plus de nuances)
    force["left_strong"] = fuzz.trimf(force.universe, [-10, -10, -5])
    force["left_weak"] = fuzz.trimf(force.universe, [-7, -2, 0])
    force["zero"] = fuzz.trimf(force.universe, [-1, 0, 1])
    force["right_weak"] = fuzz.trimf(force.universe, [0, 2, 7])
    force["right_strong"] = fuzz.trimf(force.universe, [5, 10, 10])

    # TODO: Define the rules.
    rules = []
    # Règles de contrôle proportionnel optimisées
    rules = [
        ctrl.Rule(angle["negative"], force["left_strong"]),
        ctrl.Rule(angle["zero"], force["zero"]),
        ctrl.Rule(angle["positive"], force["right_strong"]),
    ]

    # Conjunction (and_func) and disjunction (or_func) methods for rules:
    #     np.fmin
    #     np.fmax
    for rule in rules:
        rule.and_func = np.fmin
        rule.or_func = np.fmax

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    return sim


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    # Create the environment and fuzzy controller
    env = CartPoleEnv("human")
    fuzz_ctrl = createFuzzyController()

    # Display rules
    print("------------------------ RULES ------------------------")
    for rule in fuzz_ctrl.ctrl.rules:
        print(rule)
    print("-------------------------------------------------------")

    # Display fuzzy variables
    for var in fuzz_ctrl.ctrl.fuzzy_variables:
        var.view()
    plt.show()

    VERBOSE = False

    for episode in range(10):
        print("Episode no.%d" % (episode))
        env.reset()

        isSuccess = True
        action = np.array([0.0], dtype=np.float32)
        for _ in range(1000):
            env.render()
            time.sleep(0.01)

            # Execute the action
            observation, _, done, _ = env.step(action)
            if done:
                # End the episode
                isSuccess = False
                break

            # Select the next action based on the observation
            cartPosition, cartVelocity, poleAngle, poleVelocityAtTip = observation

            # Connecter l'angle réel du pendule au contrôleur flou
            fuzz_ctrl.input["angle"] = poleAngle

            fuzz_ctrl.compute()
            if VERBOSE:
                fuzz_ctrl.print_state()

            # Récupérer la force calculée par le contrôleur
            force = fuzz_ctrl.output["force"]

            action = np.array(force, dtype=np.float32).flatten()

    env.close()
