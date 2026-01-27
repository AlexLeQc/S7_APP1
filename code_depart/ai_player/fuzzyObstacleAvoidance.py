import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyObstacleAvoidance:
    def __init__(self):
        self.simulation = self._setup_fuzzy_system()

    def _setup_fuzzy_system(self):
        distance = ctrl.Antecedent(np.linspace(0, 60, 1000), "Distance")
        angle = ctrl.Antecedent(np.linspace(-90, 90, 1000), "Angle")
        steering = ctrl.Consequent(
            np.linspace(-180, 180, 360), "Steering", defuzzify_method="som"
        )
        steering.accumulation_method = np.fmax

        # Distance membership
        distance["Near"] = fuzz.trapmf(distance.universe, [0, 0, 5, 20])
        distance["Medium"] = fuzz.trimf(distance.universe, [5, 10, 40])
        distance["Far"] = fuzz.trapmf(distance.universe, [35, 45, 60, 60])

        # Angle membership
        angle["HardLeft"] = fuzz.trapmf(angle.universe, [-180, -180, -90, -45])
        angle["SoftLeft"] = fuzz.trimf(angle.universe, [-60, -30, 0])
        angle["Center"] = fuzz.trimf(angle.universe, [-20, 0, 20])
        angle["SoftRight"] = fuzz.trimf(angle.universe, [0, 30, 60])
        angle["HardRight"] = fuzz.trapmf(angle.universe, [45, 90, 180, 180])

        # Steering membership
        steering["HardLeft"] = fuzz.trimf(steering.universe, [-90, -90, -45])
        steering["SoftLeft"] = fuzz.trimf(steering.universe, [-60, -30, 0])
        steering["None"] = fuzz.trimf(steering.universe, [-10, 0, 10])
        steering["SoftRight"] = fuzz.trimf(steering.universe, [0, 30, 60])
        steering["HardRight"] = fuzz.trimf(steering.universe, [45, 90, 90])
        steering["BackLeft"] = fuzz.trimf(steering.universe, [-180, -150, -120])
        steering["BackRight"] = fuzz.trimf(steering.universe, [120, 150, 180])

        # Rules
        rules = [
            ctrl.Rule(distance["Far"], steering["None"]),
            ctrl.Rule(distance["Medium"] & angle["HardLeft"], steering["SoftRight"]),
            ctrl.Rule(distance["Medium"] & angle["SoftLeft"], steering["SoftRight"]),
            ctrl.Rule(distance["Medium"] & angle["Center"], steering["BackRight"]),
            ctrl.Rule(distance["Medium"] & angle["SoftRight"], steering["SoftLeft"]),
            ctrl.Rule(distance["Medium"] & angle["HardRight"], steering["SoftLeft"]),
            ctrl.Rule(distance["Near"] & angle["HardLeft"], steering["None"]),
            ctrl.Rule(distance["Near"] & angle["SoftLeft"], steering["HardRight"]),
            ctrl.Rule(distance["Near"] & angle["SoftRight"], steering["HardLeft"]),
            ctrl.Rule(distance["Near"] & angle["HardRight"], steering["None"]),
        ]

        for r in rules:
            r.and_func = np.fmin
            r.or_func = np.fmax

        system = ctrl.ControlSystem(rules)
        return ctrl.ControlSystemSimulation(system)

    def compute_steering(self, distance_to_obs, angle_to_obs):
        self.simulation.input["Distance"] = distance_to_obs
        self.simulation.input["Angle"] = angle_to_obs
        self.simulation.compute()
        return self.simulation.output["Steering"]
