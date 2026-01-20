import numpy as np
from Constants import NUM_ATTRIBUTES

import ai_player.genetic as genetic


class Fighting:
    def __init__(self, monster, pop_size=200, nbits=16):
        self.monster = monster
        self.player = None

        self.num_params = NUM_ATTRIBUTES
        self.pop_size = pop_size
        self.nbits = nbits

        self.ga = genetic.Genetic(self.num_params, self.pop_size, self.nbits)
        self.ga.init_pop()

        self.ga.set_sim_parameters(
            num_generations=400, mutation_prob=0.03, crossover_prob=0.95
        )
        self.ga.set_fit_fun(self.fitness_function)

    def fitness_function(self, *args):
        player_attributes = list(args)

        if self.player:
            self.player.set_attributes(player_attributes)

            rounds_won, fitness_value = self.monster.mock_fight(self.player)

            base_reward = (rounds_won**3) * 100
            bonus = fitness_value * (rounds_won + 1) * 0.1

            score = base_reward + bonus

            extreme_penalty = sum(1 for attr in player_attributes if abs(attr) > 900)
            score -= extreme_penalty * 100

            return score

        return -1000

    def optimize_player_attributes(self, player):
        self.player = player

        self.ga.current_gen = 0

        print(f"AG: Optimisation contre monstre à {self.monster.rect}")

        for i in range(self.ga.num_generations):
            self.ga.decode_individuals()
            self.ga.eval_fit()

            if i % 10 == 0 or i == self.ga.num_generations - 1:
                self.ga.print_progress()

            if i < self.ga.num_generations - 1:
                self.ga.new_gen()

        best_fitness = self.ga.bestIndividualFitness
        print(f"Meilleurs attributs trouvés (fitness: {best_fitness:.2f})")

        best_index = np.where(self.ga.fitness.flatten() == best_fitness)[0]
        if len(best_index) > 0:
            best_attributes = self.ga.cvalues[best_index[0]]
        else:
            best_index = np.argmax(self.ga.fitness)
            best_attributes = self.ga.cvalues[best_index]

        optimal_attrs = [int(attr) for attr in best_attributes]

        self.player.set_attributes(optimal_attrs)
        rounds_won, _ = self.monster.mock_fight(self.player)
        print(f"Verification: {rounds_won} rounds gagnés sur 4")

        return optimal_attrs
