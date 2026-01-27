import numpy as np
from Constants import NUM_ATTRIBUTES

import ai_player.genetic as genetic


class Fighting:
    def __init__(self, monster, pop_size=500, nbits=12):
        self.monster = monster
        self.player = None
        self.num_params = NUM_ATTRIBUTES
        self.pop_size = pop_size
        self.nbits = nbits

        self.stagnation_counter = 0
        self.last_best_rounds = 0

    def optimize_player_attributes(self, player):
        self.player = player

        self.ga = genetic.Genetic(
            self.num_params, self.pop_size, self.nbits, self.player
        )
        self.ga.init_pop()

        # ******Mutation modifier suite à la défense de la problématique*****
        self.ga.set_sim_parameters(
            num_generations=1000, mutation_prob=0.01, crossover_prob=0.8
        )
        self.ga.set_fit_fun(self.monster.mock_fight)

        print(f"AG: Optimisation déterministe contre monstre à {self.monster.rect}")

        for i in range(self.ga.num_generations):
            self.ga.decode_individuals()
            self.ga.eval_fit()

            best_index = np.argmax(self.ga.fitness)
            best_attrs = self.ga.cvalues[best_index]

            self.player.set_attributes([int(round(a)) for a in best_attrs])
            current_rounds, _ = self.monster.mock_fight(self.player)

            if i % 50 == 0 or i == self.ga.num_generations - 1:
                self.ga.print_progress()
                print(f"   → Stats actuelles: {current_rounds}/4 rounds")

            if current_rounds == 4:
                break

            if i < self.ga.num_generations - 1:
                self.ga.new_gen()

        optimal_attrs = self.ga.get_best_individual()

        return optimal_attrs
