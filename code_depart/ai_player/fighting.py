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
            num_generations=500, mutation_prob=0.05, crossover_prob=0.95
        )
        self.ga.set_fit_fun(self.fitness_function)

        self.stagnation_counter = 0
        self.last_best_rounds = 0

    def fitness_function(self, *args):
        player_attributes = list(args)

        if self.player:
            self.player.set_attributes(player_attributes)

            rounds_won, fitness_value = self.monster.mock_fight(self.player)

            if rounds_won == 4:
                base_reward = 10000
            elif rounds_won == 3:
                base_reward = 1000
            elif rounds_won == 2:
                base_reward = 100
            else:
                base_reward = rounds_won * 10

            bonus = fitness_value * (1 + rounds_won * 0.5)

            score = base_reward + bonus

            extreme_penalty = sum(1 for attr in player_attributes if abs(attr) > 900)
            score -= extreme_penalty * 150

            return score

        return -1000

    def optimize_player_attributes(self, player):
        self.player = player
        self.ga.current_gen = 0
        self.stagnation_counter = 0
        self.last_best_rounds = 0

        print(f"AG: Optimisation contre monstre à {self.monster.rect}")

        for i in range(self.ga.num_generations):
            self.ga.decode_individuals()
            self.ga.eval_fit()

            best_index = np.argmax(self.ga.fitness)
            best_attrs = self.ga.cvalues[best_index]
            self.player.set_attributes(best_attrs)
            current_rounds, _ = self.monster.mock_fight(self.player)

            if current_rounds == self.last_best_rounds:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                self.last_best_rounds = current_rounds

            if self.stagnation_counter > 25 and current_rounds < 4:
                diversity_boost = (
                    np.random.randn(self.pop_size // 4, self.num_params) * 200
                )
                for j in range(self.pop_size // 4):
                    idx = np.random.randint(0, self.pop_size)
                    self.ga.cvalues[idx] += diversity_boost[j]
                    self.ga.cvalues[idx] = np.clip(self.ga.cvalues[idx], -1000, 1000)
                self.stagnation_counter = 0
                if i % 10 == 0:
                    print(
                        f"🔄 Injection de diversité (Gen {i}, {current_rounds} rounds)"
                    )

            if i % 10 == 0 or i == self.ga.num_generations - 1:
                self.ga.print_progress()
                print(f"   → Meilleur: {current_rounds} rounds gagnés")

            if current_rounds == 4:
                print(f"🎉 Solution parfaite trouvée à la génération {i}!")
                break

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


def get_optimal_attributes_for_monster(monster, player, pop_size=200, generations=500):
    fighter = Fighting(monster, pop_size=pop_size)
    fighter.ga.num_generations = generations
    optimal_attrs = fighter.optimize_player_attributes(player)
    return optimal_attrs
