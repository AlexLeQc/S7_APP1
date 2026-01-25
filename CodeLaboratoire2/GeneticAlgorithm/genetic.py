# Helper class for genetic algorithms
# Copyright (c) 2018, Audrey Corbeil Therrien, adapted from Simon Brodeur
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA,
# OR PROFITS OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Université de Sherbrooke
# Code for Artificial Intelligence module
# Adapted by Audrey Corbeil Therrien for Artificial Intelligence module
import numpy as np


class Genetic:
    num_params = 0
    pop_size = 0
    nbits = 0
    population = []

    def __init__(self, num_params, pop_size, nbits):
        # Input:
        # - NUMPARAMS, the number of parameters to optimize.
        # - POPSIZE, the population size.
        # - NBITS, the number of bits per indivual used for encoding.
        self.num_params = num_params
        self.pop_size = pop_size
        self.nbits = nbits
        self.fitness = np.zeros((self.pop_size, 1))
        self.fit_fun = np.zeros
        self.cvalues = np.zeros((self.pop_size, num_params))
        self.num_generations = 1
        self.mutation_prob = 0
        self.crossover_prob = 0
        self.bestIndividual = []
        self.bestIndividualFitness = -1e10
        self.maxFitnessRecord = np.zeros((self.num_generations,))
        self.overallMaxFitnessRecord = np.zeros((self.num_generations,))
        self.avgMaxFitnessRecord = np.zeros((self.num_generations,))
        self.current_gen = 0
        self.crossover_modulo = 0

    def init_pop(self):
        # Initialize the population as a matrix, where each individual is a binary string.
        # Output:
        # - POPULATION, a binary matrix whose rows correspond to encoded individuals.
        self.population = np.random.randint(
            0, 2, (self.pop_size, self.num_params * self.nbits)
        )

    def set_fit_fun(self, fun):
        # Set the fitness function
        self.fit_fun = fun

    def set_crossover_modulo(self, modulo):
        # Set the fitness function
        self.crossover_modulo = modulo

    def set_sim_parameters(self, num_generations, mutation_prob, crossover_prob):
        # set the simulation/evolution parameters to execute the optimization
        # initialize the result matrices
        self.num_generations = num_generations
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.bestIndividual = []
        self.bestIndividualFitness = -1e10
        self.maxFitnessRecord = np.zeros((num_generations,))
        self.overallMaxFitnessRecord = np.zeros((num_generations,))
        self.avgMaxFitnessRecord = np.zeros((num_generations,))
        self.current_gen = 0

    def eval_fit(self):
        # Evaluate the fitness function
        # Record the best individual and average of the current generation
        # WARNING, number of arguments need to be adjusted if fitness function changes
        self.fitness = self.fit_fun(self.cvalues[:, 0], self.cvalues[:, 1])
        if np.max(self.fitness) > self.bestIndividualFitness:
            self.bestIndividualFitness = np.max(self.fitness)
            self.bestIndividual = self.population[self.fitness == np.max(self.fitness)][
                0
            ]
        self.maxFitnessRecord[self.current_gen] = np.max(self.fitness)
        self.overallMaxFitnessRecord[self.current_gen] = self.bestIndividualFitness
        self.avgMaxFitnessRecord[self.current_gen] = np.mean(self.fitness)

    def print_progress(self):
        # Prints the results of the current generation in the console
        print(
            "Generation no.%d: best fitness is %f, average is %f"
            % (
                self.current_gen,
                self.maxFitnessRecord[self.current_gen],
                self.avgMaxFitnessRecord[self.current_gen],
            )
        )
        print("Overall best fitness is %f" % self.bestIndividualFitness)

    def get_best_individual(self):
        # Prints the best individual for all of the simulated generations
        # TODO : Decode individual for better readability
        return self.bestIndividual

    def encode_individuals(self):
        # Étape 1: Mise à l'échelle des valeurs de [-3,3] vers [0,1]
        scaled_cvalues = (self.cvalues + 3.0) / 6.0

        # Étape 2: Encoder chaque paramètre séparément
        encoded_params = []
        for param_idx in range(self.num_params):
            param_values = scaled_cvalues[:, param_idx]
            encoded_param = ufloat2bin(param_values, self.nbits)
            encoded_params.append(encoded_param)

        # Étape 3: Concaténer tous les paramètres pour chaque individu
        self.population = np.hstack(encoded_params)

    def decode_individuals(self):
        # Étape 1: Diviser la chaîne binaire en paramètres
        bits_per_param = self.nbits
        decoded_params = []

        for param_idx in range(self.num_params):
            start_bit = param_idx * bits_per_param
            end_bit = (param_idx + 1) * bits_per_param
            param_bits = self.population[:, start_bit:end_bit]

            # Étape 2: Décoder et remettre à l'échelle [-3, 3]
            decoded_param = bin2ufloat(param_bits, self.nbits)
            scaled_param = decoded_param * 6.0 - 3.0
            decoded_params.append(scaled_param.reshape(-1, 1))

        self.cvalues = np.hstack(decoded_params)

    def doSelection(self):
        num_pairs = self.pop_size // 2

        selected_indices_1 = np.zeros(num_pairs, dtype=int)
        selected_indices_2 = np.zeros(num_pairs, dtype=int)

        # Sélection par tournoi
        tournament_size = 3

        for i in range(num_pairs):
            # Sélectionner premier parent
            tournament_indices = np.random.choice(
                self.pop_size, tournament_size, replace=False
            )
            tournament_fitness = self.fitness[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected_indices_1[i] = winner_idx

            # Sélectionner deuxième parent
            tournament_indices = np.random.choice(
                self.pop_size, tournament_size, replace=False
            )
            tournament_fitness = self.fitness[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected_indices_2[i] = winner_idx

        return [
            self.population[selected_indices_1, :],
            self.population[selected_indices_2, :],
        ]

    def doCrossover(self, pairs):
        ind1 = pairs[0]
        ind2 = pairs[1]
        num_pairs = ind1.shape[0]

        offspring = np.zeros((self.pop_size, self.num_params * self.nbits))

        for i in range(num_pairs):
            if np.random.rand() < self.crossover_prob:
                # Effectuer croisement
                if self.crossover_modulo > 0:
                    # Point de croisement contraint
                    possible_points = np.arange(
                        self.crossover_modulo,
                        self.num_params * self.nbits,
                        self.crossover_modulo,
                    )
                    if len(possible_points) > 0:
                        crossover_point = np.random.choice(possible_points)
                    else:
                        crossover_point = np.random.randint(
                            1, self.num_params * self.nbits
                        )
                else:
                    # Point de croisement aléatoire
                    crossover_point = np.random.randint(1, self.num_params * self.nbits)

                # Créer descendance
                offspring[i * 2, :] = np.concatenate(
                    [ind1[i, :crossover_point], ind2[i, crossover_point:]]
                )
                offspring[i * 2 + 1, :] = np.concatenate(
                    [ind2[i, :crossover_point], ind1[i, crossover_point:]]
                )
            else:
                # Pas de croisement, copier parents
                offspring[i * 2, :] = ind1[i, :]
                offspring[i * 2 + 1, :] = ind2[i, :]
        return offspring

    def doMutation(self):
        # Créer masque de mutation
        mutation_mask = (
            np.random.rand(self.pop_size, self.num_params * self.nbits)
            < self.mutation_prob
        )

        # Appliquer mutation (inverser bits)
        self.population = np.logical_xor(
            self.population.astype(bool), mutation_mask
        ).astype(int)

    def new_gen(self):
        # Perform a the pair selection, crossover and mutation and
        # generate a new population for the next generation.
        # Input:
        # - POPULATION, the binary matrix representing the population. Each row is an individual.
        # Output:
        # - POPULATION, the new population.
        pairs = self.doSelection()
        self.population = self.doCrossover(pairs)
        self.doMutation()
        self.current_gen += 1

    def run(self, show_progress=False, save_path=None):
        """Run the GA for the configured number of generations and return a
        serializable results dictionary. If `save_path` is provided, the
        dictionary is written to that JSON file."""
        for i in range(self.num_generations):
            self.decode_individuals()
            self.eval_fit()
            if show_progress:
                self.print_progress()
            if i < self.num_generations - 1:
                self.new_gen()

        results = {
            "params": {
                "num_params": int(self.num_params),
                "pop_size": int(self.pop_size),
                "nbits": int(self.nbits),
                "num_generations": int(self.num_generations),
                "mutation_prob": float(self.mutation_prob),
                "crossover_prob": float(self.crossover_prob),
            },
            "maxFitnessRecord": list(np.array(self.maxFitnessRecord).astype(float)),
            "overallMaxFitnessRecord": list(
                np.array(self.overallMaxFitnessRecord).astype(float)
            ),
            "avgMaxFitnessRecord": list(
                np.array(self.avgMaxFitnessRecord).astype(float)
            ),
            "bestIndividual": (
                self.bestIndividual.tolist()
                if hasattr(self.bestIndividual, "tolist")
                else self.bestIndividual
            ),
            "bestIndividualFitness": float(self.bestIndividualFitness),
            "final_cvalues": [list(map(float, row)) for row in np.array(self.cvalues)],
            "final_population": [
                list(map(int, row)) for row in np.array(self.population)
            ],
            "fitness": list(np.array(self.fitness).flatten().astype(float)),
        }

        if save_path:
            self.save_results(save_path, results)

        return results

    def save_results(self, path, results=None):
        import json
        import os

        if results is None:
            results = {
                "params": {
                    "num_params": int(self.num_params),
                    "pop_size": int(self.pop_size),
                    "nbits": int(self.nbits),
                    "num_generations": int(self.num_generations),
                    "mutation_prob": float(self.mutation_prob),
                    "crossover_prob": float(self.crossover_prob),
                },
                "final_cvalues": [
                    list(map(float, row)) for row in np.array(self.cvalues)
                ],
                "final_population": [
                    list(map(int, row)) for row in np.array(self.population)
                ],
                "fitness": list(np.array(self.fitness).flatten().astype(float)),
            }

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


# Binary-Float conversion functions
# usage: [BVALUE] = ufloat2bin(CVALUE, NBITS)
#
# Convert floating point values into a binary vector
#
# Input:
# - CVALUE, a scalar or vector of continuous values representing the parameters.
#   The values must be a real non-negative float in the interval [0,1]!
# - NBITS, the number of bits used for encoding.
#
# Output:
# - BVALUE, the binary representation of the continuous value. If CVALUES was a vector,
#   the output is a matrix whose rows correspond to the elements of CVALUES.
def ufloat2bin(cvalue, nbits):
    if nbits > 64:
        raise Exception("Maximum number of bits limited to 64")
    ivalue = np.round(cvalue * (2**nbits - 1)).astype(np.uint64)
    bvalue = np.zeros((len(cvalue), nbits))

    # Overflow
    bvalue[ivalue > 2**nbits - 1] = np.ones((nbits,))

    # Underflow
    bvalue[ivalue < 0] = np.zeros((nbits,))

    bitmask = (2 ** np.arange(nbits)).astype(np.uint64)
    bvalue[np.logical_and(ivalue >= 0, ivalue <= 2**nbits - 1)] = (
        np.bitwise_and(
            np.tile(ivalue[:, np.newaxis], (1, nbits)),
            np.tile(bitmask[np.newaxis, :], (len(cvalue), 1)),
        )
        != 0
    )
    return bvalue


# usage: [CVALUE] = bin2ufloat(BVALUE, NBITS)
#
# Convert a binary vector into floating point values
#
# Input:
# - BVALUE, the binary representation of the continuous values. Can be a single vector or a matrix whose
#   rows represent independent encoded values.
#   The values must be a real non-negative float in the interval [0,1]!
# - NBITS, the number of bits used for encoding.
#
# Output:
# - CVALUE, a scalar or vector of continuous values representing the parameters.
#   the output is a matrix whose rows correspond to the elements of CVALUES.
#
def bin2ufloat(bvalue, nbits):
    if nbits > 64:
        raise Exception("Maximum number of bits limited to 64")
    ivalue = np.sum(bvalue * (2 ** np.arange(nbits)[np.newaxis, :]), axis=-1)
    cvalue = ivalue / (2**nbits - 1)
    return cvalue
