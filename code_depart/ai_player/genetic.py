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
from Constants import MAX_ATTRIBUTE


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
        new_fitness = np.zeros((self.pop_size, 1))

        for i in range(self.pop_size):
            new_fitness[i] = self.fit_fun(*self.cvalues[i, :])

        self.fitness = new_fitness

        if np.max(self.fitness) > self.bestIndividualFitness:
            self.bestIndividualFitness = np.max(self.fitness)
            best_mask = (self.fitness == np.max(self.fitness)).flatten()
            self.bestIndividual = self.population[best_mask][0]

        self.maxFitnessRecord[self.current_gen] = np.max(self.fitness)
        self.overallMaxFitnessRecord[self.current_gen] = self.bestIndividualFitness
        self.avgMaxFitnessRecord[self.current_gen] = np.mean(self.fitness)

    def print_progress(self):
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
        # Encode the population from a vector of continuous values to a binary string.
        # Input:
        # - CVALUES, a vector of continuous values representing the parameters.
        # - NBITS, the number of bits per indivual used for encoding.
        # Output:
        # - POPULATION, a binary matrix with each row encoding an individual.
        # TODO: encode individuals into binary vectors
        scaled_cvalues = (self.cvalues + MAX_ATTRIBUTE) / (2 * MAX_ATTRIBUTE)
        encoded_params = []
        for param_idx in range(self.num_params):
            param_values = scaled_cvalues[:, param_idx]
            encoded_param = ufloat2bin(param_values, self.nbits)
            encoded_params.append(encoded_param)

        # Concatène tous les paramètres
        self.population = np.hstack(encoded_params)

    def decode_individuals(self):
        # Decode an individual from a binary string to a vector of continuous values.
        # Input:
        # - POPULATION, a binary matrix with each row encoding an individual.
        # - NUMPARAMS, the number of parameters for an individual.
        # Output:
        # - CVALUES, a vector of continuous values representing the parameters.
        # TODO: decode individuals from binary vectors
        bits_per_param = self.nbits
        decoded_params = []

        for param_idx in range(self.num_params):
            start_bit = param_idx * bits_per_param
            end_bit = (param_idx + 1) * bits_per_param
            param_bits = self.population[:, start_bit:end_bit]

            # Décode et remet à l'échelle [-MAX_ATTRIBUTE, MAX_ATTRIBUTE]
            decoded_param = bin2ufloat(param_bits, self.nbits)
            scaled_param = decoded_param * (2 * MAX_ATTRIBUTE) - MAX_ATTRIBUTE
            decoded_params.append(scaled_param.reshape(-1, 1))

        self.cvalues = np.hstack(decoded_params)

    def doSelection(self):
        # Select pairs of individuals from the population.
        # Input:
        # - POPULATION, the binary matrix representing the population. Each row is an individual.
        # - FITNESS, a vector of fitness values for the population.
        # - NUMPAIRS, the number of pairs of individual to generate.
        # Output:
        # - PAIRS, a list of two ndarrays [IND1 IND2]  each encoding one member of the pair
        # TODO: select pairs of individual in the population
        num_pairs = self.pop_size // 2

        selected_indices_1 = np.zeros(num_pairs, dtype=int)
        selected_indices_2 = np.zeros(num_pairs, dtype=int)

        tournament_size = 3

        for i in range(num_pairs):
            tournament_indices = np.random.choice(
                self.pop_size, tournament_size, replace=False
            )
            tournament_fitness = self.fitness[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected_indices_1[i] = winner_idx

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
        # Perform a crossover operation between two individuals, with a given probability
        # and constraint on the cutting point.
        # Input:
        # - PAIRS, a list of two ndarrays [IND1 IND2] each encoding one member of the pair
        # - CROSSOVER_PROB, the crossover probability.
        # - CROSSOVER_MODULO, a modulo-constraint on the cutting point. For example, to only allow cutting
        #   every 4 bits, set value to 4.
        #
        # Output:
        # - POPULATION, a binary matrix with each row encoding an individual.
        # TODO: Perform a crossover between two individuals
        ind1 = pairs[0]
        ind2 = pairs[1]
        num_pairs = ind1.shape[0]

        offspring = np.zeros((self.pop_size, self.num_params * self.nbits))

        for i in range(num_pairs):
            if np.random.rand() < self.crossover_prob:
                if self.crossover_modulo > 0:
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
                    crossover_point = np.random.randint(1, self.num_params * self.nbits)

                offspring[i * 2, :] = np.concatenate(
                    [ind1[i, :crossover_point], ind2[i, crossover_point:]]
                )
                offspring[i * 2 + 1, :] = np.concatenate(
                    [ind2[i, :crossover_point], ind1[i, crossover_point:]]
                )
            else:
                offspring[i * 2, :] = ind1[i, :]
                offspring[i * 2 + 1, :] = ind2[i, :]

        return offspring

    def doMutation(self):
        if np.std(self.fitness) < 1.0:
            current_prob = self.mutation_prob * 3
        else:
            current_prob = self.mutation_prob

        mutation_mask = (
            np.random.rand(self.pop_size, self.num_params * self.nbits) < current_prob
        )
        self.population = np.logical_xor(
            self.population.astype(bool), mutation_mask
        ).astype(int)

    def new_gen(self):
        new_cvalues = np.zeros_like(self.cvalues)

        elite_size = max(5, self.pop_size // 10)
        elite_indices = np.argsort(self.fitness.flatten())[-elite_size:]

        for i in range(elite_size):
            new_cvalues[i] = self.cvalues[elite_indices[i]]

        current_idx = elite_size
        while current_idx < self.pop_size:
            p1_idx = self.tournament_selection()
            p2_idx = self.tournament_selection()

            parent1 = self.cvalues[p1_idx]
            parent2 = self.cvalues[p2_idx]

            if np.random.rand() < self.crossover_prob:
                alpha = np.random.rand(self.num_params)
                offspring1 = alpha * parent1 + (1 - alpha) * parent2
                offspring2 = (1 - alpha) * parent1 + alpha * parent2
            else:
                offspring1 = parent1.copy()
                offspring2 = parent2.copy()

            if current_idx < self.pop_size:
                new_cvalues[current_idx] = offspring1
                current_idx += 1
            if current_idx < self.pop_size:
                new_cvalues[current_idx] = offspring2
                current_idx += 1

        self.cvalues = new_cvalues
        self.encode_individuals()

        elite_mask = np.zeros(self.pop_size, dtype=bool)
        elite_mask[:elite_size] = True

        current_prob = self.mutation_prob
        if np.std(self.fitness) < 5.0:
            current_prob = self.mutation_prob * 2

        mutation_mask = (
            np.random.rand(self.pop_size, self.num_params * self.nbits) < current_prob
        )
        mutation_mask[elite_mask, :] = False

        self.population = np.logical_xor(
            self.population.astype(bool), mutation_mask
        ).astype(int)

        self.current_gen += 1

    def tournament_selection(self):
        tournament_size = max(3, self.pop_size // 20)
        tournament_indices = np.random.choice(
            self.pop_size, tournament_size, replace=False
        )
        tournament_fitness = self.fitness[tournament_indices]
        return tournament_indices[np.argmax(tournament_fitness)]


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
