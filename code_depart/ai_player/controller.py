import math
from enum import Enum

import numpy as np
from Constants import *

from .fighting import Fighting
from .fuzzyObstacleAvoidance import FuzzyObstacleAvoidance
from .planification import find_path_to_exit
from .systeme_expert import SolvingDoors


class AgentState(Enum):
    START = 1
    NAVIGATING = 2
    DOOR_INTERACTION = 3
    COMBAT = 4


class AiController:
    def __init__(self, maze_file, maze, player):
        self.maze_file = maze_file
        self.maze = maze
        self.player = player

        self.fuzzy_navigation = FuzzyObstacleAvoidance()
        self.state = AgentState.START
        self.path = []
        self.path_step = 0

        self.door_solver = SolvingDoors()
        self.combat_optimized = False
        self.current_enemy = None
        self.movement_vector = (0, 0)

    def initialize_path(self):
        self.path = find_path_to_exit(self.maze_file)
        self.path_step = 0
        self.state = AgentState.NAVIGATING
        print("Chemin initialisé")

    def update(self):
        perception = self.maze.make_perception_list(self.player, None)

        if self.state == AgentState.START:
            self.initialize_path()

        elif self.state == AgentState.NAVIGATING:
            self._compute_movement(perception)
            if perception[4]:
                self.state = AgentState.DOOR_INTERACTION
            elif perception[3]:
                self.state = AgentState.COMBAT
                self.combat_optimized = False
                self.current_enemy = perception[3][0]

        elif self.state == AgentState.DOOR_INTERACTION:
            self._handle_door()
            self.state = AgentState.NAVIGATING

        elif self.state == AgentState.COMBAT:
            self._engage_combat(perception)

    def _compute_movement(self, perception):
        if not self.path or self.path_step >= len(self.path):
            self.movement_vector = (0, 0)
            return

        target_cell = self.path[self.path_step]
        target_pos = grid_to_world(target_cell, self.maze)
        player_pos = self.player.get_position()

        if reached_target(target_pos, player_pos, tolerance=20):
            self.path_step += 1
            self.movement_vector = (0, 0)
            return

        direction = (target_pos[0] - player_pos[0], target_pos[1] - player_pos[1])
        norm = np.linalg.norm(direction)
        unit_dir = (direction[0] / norm, direction[1] / norm) if norm != 0 else (0, 0)

        obstacles = perception[1] + perception[0]
        dist, angle = nearest_obstacle_info(
            self.player.get_rect().center, obstacles, unit_dir, 20
        )
        steering_adjust = self.fuzzy_navigation.compute_steering(dist, angle)

        self.movement_vector = rotate_vector_2d(unit_dir, steering_adjust)

    def _handle_door(self):
        doors = self.maze.look_at_door(self.player, None)
        if not doors:
            return

        door = doors[0]
        print(f"Interaction porte: {door}")

        key = self.door_solver.solve_door(door)
        if key:
            print(f"Clé utilisée: {key}")
            self.maze.unlock_door(key)

    def _engage_combat(self, perception):
        enemies = perception[3]
        if not enemies:
            print("Combat terminé")
            self.state = AgentState.NAVIGATING
            self.current_enemy = None
            self.movement_vector = (0, 0)
            return

        if not self.combat_optimized:
            enemy = enemies[0]
            print(f"Combat contre {enemy}")
            fight = Fighting(enemy)
            best_attributes = fight.optimize_player_attributes(self.player)
            self.player.set_attributes(best_attributes)
            self.combat_optimized = True
            print("Joueur optimisé pour le combat")


def grid_to_world(cell, maze):
    row, col = cell
    x = col * maze.tile_size_x + maze.tile_size_x / 2
    y = row * maze.tile_size_y + maze.tile_size_y / 2
    return (x, y)


def reached_target(target, position, tolerance=2):
    return math.hypot(target[0] - position[0], target[1] - position[1]) < tolerance


def rotate_vector_2d(vector, angle_deg):
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array(
        [vector[0] * cos_t - vector[1] * sin_t, vector[0] * sin_t + vector[1] * cos_t]
    )


def nearest_obstacle_info(player_pos, obstacles, forward_dir, player_radius):
    min_dist = float("inf")
    closest = None

    for obs in obstacles:
        dx = obs.centerx - player_pos[0]
        dy = obs.centery - player_pos[1]
        edge_dist = max(0, math.hypot(dx, dy) - (player_radius + obs.width / 2))
        if edge_dist < min_dist:
            min_dist = edge_dist
            closest = (dx, dy)

    if closest is None:
        return float("inf"), 0.0

    angle_forward = math.atan2(forward_dir[1], forward_dir[0])
    angle_to_obs = math.atan2(closest[1], closest[0])
    relative_angle = math.degrees(angle_to_obs - angle_forward)
    relative_angle = (relative_angle + 180) % 360 - 180

    return min_dist, relative_angle
