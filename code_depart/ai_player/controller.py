from .fighting import Fighting
from .planification import find_path_to_exit
from .systeme_expert import SolvingDoors


class AIController:
    def __init__(self, maze_file_path, maze, player):
        self.maze_file = maze_file_path
        self.maze = maze
        self.player = player
        self.path_index = 0
        self.door_solver = SolvingDoors()
        self.current_perception = None
        self.is_optimizing = False  # Variable pour suivre l'état
        self.optimization_complete = False
        self.is_optimized_for_monster = False
        self.last_optimization_position = None

    def calculate_path_to_exit(self):
        self.current_path = find_path_to_exit(self.maze_file)
        self.path_index = 0
        return self.current_path

    def get_next_move_towards_exit(self):
        if not self.current_path or self.path_index >= len(self.current_path):
            self.is_auto_moving = False
            return None

        self.update_perception_and_optimize()

        door_action = self.check_and_solve_doors()
        if door_action:
            return door_action

        player_center_x = self.player.x + (self.player.size_x / 2)
        player_center_y = self.player.y + (self.player.size_y / 2)

        target_pos = self.current_path[self.path_index]

        target_center_x = (target_pos[1] + 0.5) * self.maze.tile_size_x
        target_center_y = (target_pos[0] + 0.5) * self.maze.tile_size_y

        dist_x = target_center_x - player_center_x
        dist_y = target_center_y - player_center_y

        tolerance = 5.0

        if abs(dist_x) <= tolerance and abs(dist_y) <= tolerance:
            self.path_index += 1
            if self.path_index >= len(self.current_path):
                print("Arrivé à destination!")
                self.is_auto_moving = False
                return None
            target_pos = self.current_path[self.path_index]

            target_center_x = (target_pos[1] + 0.5) * self.maze.tile_size_x
            target_center_y = (target_pos[0] + 0.5) * self.maze.tile_size_y
            dist_x = target_center_x - player_center_x
            dist_y = target_center_y - player_center_y

        if abs(dist_x) > abs(dist_y):
            return "RIGHT" if dist_x > 0 else "LEFT"
        else:
            return "DOWN" if dist_y > 0 else "UP"

    def check_and_solve_doors(self):
        door_states = self.maze.look_at_door(self.player, None)

        if door_states:
            door_state = door_states[0]
            print(f"Résolution de: {door_state}")

            cle = self.door_solver.solve_door(door_state)

            if cle:
                print(f"Clé trouvé: {cle}")
                self.maze.unlock_door(cle)

        return None

    def update_perception_and_optimize(self):
        """Met à jour la perception et optimise contre les monstres détectés"""
        self.current_perception = self.maze.make_perception_list(self.player, None)

        if self.current_perception and len(self.current_perception[3]) > 0:
            # Il y a des monstres dans la perception
            monster_rect = self.current_perception[3][0]  # Premier monstre détecté

            # Vérifier si on a déjà optimisé pour ce monstre à cette position
            current_pos = (int(self.player.x), int(self.player.y))

            if (
                not self.is_optimized_for_current_monster
                or self.last_optimization_position != current_pos
            ):
                # Trouver l'instance réelle du monstre
                for monster in self.maze.monsterList:
                    if monster.rect == monster_rect:
                        if not self.is_optimizing and not self.optimization_complete:
                            print(
                                f"🤖 IA détecte monstre à {monster.rect}, lancement optimisation..."
                            )
                            self.is_optimizing = True

                            # Optimisation automatique
                            battle_fighting = Fighting(monster)
                            optimal_attrs = battle_fighting.optimize_player_attributes(
                                self.player
                            )
                            self.player.set_attributes(optimal_attrs)

                            self.is_optimizing = False
                            self.optimization_complete = True
                            self.is_optimized_for_current_monster = True
                            self.last_optimization_position = current_pos
                            print("✅ IA optimisée pour ce monstre !")
                        break
        else:
            # Aucun monstre en vue, réinitialiser
            self.is_optimized_for_current_monster = False
            self.optimization_complete = False
            self.is_optimizing = Falsete = False
            self.is_optimizing = False
