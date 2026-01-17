from .planification import find_path_to_exit


class AIController:
    def __init__(self, maze_file_path, maze, player):
        self.maze_file = maze_file_path
        self.maze = maze
        self.player = player
        self.path_index = 0

    def calculate_path_to_exit(self):
        self.current_path = find_path_to_exit(self.maze_file)
        self.path_index = 0
        return self.current_path

    def get_next_move_towards_exit(self):
        if not self.current_path or self.path_index >= len(self.current_path):
            self.is_auto_moving = False
            return None

        player_center_x = self.player.x + (self.player.size_x / 2)
        player_center_y = self.player.y + (self.player.size_y / 2)

        target_pos = self.current_path[self.path_index]

        target_center_x = (target_pos[1] + 0.5) * self.maze.tile_size_x
        target_center_y = (target_pos[0] + 0.5) * self.maze.tile_size_y

        dist_x = target_center_x - player_center_x
        dist_y = target_center_y - player_center_y

        # Tolérance : si on est assez proche du centre (5 pixels), considérer qu'on est arrivé
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
