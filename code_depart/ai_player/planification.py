import csv
import heapq


class Node:
    def __init__(self, pos, parent=None):
        self.pos = pos
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f


def load_maze_from_csv(csv_file):
    maze = []
    with open(csv_file) as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            maze.append(row)
    return maze


def find_positions_in_maze(maze, start_char="S", end_char="E"):
    start_pos = None
    end_pos = None

    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            if cell == start_char:
                start_pos = (i, j)
            elif cell == end_char:
                end_pos = (i, j)

    return start_pos, end_pos


def find_path_to_exit(csv_file_path):
    maze = load_maze_from_csv(csv_file_path)

    start_pos, end_pos = find_positions_in_maze(maze)

    if not start_pos or not end_pos:
        print(f"Erreur: Départ ({start_pos}) ou arrivée ({end_pos}) non trouvés")
        return None

    print(
        f"Labyrinthe {len(maze)}x{len(maze[0])} - Départ: {start_pos}, Arrivée: {end_pos}"
    )

    path = a_star_search(maze, start_pos, end_pos)

    if path:
        print(f"Chemin: {path}")

    return path


def a_star_search(maze_grid, start, end):
    open_list = []
    visited_g_scores = {}

    start_node = Node(start)
    heapq.heappush(open_list, start_node)
    visited_g_scores[start] = 0

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node.pos == end:
            return reconstruct_path(current_node)

        (r, c) = current_node.pos
        for move in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor_pos = (r + move[0], c + move[1])

            if not (
                0 <= neighbor_pos[0] < len(maze_grid)
                and 0 <= neighbor_pos[1] < len(maze_grid[0])
            ):
                continue
            if maze_grid[neighbor_pos[0]][neighbor_pos[1]] == "1":
                continue

            new_g = current_node.g + 1

            if (
                neighbor_pos not in visited_g_scores
                or new_g < visited_g_scores[neighbor_pos]
            ):
                visited_g_scores[neighbor_pos] = new_g

                neighbor_node = Node(neighbor_pos, current_node)
                neighbor_node.g = new_g
                neighbor_node.h = abs(neighbor_pos[0] - end[0]) + abs(
                    neighbor_pos[1] - end[1]
                )
                neighbor_node.f = neighbor_node.g + neighbor_node.h

                heapq.heappush(open_list, neighbor_node)

    return None


def reconstruct_path(node):
    path = []
    while node:
        path.append(node.pos)
        node = node.parent
    return path[::-1]
