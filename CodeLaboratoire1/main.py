# Université de Sherbrooke
# Code préparé par Audrey Corbeil Therrien
# Laboratoire 1 - Interaction avec prolog

import heapq

from swiplserver import PrologMQI


def get_successors(prolog_thread, city):
    query = f"s({city}, Successors)."
    result = prolog_thread.query(query)
    return result[0]["Successors"] if result and result[0] else []


def get_distance(prolog_thread, city1, city2):
    query = f"d({city1}, {city2}, Distance)."
    result = prolog_thread.query(query)
    return result[0]["Distance"] if result else None


def get_heuristic(prolog_thread, city):
    query = f"h({city}, Distance)."
    result = prolog_thread.query(query)
    return result[0]["Distance"] if result else None


def profondeur(prolog_thread, start, goal):
    stack = [(start, [start], 0)]
    visited = set()

    while stack:
        current, path, cost = stack.pop()

        if current == goal:
            return path, cost

        if current not in visited:
            visited.add(current)

            successors = get_successors(prolog_thread, current)
            for successor in successors:
                if successor not in visited:
                    new_cost = cost + get_distance(prolog_thread, current, successor)
                    new_path = path + [successor]
                    stack.append((successor, new_path, new_cost))
    return None, float("inf")


def a_star(prolog_thread, start, goal):
    priority_queue = [(get_heuristic(prolog_thread, start), 0, start, [start], 0)]
    g_score = {start: 0}

    while priority_queue:
        _, _, current, path, cost = heapq.heappop(priority_queue)

        if current == goal:
            return path, cost

        successors = get_successors(prolog_thread, current)
        for successor in successors:
            tentative_g_score = g_score[current] + get_distance(
                prolog_thread, current, successor
            )

            if successor not in g_score or tentative_g_score < g_score[successor]:
                g_score[successor] = tentative_g_score
                f_score = tentative_g_score + get_heuristic(prolog_thread, successor)
                new_path = path + [successor]
                heapq.heappush(
                    priority_queue,
                    (
                        f_score,
                        tentative_g_score,
                        successor,
                        new_path,
                        tentative_g_score,
                    ),
                )

    return None, float("inf")


def main():
    print("Recherche de chemin d'Arad à Bucharest")
    print("=" * 40)

    with PrologMQI() as mqi:
        with mqi.create_thread() as prolog_thread:
            # Charger la base de connaissances Roumanie
            prolog_thread.query("[prolog/roumanie].")

            start = "arad"
            goal = "bucharest"

            # Test des algorithmes
            algorithms = [("A*", a_star)]

            for name, algorithm in algorithms:
                print(f"\n{name}:")
                path, cost = algorithm(prolog_thread, start, goal)

                if path:
                    print(f"  Chemin: {' -> '.join(path)}")
                    print(f"  Coût: {cost}")
                else:
                    print("  Aucun chemin trouvé")


if __name__ == "__main__":
    main()
