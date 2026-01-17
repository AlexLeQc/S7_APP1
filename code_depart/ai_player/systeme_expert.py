from swiplserver import PrologMQI


class SolvingDoors:
    def solve_door(self, door_state):
        if not door_state:
            return None

        with PrologMQI() as mqi:
            with mqi.create_thread() as prolog_thread:
                prolog_thread.query("['ai_player/prolog/porte.pl']")

                query = f"resoudre_porte({door_state}, Cle)."

                print(f"Query Prolog: {query}")
                result = prolog_thread.query(query)

                if result:
                    return result[0]["Cle"]
                return None
