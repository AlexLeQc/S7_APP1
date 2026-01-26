import random


class Door:
    STATES = [
        ["gold", "yellow", "blue", "white", "white", "", ""],
        ["gold", "blue", "blue", "blue", "", "", ""],
        ["gold", "blue", "white", "red", "red", "white", "black"],
        ["silver", "black", "red", "red", "red", "", ""],
        ["gold", "red", "blue", "red", "blue", "black", ""],
        ["silver", "blue", "red", "yellow", "", "", ""],
        ["bronze", "blue", "blue", "blue", "blue", "", ""],
        ["bronze", "white", "black", "black", "blue", "yellow", ""],
        ["bronze", "blue", "blue", "red", "yellow", "black", "white"],
        ["bronze", "black", "white", "red", "red", "black", ""],
    ]

    KEYS = [
        "first",
        "second",
        "fourth",
        "fourth",
        "fourth",
        "first",
        "second",
        "first",
        "fourth",
        "first",
    ]

    def __init__(self, rect):
        self.rect = rect
        i = random.randrange(0, 10)
        self.state = self.STATES[i]
        self.key = self.KEYS[i]

    def check_door(self):
        return self.state

    def unlock_door(self, key):
        if key == self.key:
            return True
        else:
            i = random.randrange(0, 10)
            self.state = self.STATES[i]
            self.key = self.KEYS[i]
            return False
