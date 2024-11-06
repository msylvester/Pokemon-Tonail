class Actions:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    ACTION1 = 4  # Corresponds to 'A' action
    ACTION2 = 5  # Corresponds to 'B' action

    @classmethod
    def list(cls):
        return [cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT, cls.ACTION1, cls.ACTION2]
