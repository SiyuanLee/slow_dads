from enum import Enum

class Color(Enum):
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37

def print_color(text: str):
    print(f'\033[{Color.RED.value}m{text}\033[0m')

# print_color('Hello World')