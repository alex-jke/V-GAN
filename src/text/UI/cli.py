import sys

from text.UI.user_interface import UserInterface


class ConsoleUserInterface(UserInterface):
    def display(self, data: str):
        print(data)

    def update(self, data: str):
        sys.stdout.write("\r"+ data)
        sys.stdout.flush()


if __name__ == "__main__":
    ui = ConsoleUserInterface()
    for i in range(10):
        ui.update(f"\rUpdating {i}")