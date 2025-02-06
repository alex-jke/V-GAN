import sys

from text.UI.user_interface import UserInterface



class ConsoleUserInterface(UserInterface):

    _instance = None

    @staticmethod
    def get():
        if ConsoleUserInterface._instance is None:
            ConsoleUserInterface._instance = ConsoleUserInterface()
        return ConsoleUserInterface._instance

    def __init__(self):
        self.data_list = []

    def display(self, data: str):
        self.data_list.append(data)
        self._print()

    def update(self, data: str):
        if len(self.data_list) == 0:
            self.data_list = [data]
        else:
            self.data_list[-1] = data
        self._print()

    def _print(self):
        return
        sys.stdout.write("\r")
        for data in self.data_list:
            sys.stdout.write("\t" + data)
        sys.stdout.flush()

    def done(self):
        self.data_list.pop()


if __name__ == "__main__":
    ui = ConsoleUserInterface()
    for i in range(10):
        ui.update(f"\rUpdating {i}")