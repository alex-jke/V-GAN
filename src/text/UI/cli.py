import sys
import time
from typing import Optional

from text.UI.user_interface import UserInterface


class ConsoleUserInterface(UserInterface):
    """
    Manages a console-based user interface to display, update, and manage data entries in a list.

    This class is designed to handle a text-based interface where data can be added to a list and displayed in the console.
    It supports automatic cleanup using a context manager, allowing the last displayed data to be removed when the context
    is exited.
    """
    _instance: Optional['ConsoleUserInterface'] = None

    @staticmethod
    def get() -> 'ConsoleUserInterface':
        if ConsoleUserInterface._instance is None:
            ConsoleUserInterface._instance = ConsoleUserInterface()
        return ConsoleUserInterface._instance

    class DisplayContextManager:
        def __init__(self, ui: 'ConsoleUserInterface'):
            self.ui = ui

        def __enter__(self):
            return self  # Not used, but can return self or None

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.ui.done()

    def __init__(self):
        self.data_list: list[str] = []

    def display(self, data: str = None) -> DisplayContextManager:
        """
        Add a new data to the list and return a context manager that
        automatically calls done() when exited.
        """
        if data is None:
            data = ""
        self.data_list.append(data)
        self._print()
        return self.DisplayContextManager(self)

    def update(self, data: str):
        """
        Update the last data in the list. Replaces the last entry.
        """
        if not self.data_list:
            self.data_list = [data]
        else:
            self.data_list[-1] = data
        self._print()

    def _print(self):
        #print(" | ".join(self.data_list))
        sys.stdout.write("\r")
        sys.stdout.write(" | ".join(self.data_list))
        sys.stdout.flush()

    def done(self):
        """
        Remove the last data from the list.
        """
        if self.data_list:
            self.data_list.pop()
            self._print()

# Example usage:
if __name__ == "__main__":
    ui = ConsoleUserInterface.get()
    with ui.display():
        for progress in range(10):
            with ui.display():
                for sub_progress in range(100):
                    ui.update(f"Subtask progress: {sub_progress}%")
                    time.sleep(0.01)
            ui.update(f"Task progress: {progress}%")
            time.sleep(0.1)
        # After exiting contexts, all tasks are cleaned up