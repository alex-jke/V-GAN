from abc import abstractmethod


class UserInterface():

    @abstractmethod
    def display(self, data: str):
        pass

    @abstractmethod
    def update(self, data: str):
        pass

    @abstractmethod
    def done(self):
        pass