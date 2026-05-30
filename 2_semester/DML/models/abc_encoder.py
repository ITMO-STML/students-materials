from abc import ABC, abstractmethod


class Encoder(ABC):
    @abstractmethod
    def encode() -> None:
        pass
