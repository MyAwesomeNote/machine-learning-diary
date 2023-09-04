class Fruit:

    def __init__(
            self,
            name: str
    ):
        self.name = name

    def __eq__(self, other):
        if isinstance(self, Fruit):
            print("Both classes inherit from the same parent!")
            return True
        return False

    def hello(self):
        print(f"Hello from \"{self.name}\" 🥭")


class Apple(Fruit):

    def __init__(
            self,
            name: str
    ):
        super().__init__(name)

    def hello(self):
        print(f"Hello from {self.name} 🍎")


def main():
    origin_fruit = Fruit("The ancient fruit")
    apple = Apple("Notch's Apple")

    origin_fruit.hello()
    apple.hello()

    print(origin_fruit == apple)


if __name__ == "__main__":
    main()
