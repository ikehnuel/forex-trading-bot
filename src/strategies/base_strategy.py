class BaseStrategy:
    def __init__(self):
        pass

    def execute(self):
        raise NotImplementedError("Execute method must be implemented by subclasses.")

    def evaluate(self):
        raise NotImplementedError("Evaluate method must be implemented by subclasses.")