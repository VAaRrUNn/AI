import random
from .autodiff import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

    def update_params(self, lr=1e-2):
        for p in self.parameters():
            p.data -= lr * p.grad


class Neuron(Module):
    """
    A simple neuron
    Single computaion
    """

    def __init__(self, input_units, activation="tanh"):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(input_units)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation
        self.inputs = input_units

    def __call__(self, x):
        """
        returns the computed value by the neuron
        """

        out = sum((w * x for w, x in zip(self.w, x)), self.b)
        if self.activation == "tanh":
            return out.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"Neuron ({self.inputs})"


class Layer(Module):
    def __init__(self, input_units, output_units, activation="tanh"):
        super().__init__()
        self.input_units = input_units
        self.output_units = output_units
        self.layer = [
            Neuron(input_units=input_units, activation=activation)
            for _ in range(output_units)
        ]

    def __call__(self, x):
        out = [n(x) for n in self.layer]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.layer for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer ({self.input_units}: {self.output_units})"


class MLP(Module):
    def __init__(self, layers, activation="tanh"):
        """
        Args:
            layers: List of units in a layers

        Eg: [2, 2, 1]. It will create a NN, which has 2 input units,
        2 hidden units and 1 output unit.
        """
        super().__init__()
        self.layers = layers
        self.mlp = [
            Layer(input_units=i, output_units=j, activation=activation)
            for i, j in zip(layers[:-1], layers[1:])
        ]

    def __call__(self, x):
        out = x
        for layer in self.mlp:
            out = layer(out)
        return out

    def parameters(self):
        return [p for layer in self.mlp for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP: ({self.layers})"
