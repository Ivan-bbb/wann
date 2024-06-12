from numpy import random
from activation_functions import ActivationFunction

class Node:
    def __init__(self, id: int, layer: int, act_func: ActivationFunction):
        self.id = id
        self.layer = layer
        self.act_func = act_func

class Connection:
    def __init__(self, from_node: Node, to_node: Node, enabled: bool, weight: float, innovation_number: int):
        self.from_node = from_node
        self.to_node = to_node
        self.enabled = enabled
        self.weight = weight
        self.innovation_number = innovation_number

    @property
    def connection_id(self):
        return self.from_node.id, self.to_node.id

class InnovationTracker:
    def __init__(self):
        self.innovations = {}
        self.current_innovation_number = 0

    def get_innovation_number(self, from_node: Node, to_node: Node):
        key = (from_node.id, to_node.id)
        if key not in self.innovations:
            self.innovations[key] = self.current_innovation_number
            self.current_innovation_number += 1
        return self.innovations[key]

    def reset(self):
        self.innovations = {}
        self.current_innovation_number = 0

innovation_tracker = InnovationTracker()

class Genome:
    def __init__(self):
        self.connections: dict[tuple[int, int], Connection] = dict()
        self.nodes: list[Node] = list()
        self.fitness: float = 0

    def add_connection(self, from_node: Node, to_node: Node, weight: float) -> Connection:
        connection = Connection(from_node, to_node, True, weight, innovation_number=innovation_tracker.get_innovation_number(from_node, to_node))
        self.connections[connection.connection_id] = connection
        return connection

    def add_node(self, layer: int) -> Node:
        node = Node(len(self.nodes), layer, random.choice(list(ActivationFunction)))
        self.nodes.append(node)
        return node

    