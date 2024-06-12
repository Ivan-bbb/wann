class Node:
    def __init__(self, id: int, layer: int, act_func):
        self.id = id
        self.layer = layer
        self.act_func = act_func

class Connection:
    def __init__(self, from_node: Node, to_node: Node, weight: float, enabled: bool):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.enabled = enabled

    @property
    def connection_id(self):
        return self.from_node.id, self.to_node.id

class Genome:
    def __init__(self):
        self.connections: dict[tuple[int, int], Connection] = dict()
        self.nodes: list[Node] = list()
        self.fitness: float = 0