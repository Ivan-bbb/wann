from numpy import random
from activation_functions import ActivationFunction
from copy import deepcopy
from config import config

class Node:
    def __init__(self, id: int, layer: int, activation: ActivationFunction):
        self.id = id
        self.layer = layer
        self.activation = activation

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

    def set_weights(self, value: float):
        for conn in self.connections.values():
            conn.weight = value

    def mutate_change_activation(self):
        node = random.choice(self.nodes)
        act = random.choice(list(ActivationFunction))
        node.activation = act

    def mutate_disable_connection(self):
        possible_connections = []

        for conn in self.connections.values():
            if conn.enabled:
                possible_connections.append(conn)

        if not possible_connections:
            return

        conn = random.choice(list(possible_connections))
        conn.enabled = False

    def mutate_enable_connection(self):
        possible_connections = []

        for conn in self.connections.values():
            if not conn.enabled:
                possible_connections.append(conn)

        if not possible_connections:
            return

        conn = random.choice(list(possible_connections))
        conn.enabled = True

    def add_connection(self, from_node: Node, to_node: Node, weight: float) -> Connection:
        connection = Connection(from_node, to_node, True, weight, innovation_number=innovation_tracker.get_innovation_number(from_node, to_node))
        self.connections[connection.connection_id] = connection
        return connection
    
    def mutate_add_connection(self):
        possible_connections = []

        for n1 in self.nodes:
            for n2 in self.nodes:
                if (n1.layer < n2.layer and (n1.id, n2.id) not in self.connections):
                    possible_connections.append((n1, n2))

        if not possible_connections:
            return

        from_node, to_node = random.choice(possible_connections)
        self.add_connection(from_node, to_node, 1.0)

    def add_node(self, layer: int) -> Node:
        node = Node(len(self.nodes), layer, random.choice(list(ActivationFunction)))
        self.nodes.append(node)
        return node

    def mutate_split_connection(self):
        possible_connections = []

        for conn in self.connections.values():
            if (conn.from_node.layer + conn.to_node.layer) // 2 != conn.from_node.layer:
                possible_connections.append(conn)

        if not possible_connections:
            return

        conn = random.choice(possible_connections)
        node = self.add_node(conn.from_node.layer + 1)
        self.add_connection(conn.from_node, node, conn.weight)
        self.add_connection(node, conn.to_node, conn.weight)

    def mutate(self):
        if random.random() < config.prob_change_act:
            self.mutate_change_activation()
        if random.random() < config.prob_disable_conn:
            self.mutate_disable_connection()
        if random.random() < config.prob_enable_conn:
            self.mutate_enable_connection()
        if random.random() < config.prob_add_conn:
            self.mutate_add_connection()
        if random.random() < config.prob_split_conn:
            self.mutate_split_connection()

def connection_crossover(connection_id: tuple[int, int], genome1: Genome, genome2: Genome):
    conn = deepcopy(genome1.connections[connection_id])
    conn.enabled = (genome1.connections[connection_id].enabled and genome2.connections[connection_id].enabled) or random.random() > config.prob_enable_conn
    return conn

def genome_crossover(genome1: Genome, genome2: Genome) -> Genome:
    child = Genome()

    if genome1.fitness > genome2.fitness:
        parent1, parent2 = genome1, genome2
    elif genome1.fitness < genome2.fitness:
        parent1, parent2 = genome2, genome1
    else:
        parent1 = max(genome1, genome2, key=lambda x: len(x.nodes))
        parent2 = min(genome1, genome2, key=lambda x: len(x.nodes))

    for node, _ in zip(parent1.nodes, parent2.nodes):
        child.nodes.append(deepcopy(node))

    if len(parent1.nodes) > len(parent2.nodes):
        for node in parent1.nodes[len(parent2.nodes):]:
            child.nodes.append(deepcopy(node))

    intersection = (parent1.connections.keys() & parent2.connections.keys())

    for conn_id in intersection:
        conn = connection_crossover(conn_id, parent1, parent2)
        child.connections[conn_id] = conn

    if parent1.fitness == parent2.fitness:
        conn_union = parent1.connections | parent2.connections
    else:
        conn_union = parent1.connections

    for conn_id, conn in conn_union.items():
        if conn_id not in intersection:
            child.connections[conn_id] = deepcopy(conn)

    return child

def genome_distance(genome1: Genome, genome2: Genome):
    genome1_innovations = {c.innovation_number: c for c in genome1.connections.values()}
    genome2_innovations = {c.innovation_number: c for c in genome2.connections.values()}

    innovations = genome1_innovations | genome2_innovations

    min_innovation = min(max(genome1_innovations.keys()), max(genome2_innovations.keys()))

    excess = 0
    disjoint = 0

    for innov in innovations.keys():
        if innov not in genome1_innovations or innov not in genome2_innovations:
            if innov <= min_innovation:
                disjoint += 1
            else:
                excess += 1

    return (config.dist_excess * excess + config.dist_disjoint * disjoint)