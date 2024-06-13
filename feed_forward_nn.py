from wann import Genome


class NeuralNetwork:
    def __init__(self, genome: Genome):
        self.genome = genome
        self.nodes = {node.id: node for node in genome.nodes}
        self.connections = [conn for conn in genome.connections.values() if conn.enabled]

    def feed(self, inputs: list[float]) -> list[float]:
        node_values = {node.id: 0.0 for node in self.nodes.values()}

        input_nodes = [node for node in self.nodes.values() if node.layer == 0]
        for i, input_value in enumerate(inputs):
            node_values[input_nodes[i].id] = input_value

        # bias
        node_values[input_nodes[-1].id] = 1.0

        layers = sorted(set(node.layer for node in self.nodes.values()))
        input_sum = 0

        for layer in layers[1:]:
            for node in [n for n in self.nodes.values() if n.layer == layer]:
                for conn in self.connections:
                    if conn.to_node.id == node.id:
                        input_sum += node_values[conn.from_node.id] * conn.weight

                node_values[node.id] = node.activation.value(input_sum)

        output_nodes = [node for node in self.nodes.values() if node.layer == max(layers) ]

        return [node_values[node.id] for node in output_nodes]
    
    def set_weights(self, weight: float):
        for conn in self.connections:
            conn.weight = weight