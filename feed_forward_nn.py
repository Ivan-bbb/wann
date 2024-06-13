from wann import Genome
import pickle
import networkx as nx
import matplotlib.pyplot as plt


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

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.genome, f)

    def load(filename: str) -> "NeuralNetwork":
        with open(filename, 'rb') as f:
            genome = pickle.load(f)
        return NeuralNetwork(genome)
    
    def visualize(self, show_image=True, save=True,name =""):
        G = nx.DiGraph()

        pos = {}
        labels = {}

        layers = sorted(set(node.layer for node in self.nodes.values()))
        layer_nodes = {
            layer: [node for node in self.nodes.values() if node.layer == layer]
            for layer in layers
        }

        max_nodes_in_layer = max(len(nodes) for nodes in layer_nodes.values())
        horizontal_spacing = 2
        vertical_spacing = 2

        for layer in layers:
            nodes = layer_nodes[layer]
            num_nodes = len(nodes)
            y_offset = (max_nodes_in_layer - num_nodes) * vertical_spacing / 2
            for i, node in enumerate(nodes):
                pos[node.id] = (
                    layer * horizontal_spacing,
                    i * vertical_spacing + y_offset,
                )
                if layer == 0 and i == len(nodes) - 1:
                    labels[node.id] = "bias"
                else:
                    labels[node.id] = f"{node.id} ({node.layer})\n{node.activation.name}"

        edge_labels = {}
        for conn in self.connections:
            edge_color = "black"
            G.add_edge(conn.from_node.id, conn.to_node.id, color=edge_color)
            edge_labels[(conn.from_node.id, conn.to_node.id)] = (f"{conn.weight:.2f}")

        edges = G.edges()
        colors = [G[u][v]["color"] for u, v in edges]

        plt.figure(figsize=(15, 10))
        nx.draw(
            G,
            pos,
            with_labels=True,
            labels=labels,
            node_size=2500,
            node_color="orange",
            font_size=8,
            font_weight="bold",
            arrowsize=15,
            edge_color=colors,
        )
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="blue")

        plt.title("Neural Network")
        if(save):
            plt.savefig(f'./topologies/{name}.png')
        if (show_image):
            plt.show()