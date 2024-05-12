import networkx as nx
import matplotlib.pyplot as plt
import random
from numpy.random import Generator, SeedSequence, MT19937
import numpy as np
from copy import deepcopy
from math import sqrt
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
import seaborn as sns
import csv


def random_attack(graph: nx.Graph) -> list:
    nodes_to_remove = list(graph.nodes())
    random.shuffle(nodes_to_remove)
    return nodes_to_remove


def degree_attack(graph: nx.Graph) -> list:
    nodes_to_remove = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)
    return nodes_to_remove


def pagerank_attack(graph: nx.Graph) -> list:
    pr = nx.pagerank(graph)
    nodes_to_remove = sorted(graph.nodes(), key=lambda x: pr[x], reverse=True)
    return nodes_to_remove


def betweenness_attack(graph: nx.Graph) -> list:
    bc = nx.betweenness_centrality(graph)
    nodes_to_remove = sorted(graph.nodes(), key=lambda x: bc[x], reverse=True)
    return nodes_to_remove


def closeness_attack(graph: nx.Graph) -> list:
    cc = nx.closeness_centrality(graph)
    nodes_to_remove = sorted(graph.nodes(), key=lambda x: cc[x], reverse=True)
    return nodes_to_remove


def simulate_attacks(graph: nx.Graph, attack_mode="random") -> tuple:
    original_size = graph.number_of_nodes()
    giant_component_sizes = []
    diameters = []
    removed_fraction = []
    function_map = {
        "random": random_attack,
        "degree": degree_attack,
        "pagerank": pagerank_attack,
        "betweenness": betweenness_attack,
        "closeness": closeness_attack,
    }
    nodes_to_remove = function_map[attack_mode](graph)
    for i, node in enumerate(nodes_to_remove):
        print(
            f"Removing node {node} ({i + 1}/{original_size})" if i % 100 == 0 else "",
            end="\r",
        )
        graph.remove_node(node)
        number_of_nodes = graph.number_of_nodes()
        if number_of_nodes > 0:
            largest_cc = max(nx.connected_components(graph), key=len)
            largest_component = graph.subgraph(largest_cc)
            largest_component_size = largest_component.number_of_nodes()
            giant_component_sizes.append(largest_component_size / original_size)
            if nx.is_connected(largest_component):
                diameters.append(nx.diameter(largest_component))
            else:
                diameters.append(float("inf"))
        else:
            giant_component_sizes.append(0)
            diameters.append(float("inf"))
        removed_fraction.append((i + 1) / original_size)
    return attack_mode, removed_fraction, giant_component_sizes, diameters


def setup_graph():
    DATA_PATH: Path = Path("data/bn-mouse-kasthuri_graph_v4.edges")
    print(f"Using dataset from {DATA_PATH.absolute()}")
    G = nx.read_edgelist(path=DATA_PATH, create_using=nx.Graph(), nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))
    print("nodes:", len(G.nodes()), "and edges:", len(G.edges()))
    G = nx.k_core(G, k=2)
    pos = nx.spring_layout(G, seed=SEED, k=15 / sqrt(len(G.nodes())))
    infected_time_init: int = 0
    nx.set_node_attributes(G, "S", "state")
    nx.set_node_attributes(G, infected_time_init, "infection_cooldown")
    nx.set_node_attributes(G, 0, "recovery_step")
    nx.set_node_attributes(G, 0, "infection_step")
    number_of_nodes = G.number_of_nodes()
    average_number_of_edges = len(G.edges()) / number_of_nodes

    return G, pos, number_of_nodes, average_number_of_edges


def worker(args):
    graph, attack_mode = args  # Unpack the tuple into graph and attack_mode
    print(
        f"Simulating {attack_mode} attack on graph with {graph.number_of_nodes()} nodes..."
    )
    graph_copy = deepcopy(graph)  # Make a deep copy of the graph
    return simulate_attacks(graph_copy, attack_mode)


def plot_and_save(graph_name: str, result_dict: dict, attack_modes: list[str]):
    file_name = "-".join(graph_name.lower().split())
    plot_folder_out = Path("plots")
    plot_folder_out.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    # Preparing data for plotting
    data1 = []
    data2 = []
    for mode in attack_modes:
        x, y, z = result_dict[mode]
        data1.extend([(xx, yy, mode) for xx, yy in zip(x, y)])
        data2.extend([(xx, zz, mode) for xx, zz in zip(x, z)])
    STR_CONST_1: str = "Fraction of Nodes Removed"
    df1 = pd.DataFrame(
        data1, columns=[STR_CONST_1, "Relative Size of Largest Component", "Mode"]
    )
    df2 = pd.DataFrame(
        data2, columns=[STR_CONST_1, "Diameter of Largest Component", "Mode"]
    )

    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plotting using Seaborn on the first subplot
    sns.lineplot(
        x=STR_CONST_1,
        y="Relative Size of Largest Component",
        hue="Mode",
        data=df1,
        ax=ax1,
    )
    ax1.set_title(f"Robustness Analysis of {graph_name}")
    ax1.grid(True)
    # Plotting using Seaborn on the second subplot
    sns.lineplot(
        x=STR_CONST_1, y="Diameter of Largest Component", hue="Mode", data=df2, ax=ax2
    )
    ax2.set_title(f"Diameter Changes of {graph_name}")
    ax2.grid(True)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(plot_folder_out / f"robustness-diameter-analysis-{file_name}.png")
    plt.close(fig)  # Close the figure to free up memory


def calculate_critical_threshold(graph):
    degrees = [d for n, d in graph.degree()]
    k = np.mean(degrees)
    k2 = np.mean([deg**2 for deg in degrees])
    if k == 0:
        return 0
    return 1 - (k / (k2 - k))


def output_results_to_csv(graph_data, filename):
    df = pd.DataFrame(graph_data)
    df.to_csv(filename, index=False)
    print(f"Output saved to {filename}")


def calculate_and_save_critical_thresholds(graphs, output_dir):
    data = [
        {"Graph": name, "Critical Threshold": calculate_critical_threshold(graph)}
        for name, graph in graphs.items()
    ]
    output_path = Path(__file__).parent.joinpath(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(
        output_path.joinpath("critical_thresholds.csv"), mode="w", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def process_graphs(graphs, attack_modes):
    result_dict = {}
    with Pool(processes=len(attack_modes)) as pool:
        graph_and_modes = [(graph, mode) for graph, mode in graphs]
        results = pool.map(worker, graph_and_modes)
        result_dict = {result[0]: result[1:] for result in results}
    return result_dict


if __name__ == "__main__":
    # Setup
    SEED = 2137
    random.seed(SEED)
    attack_modes = ["random", "degree", "pagerank", "betweenness", "closeness"]
    mouse_graph, pos, number_of_nodes, average_number_of_edges = setup_graph()

    random_graph = nx.barabasi_albert_graph(
        n=number_of_nodes, m=int(average_number_of_edges), seed=SEED
    )
    # Process critical thresholds and save results
    graphs = {"mouse-kasthuri-graph-v4": mouse_graph, "random": random_graph}
    calculate_and_save_critical_thresholds(graphs, "../out")

    # Process and plot results for random and real graphs
    for graph_name, graph in graphs.items():
        print(f"Processing {graph_name} simulations...")
        graph_and_modes = [(graph, mode) for mode in attack_modes]
        result_dict = process_graphs(graph_and_modes, attack_modes)
        plot_and_save(
            graph_name=graph_name, result_dict=result_dict, attack_modes=attack_modes
        )
        print(f"All {graph_name.lower()} simulations completed.")
