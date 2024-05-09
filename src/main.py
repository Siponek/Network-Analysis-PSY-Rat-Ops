import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from copy import deepcopy
from math import sqrt
from pathlib import Path
from multiprocessing import Pool


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
    G = nx.k_core(G, k=2)
    print("nodes:", len(G.nodes()), "and edges:", len(G.edges()))
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

    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plotting on the first subplot
    ax1.grid(True)
    for mode in attack_modes:
        x, y, _ = result_dict[mode]
        ax1.plot(x, y, label=f"{mode}")
    ax1.set_xlabel("Fraction of Nodes Removed")
    ax1.set_ylabel("Relative Size of Largest Component")
    ax1.set_title(f"Robustness Analysis of {graph_name}")
    ax1.legend()

    # Plotting on the second subplot
    ax2.grid(True)
    for mode in attack_modes:
        x, _, z = result_dict[mode]
        ax2.plot(x, z, label=f"{mode}")
    ax2.set_xlabel("Fraction of Nodes Removed")
    ax2.set_ylabel("Diameter of Largest Component")
    ax2.set_title(f"Diameter Changes of {graph_name}")
    ax2.legend()

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(plot_folder_out / f"robustness-diameter-analysis-{file_name}.png")
    plt.close(fig)  # Close the figure to free up memory


if __name__ == "__main__":
    # Setup
    SEED = 2137
    np.random.seed(SEED)
    random.seed(SEED)
    attack_modes = ["random", "degree", "pagerank", "betweenness", "closeness"]
    mouse_graph, pos, number_of_nodes, average_number_of_edges = setup_graph()

    random_graph = nx.barabasi_albert_graph(n=500, m=int(5), seed=SEED)
    random_graph = nx.k_core(random_graph, k=2)
    result_dict = {}
    results: list = []
    # temp_list_result = []
    # for mode in attack_modes:
    #     print(f"Simulating {mode} attack on random graph...")
    #     temp_list_result.append(simulate_attacks(deepcopy(random_graph), mode))
    #     print(f"Simulating {mode} finished...")
    # result_dict[mode] = simulate_attacks(mouse_graph, mode)
    # Multiprocessing

    with Pool(processes=len(attack_modes)) as pool:
        graph_and_modes = [(random_graph, mode) for mode in attack_modes]
        results = pool.map(worker, graph_and_modes)
    # Organize results in a dictionary
    result_dict = {result[0]: result[1:] for result in results}
    # result_dict = {result[0]: result[1:] for result in temp_list_result}
    print("All simulations completed.")
    # close the pool
    # No more tasks can be submitted, tell the pool it can begin shutting down.
    pool.close()
    # Wait for all worker processes to finish.
    pool.join()

    # # Real graph
    # plot_and_save(
    #     graph_name="Random Graph", result_dict=result_dict, attack_modes=attack_modes
    # )

    # with Pool(processes=len(attack_modes)) as pool:
    #     graph_and_modes = [(mouse_graph, mode) for mode in attack_modes]
    #     results = pool.map(worker, graph_and_modes)
    # result_dict = {result[0]: result[1:] for result in results}
    # print("All simulations completed.")
    # pool.close()
    # pool.join()
    # plot_and_save(
    #     graph_name="Mouse Brain Graph",
    #     result_dict=result_dict,
    #     attack_modes=attack_modes,
    # )
