"""Helper functions file.

This module contains helper functions for quantum information tasks.
"""
from typing import Any
import numpy as np
import networkx as nx
import pylab as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions import Initialize
from qiskit.visualization import plot_histogram
from qiskit import execute


def random_state(nqubits: int) -> np.ndarray:
    """Generates a random nqubit state vector.
    
    Args:
        nqubits (int): number of qubits of the state to generate.
    
    Returns:
        ndarray: Randomly generated nqubit states.
    """
    # Generate real and imaginary parts of the nqubit states.
    real_parts, im_parts = np.array([]), np.array([])
    for _ in range(2**nqubits):
        real_parts = np.append(real_parts, (np.random.random()*2)-1)
        im_parts = np.append(im_parts, (np.random.random()*2)-1)
    
    # Combine into list of complex numbers:
    amps = real_parts + 1j*im_parts
   
    # Normalise
    magnitude_squared = sum(abs(a)**2 for a in amps)
    return amps/np.sqrt(magnitude_squared)


def counts_of_one_register(counts: dict, register: int) -> dict:
    """Returns the counts of only one register, given the counts of various registers.
    
    Args:
        counts (dict): The counts of a results.
        register (int): The register from the passed one, that you want the counts from. The order is the one from the original counts.
        
    Returns:
        dict: The counts of the given register.
    """
    zeros, ones = 0, 0
    for state, count in counts.items():
        if state[2*register] == "0":
            zeros += count
        elif state[2*register] == "1":
            ones += count

    return {
        0: zeros,
        1: ones,
    }
    
    
def get_probabilities(counts: dict) -> dict:
    """Returns the counts as probabilities.
    
    Args:
        counts (dict): The counts of a results.
    
    Returns:
        dict: The probabilities associated to the given counts.
    """
    norm = sum(counts.values())
    return {i: count/norm for i, count in counts.items()}


def execute_get_probabilities_and_plot(circuit: QuantumCircuit, backend: Any, shots: int, register: int):
    """Executes circuit, gets probabilities and plots.

    Args:
       circuit (QuantumCircuit): Circuit to execute.
       backend (Any): Backend to run on. 
       shots (int): Number of shots.
       register (int): Register to get the probabilities from and plot them.

    Returns:
       Plot figure.
    """
    counts = execute(circuit, backend, shots=shots).result().get_counts()
    bob_counts = counts_of_one_register(counts, register)
    probabilities = get_probabilities(bob_counts)
    return plot_histogram(probabilities)
            

def create_networkx_graph(edges: dict) -> nx.Graph:
    """Create a networkx graph from a dicionary of edges.
    
    Args:
        edges (dict): The dicionary key is a tuple of the two nodes, and the value is the edge label.
    
    Returns:
        nx.Graph: Returns the constructed graph.
    """
    G = nx.Graph()
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    
    return G
    
    
def print_networkx_graph(G: nx.Graph, labels: dict):
    """Prints a given graph, given the passed labels.
    
    Args:
        G (nx.Graph): graph to plot.
        labels (dict): labels to in the graph edges. The key is a tuple of the two nodes, and the value is the edge label.

    Returns:
        Displays the passed graph.
    """
    options = {"node_size": 1000, "node_color": "blue", "with_labels": True, "font_weight":'bold'}
    pos=nx.spring_layout(G)

    nx.draw(G, pos, **options)
    nx.draw_networkx_edge_labels(G, pos,edge_labels=labels,font_size=10)

    plt.show()
    

def compute_network_path(graph:nx.Graph, sender:str, receiver:str) -> list[tuple]:
    """Computes the shortest path in the network and passes it to an edge list.
    
    Args:
        G (nx.Graph): graph of the network.
        sender (str): node of the graph to start the path from.
        receiver (str): node of the graph to end the path.
        
    Returns:
        list[tuple]: edge list, of the shortest path.
    """
    paths = list(nx.shortest_simple_paths(graph, sender, receiver))
    shortest_path = paths[0]

    return [
        (node, shortest_path[i + 1])
        for i, node in enumerate(shortest_path)
        if i != len(shortest_path) - 1
    ]
        
    
def create_secure_quantum_teleportation_path_circuit(init_gate: Initialize, edges: list[tuple]):
    """Generates a qiskit circuit for the secure quantum teleportation network for a concrete path.

    Args:
        edges (list[tuple]): List of tuples containing the edges of the graph, starting by the emmiter, and ending in the receiver.
    
    Returns:
        QuantumCircuit: Returns the builded circuit.
    """
    # Define the circuit quantum channels:
    message_qr = QuantumRegister(1, f"{edges[0][0]} initial Ψ")  
    
    q_registers1, q_registers2 = [], []
    for edge in edges:
        q_registers1.append(QuantumRegister(1, f"{edge[0]} q entg w {edge[1]}"))
        q_registers2.append(QuantumRegister(1, f"{edge[1]} q entg w {edge[0]}"))
    
    c_registers1, c_registers2 = [], []
    for edge in edges:
        c_registers1.append(ClassicalRegister(1, f"{edge[0]}_c_bit1"))
        c_registers2.append(ClassicalRegister(1, f"{edge[0]}_c_bit2"))

    teleport_network_circuit = QuantumCircuit(message_qr)
    
    for i in range(len(q_registers1)):
        teleport_network_circuit.add_register(q_registers1[i])
        teleport_network_circuit.add_register(q_registers2[i])
        
    for i in range(len(c_registers1)):
        teleport_network_circuit.add_register(c_registers1[i]) 
        teleport_network_circuit.add_register(c_registers2[i])  
        
    for i in range(len(edges)):
        teleport_network_circuit.h(2*i+1) 
        teleport_network_circuit.cx(2*i+1,2*i+2)
        
    teleport_network_circuit.barrier()
    
    # After some time, now Alice wants to teleport a state to Bob, given by:
    teleport_network_circuit.append(init_gate, [0])
    teleport_network_circuit.barrier()

    for i in range(len(edges)):
        teleport_network_circuit.cx(2*i, 2*i+1) 
        teleport_network_circuit.h(2*i)
        teleport_network_circuit.measure(2*i, 2*i)
        teleport_network_circuit.measure(2*i+1, 2*i+1)
    teleport_network_circuit.barrier()
    
    for i in range(len(edges)):
        teleport_network_circuit.z(2*len(edges)).c_if(c_registers1[i], 1)  #if cr1 is 1 apply Z gate
        teleport_network_circuit.x(2*len(edges)).c_if(c_registers2[i], 1)  #if cr2 is 1 apply X gate


    cr_result = ClassicalRegister(1, "meas")
    teleport_network_circuit.add_register(cr_result)
    teleport_network_circuit.measure(len(edges)*2,len(edges)*2)
    
    return teleport_network_circuit
