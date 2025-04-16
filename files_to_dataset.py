import pandas as pd
import networkx as nx
import argparse
import pickle

def parse_input():
    parser = argparse.ArgumentParser(description='Convert csv files to pickled networkx graph')
    parser.add_argument('nodes_path', type=str, help='Path to nodes csv')
    parser.add_argument('edges_path', type=str, help='Path to edges csv')
    parser.add_argument('-o','--output', type=str, help='Path to output pickle file')
    args = parser.parse_args()
    return args.nodes_path, args.edges_path, args.output

if __name__ == '__main__':
    nodes_path, edges_path, output_path = parse_input()
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    
    G = nx.DiGraph()

    for idx, row in nodes.iterrows():
        G.add_node(row['name'])
        G.nodes[row['name']]['label'] = row['type']

    for idx, row in edges.iterrows():
        G.add_edge(row['subject'], row['object'])
        G.edges[row['subject'], row['object']]['feature'] = row['predicate']
        
    pickle.dump(G, open(output_path, 'wb'))
    
    