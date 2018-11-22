from networkStructure import NetworkStructure
from networkConnection import NetworkConnection

instance = [[0,0,0],
            [0,1,1],
            [1,0,1],
            [1,1,0]]

num_of_features = len(instance[0]) - 1

hidden_nodes = [3, 3]

nodes = NetworkStructure.create_nodes(num_of_features, hidden_nodes)

weights = NetworkConnection.create_weights(nodes, num_of_features, hidden_nodes)