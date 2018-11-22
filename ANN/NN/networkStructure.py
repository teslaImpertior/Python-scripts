from node import Node

class NetworkStructure:
    def create_nodes(num_of_features, hidden_nodes):
        nodes = []
        nodeIndex = 0
        
        #--------------------------------------------
        # input layer
        print("input layer: \t", end='')
        
        # bias unit
        node = Node()
        node.set_level(0)
        node.set_label("+1")
        node.set_index(nodeIndex)
        node.set_is_bias_unit(True)
        nodes.append(node)  # append this bias unit to nodes array
        nodeIndex = nodeIndex + 1 # increment index
        
        print(node.get_label(), "\t", end='')
        
        #--------------------------------------------
        
        for i in range(num_of_features):
            node = Node()
            node.set_level(0)
            node.set_label("X"+str(i+1))
            node.set_index(nodeIndex)
            node.set_is_bias_unit(False)
            nodes.append(node)  # append this bias unit to nodes array
            nodeIndex = nodeIndex + 1 # increment index
            
            print(node.get_label(), "\t", end='')
        
        print("")
        
        #--------------------------------------------  
        # hidden layers        
        
        for i in range(len(hidden_nodes)):
            
            print("hidden layer: ", end='')
            
            # bias unit
            node = Node()
            node.set_level(i+1)
            node.set_label("+1")
            node.set_index(nodeIndex)
            node.set_is_bias_unit(True)
            nodes.append(node)  # append this bias unit to nodes array
            nodeIndex = nodeIndex + 1 # increment index         
                
            print(node.get_label(), "\t", end='')
            
            for j in range(hidden_nodes[i]): # how many current node X's from current hidden layer
                node = Node()
                node.set_level(i+1)
                node.set_label(("N["+str(i+1)+"]["+str(j+1)+"]"))
                node.set_index(nodeIndex)
                node.set_is_bias_unit(False)
                nodes.append(node)  # append this bias unit to nodes array
                nodeIndex = nodeIndex + 1 # increment index
            
                print(node.get_label(),"\t", end='')
            
            print("")
        #--------------------------------------------  
        # output layer
        
        node = Node()
        node.set_level(1+len(hidden_nodes))
        node.set_label("output")
        node.set_index(nodeIndex)
        node.set_is_bias_unit(False)
        nodes.append(node)  # append this bias unit to nodes array
        nodeIndex = nodeIndex + 1 # increment index       
        
        print("output layer: ",node.get_label())
        
        #--------------------------------------------  
        return nodes
    
