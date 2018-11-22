class Node:
    # get and set for index variable
    def get_index(self):
        return self.__index
    
    def set_index(self, index):
        self.__index = index
    
    # get and set for label    
    def get_label(self):
        return self.__label
    
    def set_label(self, label):
        self.__label = label
        
    # get and set for bias unit   
    def get_is_bias_unit(self):
        return self.__is_bias_unit
    
    def set_is_bias_unit(self, is_bias_unit):
        self.__is_bias_unit = is_bias_unit
        
    #--------------------------------------
    # store layer names 
    def get_level(self):
        return self.__level
    
    def set_level(self, level):
        self.__level = level 