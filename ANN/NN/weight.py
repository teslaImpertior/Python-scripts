# define weight entity
class Weight:
    # get and set for weight index
    def get_weight_index(self):
        return self.__weight_index
    
    def set_weight_index(self, weight_index):
        self.__weight_index = weight_index
        
    # node index
    def get_from_index(self):
        return self.__from_index
    
    def set_from_index(self, from_index):
        self.__from_index = from_index
        
    def get_to_index(self):
        return self.__to_index
    
    def set_to_index(self, to_index):
        self.__to_index = to_index
     
    # store value of weights    
    def get_value(self):
        return self.__value
    
    def set_value(self, value):
        self.__value = value