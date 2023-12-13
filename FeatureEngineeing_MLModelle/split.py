
class DataSplitter:
    def __init__(self, data, split_ratio=0.8):
    
        self.data = data
        self.split_ratio = split_ratio
        self.train_data = None
        self.test_data = None

    def split(self): #Split der Daten in Train und Test
        split_index = int(len(self.data) * self.split_ratio)
        self.train_data = self.data.iloc[:split_index]
        self.test_data = self.data.iloc[split_index:]

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

