class Xy_DataSplitter:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_into_features_and_target(self, target_column):
        # Keine Notwendigkeit, die Indexbedingungen erneut zu überprüfen
        self.X_train = self.train_data.copy()
        self.X_test = self.test_data.copy()    
        
        # Direktes Zuweisen der Zielvariablen
        if target_column == 'close':
            self.y_train = self.train_data['close_PctChange']
            self.y_test = self.test_data['close_PctChange']

            delete_columns = ['open', 'high', 'low', 'close', 'volume', 
                        'open_PctChange', 'high_PctChange', 'low_PctChange', 'close_PctChange', 'volume_PctChange']
            self.X_train.drop(delete_columns, axis=1, inplace=True)
            self.X_test.drop(delete_columns, axis=1, inplace=True)
        
        # Für die Zielvariable 'open' (falls benötigt)
        if target_column == 'open':
            pass

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test
