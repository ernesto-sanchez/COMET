





class Evaluate:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    def evaluate(self):
        patients, organs, outcomes, outcomes_noiseless, effects = self.data_loader.load_data()




        #If clustering is needed
        if self.clustering:
            organs = encode_clustering(organs)
        model = self.model




if __name__ == "__main__":
    pass
    # evaluate = Evaluate(model, data_loader)
    # evaluate.evaluate()
    # evaluate.plot()
    # evaluate.save_results()
    



