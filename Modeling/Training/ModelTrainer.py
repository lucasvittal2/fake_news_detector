import numpy as np
from Environment.Parameters import SEED
from Utils.GraphicPlotter import GraphicPlotter
from  Environment.PathsParameters import TRAIN_HIST_ASSET_PATH, MODELS_PATH
from matplotlib.pyplot import title

class ModelTrainer():
    
    def __init__(self, epochs=40):
        self.sim_metrics = {'mean_accuracy': {}, 'mean_loss': {}, 'mean_val_accuracy': {}, 'mean_val_loss': {}}
        self.epochs = epochs

    def save_metrics(self, train_hist, model_name):
        #save mean metrics
        mean_acc = np.mean(train_hist.history['accuracy'])
        mean_loss = np.mean(train_hist.history['loss'])
        mean_val_acc = np.mean(train_hist.history['val_accuracy'])
        mean_val_loss=  np.mean(train_hist.history['val_loss'])
        
        
        self.sim_metrics['mean_accuracy'].update({model_name: mean_acc})
        self.sim_metrics['mean_loss'].update({model_name: mean_loss})
        self.sim_metrics['mean_val_accuracy'].update({model_name: mean_val_acc})
        self.sim_metrics['mean_val_loss'].update({model_name:mean_val_loss})
        
    def get_sim_metrics(self):
        return self.sim_metrics
    
    def train_model (self, algorithms, X_train, y_train):
        np.random.seed(SEED)
        grapPlotter = GraphicPlotter()
        for name, alg in algorithms:
            
                print('*'*150)
                print(f'Training {name} ...')

                if "EnsembleCNN" not in name:
                    train_hist = alg.fit(X_train, y_train, validation_split=0.2, epochs=40, batch_size=32, verbose=1)
                    
                else:
                    train_hist = alg.fit([X_train, X_train] , y_train, validation_split=0.2, epochs=self.epochs, batch_size=32, verbose=1)
                    
                print(f"Model {name} is trained !!")
                print('-'*150)
                
                print('Saving training data...\n')
                
                print('Saving figure...')    
                #save train hit fig    
                fig = grapPlotter.plot_train_history(train_hist, name)
                fig.savefig(TRAIN_HIST_ASSET_PATH + name + '.png')
                
                
                print('Figure saved !!')
                
                print('Saving metrics...')
                #save mean metrics
                self.save_metrics(train_hist,  name)
                print('Metrics Saved !!')
                #save the model
                print(f'Saving  the {name} model...')
                alg.save(MODELS_PATH + name + '.h5')
                print(f'{name} model saved !!')
            
       