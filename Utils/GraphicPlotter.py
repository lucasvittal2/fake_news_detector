import matplotlib.pyplot as plt
from Environment.Parameters import *
from tensorflow.python.keras.callbacks import History

class GraphicPlotter():
    
    def __init__(self):
        pass
    
    
    def plot_train_history(self, train_history: History, model_name: str):
        
        
        
        #plot accuracy hist
        fig, axs = plt.subplots(1,2,figsize=(40,15))
        fig.set_facecolor('white')
        axs[0].plot(train_history.history['accuracy'])
        axs[0].plot(train_history.history['val_accuracy'])
        axs[0].set_title('model accuracy', fontsize=TITLE_SIZE)
        axs[0].set_ylabel('accuracy',fontsize=FONT_SIZE)
        axs[0].set_xlabel('epoch', fontsize=FONT_SIZE)
        axs[0].legend(['train', 'test'], loc='upper left')


        #plot loss hist
        axs[1].plot(train_history.history['loss'])
        axs[1].plot(train_history.history['val_loss'])
        axs[1].set_title('model Loss', fontsize=TITLE_SIZE)
        axs[1].set_ylabel('Loss',fontsize=FONT_SIZE)
        axs[1].set_xlabel('Epoch', fontsize=FONT_SIZE)
        axs[1].legend(['train', 'test'], loc='upper left')
        
        fig.suptitle(f'{model_name} Train History', fontsize=40)

        return fig