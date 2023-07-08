
#Dataset Paths

BASE_PROJECT_PATH = 'C:/Users/lucas/Documents/Studying/Pos-graduacao - Ciencia de Dados e  Analytics/Sprints/Sprint II - Machine Learning and Advanced Analytics/Fake News Detection'
LOGS_PATH= BASE_PROJECT_PATH + '/Docs/logs/'
DATASET_PATH = BASE_PROJECT_PATH + '/Dataset/'
PREPROCESSED_DATA_PATH = DATASET_PATH + 'Preprocessed/'
PREPROCESSED_DATA_PARAMS_PATH  = PREPROCESSED_DATA_PATH + '/Params/'
TRUE_NEWS_DATASET = DATASET_PATH + 'True.csv'
FAKE_NEWS_DATASET= DATASET_PATH + 'Fake.csv'
WEL_FAKE_DATASET=DATASET_PATH +'WELFake_Dataset.csv'


# Main Modules paths

MAIN_PROJECT_MODULES = [ 'PreProcessing', '']

# PretTrained Model

GOOGLE_PRETRAINED_MODEL_PATH = BASE_PROJECT_PATH + '/PreTrainedModels/GoogleNews-vectors-negative300.bin'

#PATHS for modeling

TRAIN_HIST_ASSET_PATH = BASE_PROJECT_PATH + '/Assets/FigsTrainHist/'
MODELS_PATH = BASE_PROJECT_PATH + '/Code/Modeling/Training/TrainedModels/'
EXPERIMENTATION_TRAIN_PATH = MODELS_PATH + '/Experimentation/'
TUNNELD_TRAIN_MODEL_PATH = MODELS_PATH + '/Tunnelled/'
TUNNELED_MODELS_PATH = BASE_PROJECT_PATH + '/Code/Modeling/Tunelling/TunelledModels/'
