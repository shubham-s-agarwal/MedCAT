import pandas as pd
import numpy as np
import seaborn as sns
import json
# from matplotlib import pyplot as plt
# from medcat.cat import CAT
# from medcat.cdb import CDB
# from medcat.config import Config
# from medcat.vocab import Vocab
from medcat.meta_cat import MetaCAT
# from medcat.config_meta_cat import ConfigMetaCAT
# from medcat.preprocessing.tokenizers_med import TokenizerWrapperBPE, TokenizerWrapperBERT
# from tokenizers_med import ByteLevelBPETokenizer

DATA_DIR = "MedCAT/data/"

data = json.load(open(DATA_DIR + "MedCAT_Export.json"))
mc = MetaCAT.load('C:\\Users\\yuvas\\PycharmProjects\\SA_KCL\\MedCAT\\data\\mc_status\\Status')

mc.config.model['input_size'] = 768
mc.config.model['hidden_size'] = 300

mc.config.train['nepochs'] = 55
mc.config.train['auto_save_model'] = True

# Training the MetaCAT model
mc.train(json_path= DATA_DIR+"MedCAT_Export.json", save_dir_path='status')