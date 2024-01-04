
import json
from medcat.meta_cat import MetaCAT
from medcat.preprocessing.tokenizers import TokenizerWrapperBPE, TokenizerWrapperBERT
from tokenizers import ByteLevelBPETokenizer

DATA_DIR = "MedCAT/data/"

data = json.load(open('./Data/MedCAT_Export.json'))
mc_config = json.load(open('./Data/status/config.json'))

updated_config = {'general':{'tokenizer_name':'bert-tokenizer'},'model':{'model_name':'bert','model_variant':'distilbert-base-uncased'}}

mc = MetaCAT.load('./Data/status',updated_config)

mc.config.model['input_size'] = 768
mc.config.model['hidden_size'] = 128
mc.config.general['tokenizer_name'] = 'bert-tokenizer'
mc.config.train['nepochs'] = 100
mc.config.train['auto_save_model'] = True

#mc.config.model.model_name = 'bert'
mc.config.model["nclasses"] = 2

#Training the MetaCAT model
winner_report = mc.train(json_path= './Data/MedCAT_Export.json', save_dir_path='status')
print("\n\nWinner Report:",winner_report)