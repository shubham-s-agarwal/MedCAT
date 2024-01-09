
import json
from medcat.meta_cat import MetaCAT
from medcat.preprocessing.tokenizers import TokenizerWrapperBPE, TokenizerWrapperBERT
from tokenizers import ByteLevelBPETokenizer

DATA_DIR = "MedCAT/data/"

model_ = 'lstm'
tokenizer_ = 'bbpe'

model_ = 'bert'
tokenizer_ = 'bert-tokenizer'

data = json.load(open('./Data/MedCAT_Export.json'))
mc_config = json.load(open('./Data/status/config.json'))

mc_config['general']['tokenizer_name'] = tokenizer_
mc_config['model']['model_name'] = model_
json.dump(mc_config, open("config.json", 'w'))

#print(mc_config)

mc = MetaCAT.load('./Data/status')

mc.config.model['input_size'] = 768
mc.config.model['hidden_size'] = 256
mc.config.general.cntx_left = 25
mc.config.general.cntx_right = 10

mc.config.general['tokenizer_name'] = tokenizer_
mc.config.train['nepochs'] = 20
mc.config.train['auto_save_model'] = True
mc.config.train.batch_size = 32
mc.config.model.model_name = model_
mc.config.model["nclasses"] = 2
mc.config.model['dropout'] = 0.35
# mc.config.train.lr = 5e-5

mc.config.model['num_layers'] = 1

#Training the MetaCAT model
winner_report = mc.train(json_path= './Data/MedCAT_Export.json', save_dir_path='status_2')
print("\n\nWinner Report:",winner_report)