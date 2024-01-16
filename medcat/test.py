
import json
from medcat.meta_cat import MetaCAT
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
mc.config.model['hidden_size'] = 128
mc.config.general.cntx_left = 25
mc.config.general.cntx_right = 10

mc.config.general['tokenizer_name'] = tokenizer_
mc.config.train['nepochs'] = 15
mc.config.train['auto_save_model'] = True
mc.config.train.batch_size = 32
mc.config.model.model_name = model_
mc.config.model["nclasses"] = 2
mc.config.model['dropout'] = 0.35
mc.config.train.lr = 5e-4
mc.config.train.gamma = 3
mc.config.train.class_weights = [0.65, 0.35]
mc.config.train.metric['base'] = 'macro avg'
mc.config.model.model_freeze_layers = False

mc.config.model['num_layers'] = 1

model_arch_config = {'fc2': True, 'fc3': True,'lr_scheduler': False}

#Training the MetaCAT model
winner_report = mc.train(json_path= './Data/MedCAT_Export.json', save_dir_path='status_2',model_arch_config=model_arch_config)
print("\n\nWinner Report:",winner_report)

print("\n**************\nPRINTING CONFUSION MATRIX FOR TEST DATASET")
cm = winner_report['confusion_matrix']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels={'Other': 0, 'Confirmed': 1})
disp.plot()
plt.show()