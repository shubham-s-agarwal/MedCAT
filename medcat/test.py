import json
from medcat.meta_cat import MetaCAT
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DATA_DIR = "MedCAT/data/"

model_ = 'lstm'
tokenizer_ = 'bbpe'

model_ = 'bert'
tokenizer_ = 'bert-tokenizer'

load_model_dict_ = False
fine_tune_two_phase = False
category_undersample = 'Other'
train_on_full_data = False

data = json.load(open('./Data/MedCAT_Export.json'))
mc_config = json.load(open('./Data/status/config.json'))

mc_config['general']['tokenizer_name'] = tokenizer_
mc_config['model']['model_name'] = model_
mc_config['model']['model_variant'] = 'bert-base-uncased'
mc_config['model']['load_model_dict_'] = load_model_dict_
mc_config['model']['fine_tune_two_phase'] = fine_tune_two_phase
mc_config['model']['category_undersample'] = category_undersample
mc_config['model']['train_on_full_data'] = train_on_full_data
mc_config['train']['test_size'] = 0.2

json.dump(mc_config, open("./Data/status/config.json", 'w'))

#print(mc_config)

mc = MetaCAT.load(save_dir_path='./Data/status',load_model_dict_=load_model_dict_)

mc.config.model['input_size'] = 768
mc.config.model['input_size'] = 1536
mc.config.model['hidden_size'] = 16
mc.config.model['hidden_size'] = 32
mc.config.general.cntx_left = 20
mc.config.general.cntx_right = 10

mc.config.general['tokenizer_name'] = tokenizer_
mc.config.train['nepochs'] = 70
# mc.config.train['nepochs'] = 20
mc.config.train['auto_save_model'] = False
mc.config.train.batch_size = 64
mc.config.model.model_name = model_
mc.config.model["nclasses"] = 2
mc.config.model['dropout'] = 0.25
mc.config.train.lr = 5e-4
mc.config.train.gamma = 3
mc.config.train.class_weights = [0.15, 0.72]
mc.config.train.class_weights = [0.5, 0.6]

mc.config.train.metric['base'] = 'macro avg'
mc.config.model.model_freeze_layers = False
mc_config['model']['model_variant'] = 'bert-base-uncased'
mc_config['model']['load_model_dict_'] = load_model_dict_

mc_config['model']['category_undersample'] = category_undersample

mc_config['model']['fine_tune_two_phase'] = fine_tune_two_phase
mc_config['model']['train_on_full_data'] = train_on_full_data

mc.config.model['num_layers'] = 3

mc.config.general[ "category_value2id"] = {"Other": 1,"Confirmed": 0}

mc.config.train['loss_function'] = 'cross_entropy'

model_arch_config = {'fc2': True, 'fc3': False,'lr_scheduler': True}
_data = None

winner_report = mc.train(json_path= './Data/MedCAT_Export.json', save_dir_path='./Data/status',model_arch_config=model_arch_config,data_ = _data)

#Training the MetaCAT model
# winner_report = mc.train(json_path= './Data/MedCAT_Export.json', save_dir_path='./Data/status',model_arch_config=model_arch_config)
print("\n\nWinner Report:",winner_report)

print("\n**************\nPRINTING CONFUSION MATRIX FOR TEST DATASET")
cm = winner_report['confusion_matrix']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels={"Confirmed": 0,"Other": 1})
disp.plot()
plt.show()
