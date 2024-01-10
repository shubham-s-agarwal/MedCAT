
import json
from medcat.meta_cat import MetaCAT
from medcat.preprocessing.tokenizers import TokenizerWrapperBPE, TokenizerWrapperBERT
from tokenizers import ByteLevelBPETokenizer
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

model_arch_config = {'fc2': True, 'fc3': False,'lr_scheduler': False}

#Training the MetaCAT model
winner_report = mc.train(json_path= './Data/MedCAT_Export.json', save_dir_path='status_2',model_arch_config=model_arch_config)
print("\n\nWinner Report:",winner_report)

# print("PRINTING CONFUSION MATRIX FOR TRAIN DATASET")
# cm = confusion_matrix(y_train, np.argmax(np.concatenate(all_logits, axis=0), axis=1))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels={'Other': 0, 'Confirmed': 1})
# disp.plot()
# plt.show()

print("\n**************\nPRINTING CONFUSION MATRIX FOR TEST DATASET")
cm = winner_report['confusion_matrix']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels={'Other': 0, 'Confirmed': 1})
disp.plot()
plt.show()