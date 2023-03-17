# Setup

import os
import candle
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from load_data import load_data
from sklearn.model_selection import KFold
from Internal.Single.NIHGCN_Single import nihgcn_single
from myutils import *
from sampler import TargetSampler
from model import nihgcn, Optimizer

candle_data_dir = os.environ.get('CANDLE_DATA_DIR')
file_path = os.path.dirname(os.path.realpath(__file__))

class NIHGCN(candle.Benchmark):  # 1
    def set_locals(self):
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions

def initialize_parameters(default_model="default_class_model.txt"):

    # Build benchmark object
    NIHGCN_common = NIHGCN(
        file_path,
        "nihgcn_default_model.txt",
        "pytorch",
        prog="nihgcn",
        desc="from NIHGCN paper by Peng et al.",
    )


def predicting(model, params):
    res, drug_finger, exprs, null_mask, target_indexes, target_pos_num = load_data(params)
    params['target_drug_cids'] = np.array(params['target_drug_cids'])

    model.eval()
    with torch.no_grad():
        sampler = TargetSampler(response_mat=res, null_mask=null_mask, target_indexes=target_indexes,
                                pos_train_index=train_index, pos_test_index=test_index)
        model = nihgcn(adj_mat=sampler.train_data, cell_exprs=exprs, drug_finger=drug_finger,
                       layer_size=params['dense'], alpha=params['alpha'], gamma=params['gamma'],
                       device=params['gpus'])
        opt = Optimizer(sampler.train_data, exprs, drug_finger, params['dense'], params['alpha'], params['gamma'], model,
                        sampler.train_data, sampler.test_data, sampler.test_mask, 
                        sampler.train_mask, roc_auc, lr=params['learning_rate'], wd=params['weight_decay'],
                        epochs=params['epochs'], device=params['gpus']).to(params['gpus'])
        true_data, predict_data, model_clone = opt()
        true_datas = true_datas.append(translate_result(true_data))
        predict_datas = predict_datas.append(translate_result(predict_data))
    return true_datas,predict_datas

def load_model():
    

def main():
    params = initialize_parameters()
    print(params['output_dir'])
    output_path = os.path.join(params['data_dir'],params['output_dir']) #set to output directory

    model = model.load(os.path.join(output_path,params['experiment_id']+"_best_model.pt"))
    true_datas,predict_datas = predicting(model,params)
    print(predict_datas)
    
