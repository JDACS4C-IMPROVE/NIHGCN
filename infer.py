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
from model import nihgcn, Optimizer, EvalRun

candle_data_dir = os.environ.get('CANDLE_DATA_DIR')
file_path = os.path.dirname(os.path.realpath(__file__))

additional_definitions = [
   {'name':'alpha',
       'nargs':'+',
       'type': float,
       'help':'message passing weight'},
    {'name':'gamma',
       'nargs':'+',
       'type': int,
       'help':'not sure'},
    {'name':'weight_decay',
       'nargs':'+',
       'type': float,
       'help':'some kind of weight decay'},
    {'name':'target_drug_cids',
       'nargs':'+',
       'type':int,
       'help':'Drug target IDs to hold out as part of the test set'},
]

required = ['learning_rate',
            'epochs',
            'dense'
            ]

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
    # Initialize parameters
    gParameters = candle.finalize_parameters(NIHGCN_common)
    return gParameters


def predicting(model, params):
    res, drug_finger, exprs, null_mask, target_indexes, target_pos_num = load_data(params)
    params['target_drug_cids'] = np.array(params['target_drug_cids'])
    #print(target_pos_num)
    #print(params['target_drug_cids'])

    model.eval()
    with torch.no_grad():
        sampler = TargetSampler(response_mat=res, null_mask=null_mask, target_indexes=target_indexes,
                                pos_train_index=np.arange(target_pos_num), pos_test_index=np.arange(target_pos_num))
        print(sampler.train_data)
        print(sampler.test_data)
        opt = EvalRun(model,sampler.test_data,sampler.test_mask,roc_auc, device=params['gpus']).to(params['gpus'])
        true_data, predict_data, eval_result = opt()
        true_datas = translate_result(true_data)
        predict_datas = translate_result(predict_data)
    return true_datas,predict_datas

def main():
    params = initialize_parameters()
    #print(params['output_dir'])
    output_path = os.path.join(params['data_dir'],params['output_dir']) #set to output directory

    model = torch.load(os.path.join(output_path,params['experiment_id']+"_best_model.pt"))
    true_datas,predict_datas = predicting(model,params)
    print(true_datas)
    print(predict_datas)

if __name__=="__main__":
    main()
