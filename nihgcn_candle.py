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
]

required = ['learning_rate',
            'epochs',
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
        "nihgcn_params.txt",
        "pytorch",
        prog="nihgcn",
        desc="from NIHGCN paper by Peng et al.",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(NIHGCN_common)
    return gParameters

def run(params):
    #Notes for future:
    #  *The way this code words with multiple files from gdsc/ccle/tcga/pdx is that it takes
    #  the same data tables from different sources and concatenates the input data. Figuring
    #  out the data will be pushed off until someone else sorts it out.
    #  [--Alex Partin approves this message]
    #  *candle_glue.sh is being use as a placeholder for putting the GDSC test data in the
    #  right place. Goal is to get train.sh running first, then to worry about the data side
    #  after some discussion with data team [led by Yitan] and CANDLE dev team [Jamal].
    #  [--Andreas Wilke approves this message, with the promise that Chia will bring up
    #  CANDLE requirements as he finds them]
    #  *additional CANDLE requirements to consider:
    #    -dealing with multiple data input files (i.e., expression data, drug response data)
    #    and not a singular training data file (as it is currently set up)
    #Below not in nihgcn_params.txt yet because there is probably a better way of doing it
    params['layer_size']=[1024,1024]
    #parser.add_argument('--layer_size', nargs='?', default=[1024,1024],
    #                    help='Output sizes of every layer')
    
    params['target_drug_cids'] = np.array([5330286, 11338033, 24825971])
    #load data
    res, drug_finger, exprs, null_mask, target_indexes, target_pos_num = load_data(params)
    #drug_sum = np.sum(res, axis=0)

    true_datas = pd.DataFrame()
    predict_datas = pd.DataFrame()

    #Note: Original n_kfolds loop left here, but setting to 1 and waiting to see if multiple runs will
    #      be defined by candle or inside of each code. 
    n_kfolds = 1
    k = 5 #80-20 split
    for fold in range(n_kfolds):
        kfold = KFold(n_splits=k, shuffle=True, random_state=fold)
        for train_index, test_index in kfold.split(np.arange(target_pos_num)):
            sampler = TargetSampler(response_mat=res, null_mask=null_mask, target_indexes=target_indexes,
                                    pos_train_index=train_index, pos_test_index=test_index)
            model = nihgcn(adj_mat=sampler.train_data, cell_exprs=exprs, drug_finger=drug_finger,
                           layer_size=params['layer_size'], alpha=params['alpha'], gamma=params['gamma'],
                           device=params['gpus'])
            opt = Optimizer(model, sampler.train_data, sampler.test_data, sampler.test_mask,
                            sampler.train_mask, roc_auc, lr=params['learning_rate'], wd=params['weight_decay'],
                            epochs=params['epochs'], device=params['gpus']).to(params['gpus'])
            true_data, predict_data = opt()
            true_datas = true_datas.append(translate_result(true_data))
            predict_datas = predict_datas.append(translate_result(predict_data))
            #save best model here
            
            break #ensures we do just one

    output_path = os.path.join(params['data_dir'],params['output_dir']) #set to output directory
    # Check whether the specified path exists or not
    isExist = os.path.exists(output_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(output_path)
        print("The new directory is created!")
    pd.DataFrame(true_datas).to_csv(os.path.join(output_path,"true_data.csv"))
    pd.DataFrame(predict_datas).to_csv(os.path.join(output_path,"predict_data.csv"))


def main():
    params = initialize_parameters()
    print(params['data_dir'])
    run(params)
    print("Success so far!")

if __name__=="__main__":
    main()
