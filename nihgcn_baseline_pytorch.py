# Setup

import os
import candle

from load_data import load_data
from sklearn.model_selection import KFold
from Internal.Single.NIHGCN_Single import nihgcn_single
from myutils import *
from sampler import TargetSampler
from model import nihgcn, Optimizer
from pathlib import Path

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

def run(params):
    #There is probably a better way of handling 'layer_size' and 'target_drug_cids'
    #params['layer_size']=[1024,1024]
    params['target_drug_cids'] = np.array(params['target_drug_cids'])
    res, drug_finger, exprs, null_mask, target_indexes, target_pos_num = load_data(params)
    #drug_sum = np.sum(res, axis=0)

    true_datas = pd.DataFrame()
    predict_datas = pd.DataFrame()

    output_path = os.path.join(params['data_dir'],params['output_dir']) #set to output directory
    # Check whether the specified path exists or not
    isExist = os.path.exists(output_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(output_path)
        print("The new directory is created!")

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
                           layer_size=params['dense'], alpha=params['alpha'], gamma=params['gamma'],
                           device=params['gpus'])
            print(sampler.train_data)
            print(sampler.test_data)
            opt = Optimizer(sampler.train_data, exprs, drug_finger, params['dense'], params['alpha'], params['gamma'], model,
                            sampler.train_data, sampler.test_data, sampler.test_mask, 
                            sampler.train_mask, evaluate_all, lr=params['learning_rate'], wd=params['weight_decay'],
                            epochs=params['epochs'], device=params['gpus']).to(params['gpus'])
            #save best model inside of the Optimizer in model_clone
            true_data, predict_data, model_clone, metrics = opt()
            true_datas = true_datas.append(translate_result(true_data))
            predict_datas = predict_datas.append(translate_result(predict_data))
            torch.save(model_clone,os.path.join(output_path,params['experiment_id']+"_best_model.pt"))
            pd.DataFrame(true_datas).to_csv(os.path.join(output_path,"true_data.csv"))
            pd.DataFrame(predict_datas).to_csv(os.path.join(output_path,"predict_data.csv"))
            break #ensures we do just one k-fold validation
    return metrics

def main():
    params = initialize_parameters()
    print(params['data_dir'])
    scores=run(params)
    print(scores)
    output_path = os.path.join(params['data_dir'],params['output_dir'])
    print("\nIMPROVE_RESULT val_loss:\t{}\n".format(scores["CrossEntropyLoss"]))
    with open(Path(output_path) / "scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    
if __name__=="__main__":
    main()
