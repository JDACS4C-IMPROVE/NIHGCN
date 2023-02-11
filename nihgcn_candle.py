# Setup

import os
import candle

from load_data import load_data
from sklearn.model_selection import KFold
from Internal.Single.NIHGCN_Single import nihgcn_single
from myutils import *

file_path = os.path.dirname(os.path.realpath(__file__))

additional_definitions = None
required = None

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
    #some arguments not added yet
    #parser.add_argument('-data', type=str, default='gdsc', help='Dataset{gdsc,ccle}')
    #parser.add_argument('--layer_size', nargs='?', default=[1024,1024],
    #                    help='Output sizes of every layer')
    

    res, drug_finger, exprs, null_mask, pos_num = load_data(params)
    drug_sum = np.sum(res, axis=0)
"""
k = 5
n_kfolds = 5

for target_index in np.arange(res.shape[1]):
    times = []
    true_data_s = pd.DataFrame()
    predict_data_s = pd.DataFrame()
    target_pos_index = np.where(res[:, target_index] == 1)[0]
    if drug_sum[target_index] < 10:
        continue
    for fold in range(n_kfolds):
        kfold = KFold(n_splits=k, shuffle=True, random_state=fold)
        start = time.time()
        for train, test in kfold.split(target_pos_index):
            train_index = target_pos_index[train]
            test_index = target_pos_index[test]
            true_data, predict_data = nihgcn_single(cell_exprs=exprs,
                                                              drug_finger=drug_finger, res_mat=res,
                                                              null_mask=null_mask, target_index=target_index,
                                                              train_index=train_index, test_index=test_index,
                                                              evaluate_fun=roc_auc, args=args)
            true_data_s = true_data_s.append(translate_result(true_data))
            predict_data_s = predict_data_s.append(translate_result(predict_data))
        end = time.time()
        times.append(end - start)
    path = "./results_data" #set to output directory
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")
    true_data_s.to_csv(path + "/noparallel_drug_" + str(target_index) + "_" + "true_data.csv")
    predict_data_s.to_csv(path + "/noparallel_drug_" + str(target_index) + "_" + "predict_data.csv")
"""

def main():
    params = initialize_parameters()
    print(params['data_dir2'])
    run(params)
    print("Success so far!")

if __name__=="__main__":
    main()
