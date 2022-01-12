import numpy as np
import pandas as pd
from keras.models import load_model
def get_threshold_metrics(y_true, y_pred, drop_intermediate=False,
                          disease='all'):
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.metrics import precision_recall_curve, average_precision_score

    roc_columns = ['fpr', 'tpr', 'threshold']
    pr_columns = ['precision', 'recall', 'threshold']

    if drop_intermediate:
        roc_items = zip(roc_columns,
                        roc_curve(y_true, y_pred, drop_intermediate=False))
    else:
        roc_items = zip(roc_columns, roc_curve(y_true, y_pred))

    roc_df = pd.DataFrame.from_dict(dict(roc_items))

    prec, rec, thresh = precision_recall_curve(y_true, y_pred)
    pr_df = pd.DataFrame.from_records([prec, rec]).T
    pr_df = pd.concat([pr_df, pd.Series(thresh)], ignore_index=True, axis=1)
    pr_df.columns = pr_columns

    auroc = roc_auc_score(y_true, y_pred, average='weighted')
    aupr = average_precision_score(y_true, y_pred, average='weighted')

    return {'auroc': auroc, 'aupr': aupr, 'roc_df': roc_df,
            'pr_df': pr_df, 'disease': disease}

def re_result(modelpath):
    pred_all_list = []
    iter = 5
    h5 = pd.HDFStore(modelpath+"x_test_all.h5",mode='r')
    x_test_all = h5.get('x')
    h5.close()

    h5 = pd.HDFStore(modelpath+"y_test_all.h5", mode='r')
    y_test_all = h5.get('y')
    h5.close()

    for i in range (iter):
        print(i+1)
        mm = load_model(modelpath+"my_model{}.h5".format(i+1))
        pred_all = mm.predict(x_test_all)
        pred_all_list.append(pred_all.reshape(-1))

    pred_all_mean = np.mean(pred_all_list, axis=0)
    print(pred_all_mean.shape)
    metrics3 = get_threshold_metrics(y_test_all, pred_all_mean, drop_intermediate=False,
                                    disease='all')
    print(metrics3)
    roc_all = metrics3['auroc']
    return roc_all

datasetname = ["balance","imbalance"]
for name in datasetname:
    modelpath = 'model/' + name +  '/'
    print(name)
    fit, m = re_result(modelpath)
    print(name)
    print('final fitness:{}', format(fit))
    print('final model')
    print(m)
