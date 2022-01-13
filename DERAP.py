import os
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold,train_test_split
import numpy as np
import pandas as pd
import math
from keras.models import Model
from keras.layers import Dense, Dropout, Input,BatchNormalization
from keras.callbacks import EarlyStopping
import keras.losses
import keras
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from random import shuffle
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
def densemodel(inputdim):
    keras.backend.clear_session()
    input = Input(shape=(inputdim,))

    dense_0 = Dense(512,activation='relu')(input)
    bn_0 = BatchNormalization()(dense_0)
    dropout_0 = Dropout(0.5)(bn_0)

    dense_1 = Dense(256,activation='relu')(dropout_0)
    bn_1 = BatchNormalization()(dense_1)
    dropout_1 = Dropout(0.5)(bn_1)

    dense_2 = Dense(128, activation='relu')(dropout_1)
    bn_2 = BatchNormalization()(dense_2)
    dropout_2 = Dropout(0.5)(bn_2)

    dense_3 = Dense(64,activation='relu')(dropout_2)
    bn_3 = BatchNormalization()(dense_3)
    dropout_3 = Dropout(0.5)(bn_3)

    dense_4 = Dense(8, activation='relu')(dropout_3)
    bn_4 = BatchNormalization()(dense_4)
    dropout_4 = Dropout(0.5)(bn_4)

    final_1 = Dense(1, activation='sigmoid')(dropout_4)
    model = Model(inputs=input, outputs=final_1)
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-04),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def fit(X_train,X_test,Y_train,Y_test):
    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=10, verbose=0,
                                                                epsilon=0.0003, factor=0.9, min_lr=1e-6)
    inputdim = X_train.shape[1]
    model = densemodel(inputdim)
    result = model.fit(X_train, Y_train,
                       validation_data=(X_test, Y_test),
                       epochs=400,
                       batch_size=128,
                       callbacks=[EarlyStopping(monitor='val_loss', patience=50, verbose=0),learning_rate_reduction],
                       verbose=0)
    loss, accurancy = model.evaluate(X_test, Y_test, verbose=0)
    return accurancy,model

def imbalanceprocess(x,y):
    st = math.ceil((int(pd.DataFrame(y).value_counts()[1]) / int(pd.DataFrame(y).value_counts()[0]) * 10)) / 10

    over = RandomOverSampler(sampling_strategy=st)
    under = RandomUnderSampler(sampling_strategy=0.8)

    from imblearn.pipeline import Pipeline

    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X_smote, y_smote = pipeline.fit_resample(x, y)

    return X_smote, y_smote
def Stratified_sampling(y):
    sample_index = []

    y = pd.DataFrame(y, columns=['lable'])
    tags = y['lable'].unique().tolist()

    for tag in tags:
        data = y[(y['lable'] == tag)]
        sample = data.sample(int(0.9 * len(data)))
        _ = sample.index.tolist()
        sample_index.extend(_)

    shuffle(sample_index)
    return sample_index
def train(x_train,x_test,y_train,y_test,n):
    print('---------------model{}----------------'.format(n))

    index = Stratified_sampling(y_train)

    x_train = x_train[index, :]
    y_train = y_train[index, :]

    X_smote, y_smote = imbalanceprocess(x_train, y_train)

    acc,mm = fit(X_smote,x_test,y_smote,y_test)
    pred = mm.predict(x_test)
    metrics1 = get_threshold_metrics(y_test, pred, drop_intermediate=False,
                                    disease='all')
    print(metrics1)
    return acc,pred,mm
def cross_valadition(x,y,strat,modelpath,resultpath):
    print("cross valadition --------------------")
    k_size = 5
    best_roc = 0
    best_model = []
    kf = 0
    iter = 5
    ch = 0
    x_train_all, x_test_all, y_train_all, y_test_all = train_test_split(pd.DataFrame(x), pd.DataFrame(y), test_size=0.1,
                                                                        random_state=0, stratify=strat)
    k_fold = StratifiedKFold(k_size, True, random_state=1)
    index = k_fold.split(X=x_train_all, y=y_train_all)
    for train_index, test_index in index:
        kf = kf + 1
        print('-----------{} fold-----------------'.format(kf))
        x_train = np.array(x_train_all.iloc[train_index, :])
        x_test = np.array(x_train_all.iloc[test_index, :])
        y_train = np.array(y_train_all.iloc[train_index, :])
        y_test = np.array(y_train_all.iloc[test_index, :])

        pred_list = []
        mm_list = []
        for i in range (iter):
            acc,pred,mm = train(x_train, x_test, y_train, y_test,i+1)
            pred_list.append(pred.reshape(-1))
            mm_list.append(mm)
            mm.save(modelpath+"{}my_model{}.h5".format(kf,i+1))

        pred_mean = np.mean(pred_list, axis=0)
        metrics2 = get_threshold_metrics(y_test, pred_mean, drop_intermediate=False,
                                        disease='all')
        print(metrics2)
        roc = metrics2['auroc']
        if roc>best_roc:
            best_roc=roc
            best_model=mm_list
            ch = kf

    print('-----{}------best_roc{}------------------------------'.format(ch,best_roc))
    pred_test_all_list = []
    for i in range(iter):
        mm = load_model(modelpath+str(ch)+"my_model{}.h5".format(i + 1))
        pred_test_all = mm.predict(x_test_all)
        pred_test_all_list.append(pred_test_all.reshape(-1))
    pred_test_all_mean = np.mean(pred_test_all_list, axis=0)
    test_metrics = get_threshold_metrics(y_test_all, pred_test_all_mean, drop_intermediate=False,
                                     disease='all')
    print(test_metrics)
    test_roc_all = test_metrics['auroc']
    np.save(resultpath+"test_metrics.npy", test_metrics)
    return test_roc_all,best_model,ch

def AE_train(x,y,batch_size):

    from keras import Model
    from keras.layers import Input, Dense
    import keras as ks
    ks.backend.clear_session()
    input_dim = x.shape[1]
    input = Input(shape=(input_dim,))

    encoded = Dense(50, kernel_constraint=ks.constraints.NonNeg(), kernel_regularizer=ks.regularizers.l1(0.001))(input)
    decoded = Dense(input_dim, activation='relu')(encoded)

    autoencoder = Model(inputs=input, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(x, x, batch_size=batch_size, epochs=50, verbose=0, shuffle=True,
                    )
    t = autoencoder.get_layer('dense_1')

    w = t.get_weights()[0]
    w[np.isnan(w)] = 0
    w[np.isinf(w)] = 0
    wsd = np.std(w, axis=1)
    wsd[np.isnan(wsd)] = 0
    wsd[np.isinf(wsd)] = 0
    wsd = (wsd - wsd.min()) / (wsd.max() - wsd.min())
    ks.backend.clear_session()
    return wsd
def fs_multiprocess2(data,label,n_feature,iter=3):

    batch_size = int(np.ceil(data.shape[0] / 50))
    if batch_size < 2:
        batch_size = 2
    w=[]
    for i in range (iter):
        if data.shape[0] > 2000:
            row_rand_array = np.arange(data.shape[0])
            np.random.shuffle(row_rand_array)
            x = data[row_rand_array[0:2000]]
            y = label[row_rand_array[0:2000]]
        else:
            x = np.array(data)
            y = np.array(label)

        w.append(AE_train(x,y,batch_size=batch_size))

    wsd=np.array(w)
    wsd = np.average(wsd, axis=0)
    sort_index = np.argsort(-np.array(wsd))
    index = sort_index[:n_feature]
    return index
def load_data(path,genenumber):
    print('load_data...')

    h5 = pd.HDFStore(path+'_data.h5'
                     , mode='r')
    x = h5.get('x')
    h5.close()
    h5 = pd.HDFStore(path+'_label.h5'
                     , mode='r')
    y = h5.get('y')
    h5.close()
    h5 = pd.HDFStore(path+'_strat.h5'
                     , mode='r')
    strat = h5.get('strat')
    h5.close()

    x = StandardScaler().fit_transform(x)
    index = fs_multiprocess2(x,np.array(y),genenumber,iter=3)

    return x[:, np.array(index)],np.array(y).reshape(-1),np.array(strat).reshape(-1)

def main():
    datasetname = ["balance","imbalance"]
    for name in datasetname:
        print(name)
        loadpath = name+ '/' +name
        modelpath = 'model/' + name + '/'
        resultpath = 'result/' + name + '/'

        if not os.path.exists(modelpath):
            os.makedirs(modelpath)
        if not os.path.exists(resultpath):
            os.makedirs(resultpath)

        x, y, strat = load_data(loadpath,2048)

        print(x.shape)
        print(y.shape)
        print(strat.shape)

        fit,m,ch = cross_valadition(x,y,strat,modelpath,resultpath)
        print(name)
        print('final fitness:{}'.format(fit))
        print('final {}my_model'.format(ch))
        print(m)

if __name__ == "__main__":
    main()
