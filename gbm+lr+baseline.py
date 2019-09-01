# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from multiprocessing import Pool
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import warnings
from sklearn.linear_model import LogisticRegressionCV
# from keras.models import Sequential
# from keras.layers import Dense
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.metrics import roc_auc_score

# ------path --------
path = './data/'
path_train_03 = path + 'clean_data/clean_data2019-03-31.csv'
path_train_04 = path + 'clean_data/clean_data2019-04-28.csv'
path_train_05 = path + 'clean_data/clean_data2019-06-02.csv'

# --------goal
path_this_month_0331 = './data/目标函数/2019-01-01_2019-03-31_goal.csv'
path_this_month_0428 = './data/目标函数/2019-02-11_2019-04-28_goal.csv'
path_this_month_0602 = './data/目标函数/2019-04-25_2019-06-02_goal.csv'

path_goal_0331_next_month = './data/目标函数/next_month_2019-03-31_2019-04-30_goal.csv'
path_goal_0428_next_month = './data/目标函数/next_month_2019-04-28_2019-05-31_goal.csv'
path_goal_0602_next_month = './data/目标函数/next_month_2019-06-02_2019-06-30_goal.csv'

# ---------fund-------------
path_fund_score_stock = path + '基金标签/用户基金评分股票.csv'
path_fund_score_no_stock = path + '基金标签/用户基金评分非股票.csv'
# ---------fund score--------
path_fund_score_03 = path + '用户对基金的评分（购买和浏览）/pre_deal_feature_data_2019-03-31_total.csv'
path_fund_score_04 = path + '用户对基金的评分（购买和浏览）/pre_deal_feature_data_2019-04-28_total.csv'
path_fund_score_05 = path + '用户对基金的评分（购买和浏览）/pre_deal_feature_data_2019-06-02_total.csv'


# 1\预测用户接下里一个月是否会购买
# 2\预测用户接下里一个月是否会点击

def baseline_ctr(train, this_month, next_month, fund_score_stock, month_str, pre_class):
    # 读数据
    df_train = pd.read_csv(train, dtype={'custno': str})
    df_this_three_month = pd.read_csv(this_month, dtype={'custno': str, 'fundcode': str})
    df_next_month = pd.read_csv(next_month, dtype={'custno': str, 'fundcode': str})
    df_next_month.rename(columns={'counts': '下个月浏览的次数', 'amount': '下个月购买的金额'}, inplace=True)
    df_fund_score = pd.read_csv(fund_score_stock, dtype={'custno': str, 'fundcode': str})
    df_this_three_month.fillna(0, inplace=True)

    # 用户基础数据
    df = pd.merge(df_train, df_this_three_month, on='custno', how='outer')

    # 用户对基金的评分数据
    # df_stock = pd.read_csv(fund_score_stock,dtype={'custno':str,'fundcode':str})
    # df_stock.rename(columns={'score_com':'股票型用户评分'},inplace=True)
    # df_no_stock = pd.read_csv(path_fund_score_no_stock,dtype={'custno':str,'fundcode':str})
    # df_no_stock.rename(columns={'score_com':'非股票型用户评分'},inplace=True)

    # 合并两者
    # 用户基础数据和交易浏览数据和用户对基金的评分合并
    df = pd.merge(df, df_fund_score, on=['custno', 'fundcode'], how='left')

    # 删除脏数据
    df_next_month = df_next_month[df_next_month['下个月浏览的次数'] > 0]
    # 目标函数合并
    df = pd.merge(df, df_next_month, on=['custno', 'fundcode'], how='outer')
    # df.to_excel('./data/目标函数/test.xlsx',index=False)

    # #预测用户是否购买
    # #更换用户的购买次数为[0,1]
    df['下个月购买的金额'] = [1 if x > 0 else 0 for x in df['下个月购买的金额']]

    ycol_buy = '下个月购买的金额'
    ycol_browse = '下个月浏览的次数'
    idx = 'custno'
    idfund = 'fundcode'
    feature_name = list(filter(lambda x: x not in [idx, ycol_buy, ycol_browse, idfund], df.columns))

    # 输入模型中预测
    # xgboost
    #    train,feature_names,nfold,seed,idx,ycol,month_str,pre_class
    # train, feature_names, nfold, seed, params, fit_params, idx, ycol, month_str, pre_class
    df.fillna(0,inplace=True)
    predictions = lgb_cv(df, feature_name, 5, 2019,params_lgb,fit_params_lgb, idx, ycol_buy, month_str, pre_class)
    predictions.to_csv('./data/result/lightlgb_' + month_str + pre_class + '.csv', index=False)

    # lightlgb 待完善
    # predictions = lgb_cv(df, feature_name,  5, 2019, params_lgb, fit_params_lgb, idx, ycol_buy,month_str,pre_class)
    # predictions.to_csv('./data/result/lightlgb_'+month_str+pre_class+'.csv',index=False)


def corr_cust_fund(df_train, df_goal):
    df_browse_counts = df_goal.groupby(by=['fundcode'], as_index=False)['浏览的次数', '购买的次数'].sum()
    df_browse_grow = df_goal[['fundcode', '单位净值', '本月涨幅', '今年以来涨幅0331']]
    df_browse_grow.drop_duplicates(inplace=True)
    df_funds = pd.merge(df_browse_counts, df_browse_grow, how='outer', on='fundcode')
    df_funds = df_funds.drop(columns=['fundcode'])
    df_funds = df_funds.corr('pearson')
    df_funds.to_csv(path + 'corr/基金产品相关.csv')


def cust_score():
    df_stock = pd.read_csv(path_fund_score_stock, dtype={'custno': str, 'fundcode': str})
    df_no_stock = pd.read_csv(path_fund_score_no_stock, dtype={'custno': str, 'fundcode': str})


# 评价指标
def compute_loss(target, predict):
    temp = np.log(abs(target + 1)) - np.log(abs(predict + 1))
    res = np.sqrt(np.dot(temp, temp) / len(temp))
    return res


#待完善
# lgb
def lgb_cv(train,feature_names,nfold,seed,params,fit_params,idx,ycol,month_str,pre_class):
    train_pred = pd.DataFrame({
        'true': train[ycol],
        'pred': np.zeros(len(train))})
    kfolder = KFold(n_splits=nfold, shuffle=True, random_state=seed)


    for fold_id, (trn_idx, val_idx) in enumerate(kfolder.split(train)):
        print(f'\nFold_{fold_id} Training ================================\n')
        lgb_trn = lgb.Dataset(
            data=train.iloc[trn_idx][feature_names],
            label=train.iloc[trn_idx][ycol],
            feature_name=feature_names)
        lgb_val = lgb.Dataset(
            data=train.iloc[val_idx][feature_names],
            label=train.iloc[val_idx][ycol],
            feature_name=feature_names)

        # lgb_model = lgb.LGBMClassifier()
        # lgb_model = lgb.LGBMModel(objective='binary')
        # lgb_model = lgb.LGBMModel()
        lgb_model = lgb.LGBMClassifier()
        # lgb_model.fit(train_set=lgb_trn)
        lgb_model.fit(X=train.iloc[trn_idx][feature_names],y=train.iloc[trn_idx][ycol])
        # lgb_model.fit(X=train.iloc[trn_idx][feature_names],y=train.iloc[trn_idx][ycol],eval_metric='binary_logloss')
                      # ,early_stopping_rounds=10)
        # model2 = lgb.train(params, d_train, categorical_feature=cate_features_name)

        # lgb_model.fit(X=train.iloc[trn_idx][feature_names],y=train.iloc[trn_idx][ycol],eval_metric='binary_logloss'
        #               ,early_stopping_rounds=10)
        # lgb_class = lgb.fit(params=params, train_set=lgb_trn, **fit_params,
        #                     valid_sets=[lgb_trn, lgb_val])

        val_pred = lgb_model.predict(
            train.iloc[val_idx][feature_names])
            # num_iteration=lgb_model.best_iteration)

        #处理预测结果
        #阈值设为0.5
        # threshold = 0.5
        # val_pred = [1 if pred > threshold else 0  for pred in val_pred ]


        print('AUC: %.4f' % metrics.roc_auc_score(train_pred['true'], train_pred['pred']))
        print('ACC: %.4f' % metrics.accuracy_score(train_pred['true'], train_pred['pred']))
        print('Recall: %.4f' % metrics.recall_score(train_pred['true'], train_pred['pred']))
        print('F1-score: %.4f' % metrics.f1_score(train_pred['true'], train_pred['pred']))
        print('Precesion: %.4f' % metrics.precision_score(train_pred['true'], train_pred['pred']))
        print(metrics.confusion_matrix(train_pred['true'], train_pred['pred']))


        # 输出特征重要性
        # importance = lgb_model.feature_importance()
        # names = lgb_model.feature_name()
        # with open('./data/result/lgb_feature_importance'+month_str+'.txt', 'w+') as f:
        #     for index, im in enumerate(importance):
        #         string = names[index] + ', ' + str(im) + '\n'
        #         f.write(string)

        # 输出特征重要性
        # feature_score = lgb_model.feature_importances_
        # print(feature_score)
        # feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
        # fs = []
        # for (key, value) in feature_score:
        #     fs.append("{0},{1}\n".format(key, value))
        # # with open('./data/result/xgb_feature_importance.txt', 'a') as f:
        # with open('./data/result/lgb_feature_importance_'+month_str+pre_class+'.txt', 'w+') as f:
        #     f.writelines("feature,fscore\n")
        #     f.writelines(fs)

        train_pred.loc[val_idx, 'pred'] = val_pred
    score = compute_loss(train_pred['true'], train_pred['pred'])
    print('\nCV LOSS:', score)
    return train_pred




# ====== lgb ======
params_lgb = {
    # 'num_leaves': 250,
    #           'max_depth': 5,
    #           'learning_rate': 0.02,
    #           'objective': 'regression',
    #           'boosting': 'gbdt',
    #           'verbosity': -1
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'num_leaves': 5,
    'max_depth': 6,
    'min_data_in_leaf': 450,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'lambda_l1': 1,
    'lambda_l2': 0.001,  # 越小l2正则程度越高
    'min_gain_to_split': 0.2,
    'verbose': 5,
    'is_unbalance': True
}

fit_params_lgb = {'num_boost_round': 5000,
                  'verbose_eval': 200,
                  'early_stopping_rounds': 200}




def compute_auc():
    df3 = pd.read_csv('./data/result/lightlgb_3buy.csv')
    df4 = pd.read_csv('./data/result/lightlgb_4buy.csv')
    df5 = pd.read_csv('./data/result/lightlgb_5buy.csv')

    print(roc_auc_score(df3['true'], df3['pred']))
    print(roc_auc_score(df4['true'], df4['pred']))
    print(roc_auc_score(df5['true'], df5['pred']))


# 执行主进程
if __name__ == '__main__':
    start = time.time()

    #    train,this_month,next_month,fund_score_stock,month_str,pre_class

    baseline_ctr(path_train_03, path_this_month_0331, path_goal_0331_next_month, path_fund_score_03, '3',
                 pre_class='buy')
    # baseline_ctr(path_train_04, path_this_month_0428, path_goal_0428_next_month, path_fund_score_04, '4',
    #              pre_class='buy')
    # baseline_ctr(path_train_05, path_this_month_0602, path_goal_0602_next_month, path_fund_score_05, '5',
    #              pre_class='buy')

    compute_auc()
    print("done.", time.time() - start)


