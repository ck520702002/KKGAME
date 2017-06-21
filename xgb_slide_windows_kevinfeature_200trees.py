import pandas as pd
import numpy as np
import xgboost as xgb

test_X = pd.read_csv('./assets/events_test.csv')
test_user_id = test_X.user_id.sort_values().unique()
del test_X

def save_submission(name, test_user_id, predictions):
    submission = pd.DataFrame({'user_id': test_user_id, 'title_id': predictions})
    submission = submission[['user_id', 'title_id']]
    submission.user_id = submission.user_id.map('{:08}'.format)
    submission.title_id = submission.title_id.map('{:08}'.format)
    submission.to_csv('./submission/{}.csv'.format(name), index=False)
    return submission

labels_train = pd.read_csv('./assets/labels_train.csv')
train_Y = labels_train.title_id.values

train_X_slideWindows_all = pd.read_csv('./assets/train_X_slideWindows_all.csv')
test_X_slideWindows_all = pd.read_csv('./assets/test_X_slideWindows_all.csv')

train_X_with_mostAndLastest = pd.read_csv('./assets/train_X_with_mostAndLastest.csv').drop('user_id', axis=1)
test_X_with_mostAndLastest = pd.read_csv('./assets/test_X_with_mostAndLastest.csv').drop('user_id', axis=1)
remain_features = ['w14_lastest_view_no1', 'w14_most_view_no1', 'w30_lastest_view_no1', 'w30_most_view_no1', 'w60_lastest_view_no1', 'w60_lastest_view_no2', 'w60_most_view_no1', 'w60_most_view_no2', 'w90_lastest_view_no1', 'w90_lastest_view_no2', 'w90_most_view_no1', 'w90_most_view_no2']
# train_X_slideWindows_all = train_X_slideWindows_all[remain_features].fillna(method='backfill', axis=1).fillna(method='ffill', axis=1).fillna(-1)
# test_X_slideWindows_all = test_X_slideWindows_all[remain_features].fillna(method='backfill', axis=1).fillna(method='ffill', axis=1).fillna(-1)
train_X_slideWindows_all = train_X_slideWindows_all[remain_features]
test_X_slideWindows_all = test_X_slideWindows_all[remain_features]

train_X_slideWindows_all = pd.concat([train_X_slideWindows_all, train_X_with_mostAndLastest], axis=1).fillna(-1)
test_X_slideWindows_all = pd.concat([test_X_slideWindows_all, test_X_with_mostAndLastest], axis=1).fillna(-1)
print(len(train_X_slideWindows_all.columns), len(test_X_slideWindows_all.columns))
print(len(test_X_slideWindows_all), len(test_user_id))

del train_X_with_mostAndLastest
del test_X_with_mostAndLastest


model = xgb.XGBClassifier(objective='multi:softprob', n_estimators=200, n_jobs=-1, learning_rate=0.1)
model.fit(train_X_slideWindows_all, train_Y)

import pickle
pickle.dump(model, open('slide_windows_all_7_14_30_60_90_with_most_and_latest_xgb_200.pkl', 'wb'))

pred = model.predict(test_X_slideWindows_all)
save_submission('slide_windows_all_7_14_30_60_90_with_most_and_latest_xgb_200', test_user_id, pred).head()
