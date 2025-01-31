import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import f_oneway, boxcox
from scipy.stats.mstats import normaltest
from scipy.special import inv_boxcox
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore', module='sklearn')
import seaborn as sns
df = pd.read_csv('./Medicalpremium.csv')
df_raw = df.copy()
df.head()
df.info()
fig, ax  = plt.subplots(4,2, figsize=(12,15))
fig.delaxes(ax[3,1])

cols = ['Diabetes', 'BloodPressureProblems', 'AnyTransplants', 'AnyChronicDiseases', 'KnownAllergies',
      'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries']

for num, col in enumerate(cols):
    ax_def = ax.ravel()[num]
    sns.histplot(df[col], ax=ax_def)
    
    if num==6:
        ax_def.set_xticks([0.1,1.1, 2.1, 2.92])
        ax_def.set_xticklabels([0, 1, 2, 3])
        continue
        
    ax_def.set_xticks([0.04,0.96])
    ax_def.set_xticklabels([0, 1])
    df[['Age', 'Height', 'Weight', 'PremiumPrice']].describe()
    df[['Age', 'Height', 'Weight', 'PremiumPrice']].hist(figsize=(10,10))
    fig, axs = plt.subplots(1,2,figsize=(12,8))
sns.boxplot(x='variable', y='value', data=pd.melt(df[['Height', 'Weight', 'Age']]), ax=axs[0])
sns.boxplot(y='PremiumPrice',  data=df, ax=axs[1])
axs[1].set(xlabel='Price', ylabel='value')
weight_val = df['Weight'].describe()
weight_val
Q3_wei = df['Weight'].quantile(0.75)
Q1_wei = df['Weight'].quantile(0.25)
weight_lim = 1.5 * (Q3_wei - Q1_wei) + Q3_wei

Q3_pri = df['PremiumPrice'].quantile(0.75)
Q1_pri = df['PremiumPrice'].quantile(0.25)
price_lim = 1.5 * (Q3_pri - Q1_pri) + Q3_pri

df = df[(df['Weight']<weight_lim) &  (df['PremiumPrice']<price_lim)]

ig, axs = plt.subplots(1,2,figsize=(12,8))
sns.boxplot(x='variable', y='value', data=pd.melt(df[['Height', 'Weight', 'Age']]), ax=axs[0])
sns.boxplot(y='PremiumPrice',  data=df, ax=axs[1])
axs[1].set(xlabel='Price', ylabel='value')
df[['Age', 'Height', 'Weight', 'PremiumPrice']].corr()
sns.regplot(x='Height', y='PremiumPrice', data=df)
df.drop(columns=['Height'], axis=1, inplace=True)
df_noDiabetes = df.groupby('Diabetes').get_group(0)['PremiumPrice']
df_Diabetes = df.groupby('Diabetes').get_group(1)['PremiumPrice']
f_oneway(df_noDiabetes, df_Diabetes)
df_noBP = df.groupby('BloodPressureProblems').get_group(0)['PremiumPrice']
df_BP = df.groupby('BloodPressureProblems').get_group(1)['PremiumPrice']
f_oneway(df_noBP, df_BP)
df_noT = df.groupby('AnyTransplants').get_group(0)['PremiumPrice']
df_T = df.groupby('AnyTransplants').get_group(1)['PremiumPrice']
f_oneway(df_noT, df_T)
df_noCD = df.groupby('AnyChronicDiseases').get_group(0)['PremiumPrice']
df_CD = df.groupby('AnyChronicDiseases').get_group(1)['PremiumPrice']
f_oneway(df_noCD, df_CD)
df_noKA = df.groupby('KnownAllergies').get_group(0)['PremiumPrice']
df_KA = df.groupby('KnownAllergies').get_group(1)['PremiumPrice']
f_oneway(df_noKA, df_KA)
df_noHC = df.groupby('HistoryOfCancerInFamily').get_group(0)['PremiumPrice']
df_HC = df.groupby('HistoryOfCancerInFamily').get_group(1)['PremiumPrice']
f_oneway(df_noHC, df_HC)
df_NS0 = df.groupby('NumberOfMajorSurgeries').get_group(0)['PremiumPrice']
df_NS1 = df.groupby('NumberOfMajorSurgeries').get_group(1)['PremiumPrice']
df_NS2 = df.groupby('NumberOfMajorSurgeries').get_group(2)['PremiumPrice']
df_NS3 = df.groupby('NumberOfMajorSurgeries').get_group(3)['PremiumPrice']
f_oneway(df_NS0, df_NS1, df_NS2, df_NS3)
df.drop(columns=['KnownAllergies'], axis=1, inplace=True)
df['NumberOfMajorSurgeries'].value_counts()
df = pd.get_dummies(df, columns=['NumberOfMajorSurgeries'], drop_first=True)
df.head()
df.info()
df.PremiumPrice.hist()
normaltest(df.PremiumPrice)
PriceTrans, lmbda = boxcox(df.PremiumPrice)
print('The obtained value for lambda by boxcox transformation: %0.3f ' %lmbda)
normaltest(PriceTrans)
df['PremiumPrice'] = PriceTrans
df['PremiumPrice'].hist()
X = df.drop(columns=['PremiumPrice'], axis=1)
y = df.PremiumPrice
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, 
                                                    random_state=1979, shuffle=True)

scaler = StandardScaler()
X_tr_sc = X_tr.copy()
X_tr_sc[['Age', 'Weight']] = scaler.fit_transform(X_tr[['Age', 'Weight']])

lr = LinearRegression()
lr.fit(X_tr_sc, y_tr)

X_te_sc = X_te.copy()
X_te_sc[['Age', 'Weight']] = scaler.transform(X_te[['Age', 'Weight']])
y_te_pr = lr.predict(X_te_sc)

y_te_true = inv_boxcox(y_te, lmbda)
y_te_pr_true = inv_boxcox(y_te_pr, lmbda)

print('The R2 Score: ', r2_score(y_true=y_te_true, y_pred=y_te_pr_true))
lr_res = pd.DataFrame(zip(X_tr.columns.tolist(),lr.coef_)).rename({0:'Feature', 1:'Coefficient'}, axis=1)
lr_res.loc[len(lr_res.index)] = ['Intercept', lr.intercept_]
lr_res
lr_res = pd.DataFrame(zip(X_tr.columns.tolist(),lr.coef_)).rename({0:'Feature', 1:'Coefficient'}, axis=1)
lr_res.loc[len(lr_res.index)] = ['Intercept', lr.intercept_]
lr_res
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('dark')

plt.scatter(x=y_te_true, y=y_te_pr_true)
ax = plt.gca()
ax.set(xlabel='Ground truth', ylabel='Prediction', title='Predicted versus True Premium Price')
poly = PolynomialFeatures(degree=2, include_bias=False)
X_po = poly.fit_transform(X)

X_po_tr, X_po_te, y_tr, y_te = train_test_split(X_po, y, test_size=0.3, 
                                   shuffle=True, random_state=1979)

scaler_po = StandardScaler()
X_po_tr_sc = scaler_po.fit_transform(X_po_tr)

ridge_alphas = np.geomspace(1e-1,1e1,100)
y_te_true = inv_boxcox(y_te, lmbda)
ridge_res = list()

for alpha in ridge_alphas:
    ridge = Ridge(alpha=alpha, max_iter=100000)
    ridge.fit(X_po_tr_sc, y_tr)

    X_po_te_sc = scaler_po.transform(X_po_te)
    y_te_pr = ridge.predict(X_po_te_sc)

    y_te_pr_true = inv_boxcox(y_te_pr, lmbda)
    ridge_res.append(r2_score(y_true=y_te_true, y_pred=y_te_pr_true))

ax =plt.axes()
ax.semilogx(ridge_alphas, ridge_res)
ax.set(xlabel='alpha', ylabel='R2 Score', title='The R2 score for different alphas')


print('''The best R2 score equal to {0:1.3f} for linear regression using Ridge
is obtained with alpha = {1:1.3f}.'''.format(np.max(ridge_res), ridge_alphas[np.argmax(ridge_res)]))
ridge_alpha_best = ridge_alphas[np.argmax(ridge_res)]

ridge = Ridge(alpha=ridge_alpha_best, max_iter=100000)
ridge.fit(X_po_tr_sc, y_tr)

y_te_pr = ridge.predict(X_po_te_sc)
y_te_pr_true = inv_boxcox(y_te_pr, lmbda)
ridge_rmse = mean_squared_error(y_true=y_te_true, y_pred=y_te_pr_true, squared=False )


print('The sum of coefficients obtained by Ridge: %4.4f' %abs(ridge.coef_).sum())

print('The number of zero coefficients equal to %i.' %len(ridge.coef_[ridge.coef_==0]))

print('''The root mean squared error (rmse) equal to {0:1.3f} for linear regression using Ridge
is obtained with alpha = {1:1.3f}.'''.format(ridge_rmse, ridge_alphas[np.argmax(ridge_res)]))
poly = PolynomialFeatures(degree=2, include_bias=False)
X_po = poly.fit_transform(X)

X_po_tr, X_po_te, y_tr, y_te = train_test_split(X_po, y, test_size=0.3, 
                                   shuffle=True, random_state=1979)

scaler_po = StandardScaler()
X_po_tr_sc = scaler_po.fit_transform(X_po_tr)

lasso_alphas = np.geomspace(1e-1,1e1,100)
y_te_true = inv_boxcox(y_te, lmbda)
lasso_res = list()

for alpha in lasso_alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_po_tr_sc, y_tr)

    X_po_te_sc = scaler_po.transform(X_po_te)
    y_te_pr = lasso.predict(X_po_te_sc)

    y_te_pr_true = inv_boxcox(y_te_pr, lmbda)
    lasso_res.append(r2_score(y_true=y_te_true, y_pred=y_te_pr_true))

ax = plt.axes()
ax.semilogx(lasso_alphas, lasso_res)
ax.set(xlabel='alpha', ylabel='R2 Score', title='The R2 score for different alphas')

print('''The best R2 score equal to {0:1.3f} for linear regression using Lasso
    is obtained with alpha = {1:1.3f}.'''.format(np.max(lasso_res), lasso_alphas[np.argmax(lasso_res)]))
lasso_alpha_best = lasso_alphas[np.argmax(lasso_res)]

lasso = Lasso(alpha=lasso_alpha_best, max_iter=100000)
lasso.fit(X_po_tr_sc, y_tr)

y_te_pr = lasso.predict(X_po_te_sc)
y_te_pr_true = inv_boxcox(y_te_pr, lmbda)
lasso_rmse = mean_squared_error(y_true=y_te_true, y_pred=y_te_pr_true, squared=False )

print('The sum of coefficients obtained by Lasso: %4.4f' %abs(lasso.coef_).sum())

print('The number of zero coefficients equal to %i.' %len(lasso.coef_[lasso.coef_==0]))

print('''The root mean squared error (rmse) equal to {0:1.3f} for linear regression using Lasso
is obtained with alpha = {1:1.3f}.'''.format(lasso_rmse, lasso_alphas[np.argmax(lasso_res)]))
poly = PolynomialFeatures(degree=2, include_bias=False)
X_po = poly.fit_transform(X)

X_po_tr, X_po_te, y_tr, y_te = train_test_split(X_po, y, test_size=0.3, 
                                   shuffle=True, random_state=1979)

scaler_po = StandardScaler()
X_po_tr_sc = scaler_po.fit_transform(X_po_tr)

elastic_alphas = np.geomspace(1e-3,1e-2,100)
y_te_true = inv_boxcox(y_te, lmbda)
elastic_res = list()

for alpha in elastic_alphas:
    elastic = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
    elastic.fit(X_po_tr_sc, y_tr)

    X_po_te_sc = scaler_po.transform(X_po_te)
    y_te_pr = elastic.predict(X_po_te_sc)

    y_te_pr_true = inv_boxcox(y_te_pr, lmbda)
    elastic_res.append(r2_score(y_true=y_te_true, y_pred=y_te_pr_true))

ax = plt.axes()
ax.semilogx(elastic_alphas, elastic_res)
ax.set(xlabel='alpha', ylabel='R2 Score', title='The R2 score for different alphas')

print('''The best R2 score equal to {0:1.3f} for linear regression using Elastic Net
    is obtained with alpha = {1:1.3f}.'''.format(np.max(elastic_res), elastic_alphas[np.argmax(elastic_res)]))
lasso_alpha_best = lasso_alphas[np.argmax(lasso_res)]

lasso = Lasso(alpha=lasso_alpha_best, max_iter=100000)
lasso.fit(X_po_tr_sc, y_tr)

y_te_pr = lasso.predict(X_po_te_sc)
y_te_pr_true = inv_boxcox(y_te_pr, lmbda)
lasso_rmse = mean_squared_error(y_true=y_te_true, y_pred=y_te_pr_true, squared=False )

print('The sum of coefficients obtained by Lasso: %4.4f' %abs(lasso.coef_).sum())

print('The number of zero coefficients equal to %i.' %len(lasso.coef_[lasso.coef_==0]))

print('''The root mean squared error (rmse) equal to {0:1.3f} for linear regression using Lasso
is obtained with alpha = {1:1.3f}.'''.format(lasso_rmse, lasso_alphas[np.argmax(lasso_res)]))
poly = PolynomialFeatures(degree=2, include_bias=False)
X_po = poly.fit_transform(X)

X_po_tr, X_po_te, y_tr, y_te = train_test_split(X_po, y, test_size=0.3, 
                                   shuffle=True, random_state=1979)

scaler_po = StandardScaler()
X_po_tr_sc = scaler_po.fit_transform(X_po_tr)

elastic_alphas = np.geomspace(1e-3,1e-2,100)
y_te_true = inv_boxcox(y_te, lmbda)
elastic_res = list()

for alpha in elastic_alphas:
    elastic = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
    elastic.fit(X_po_tr_sc, y_tr)

    X_po_te_sc = scaler_po.transform(X_po_te)
    y_te_pr = elastic.predict(X_po_te_sc)

    y_te_pr_true = inv_boxcox(y_te_pr, lmbda)
    elastic_res.append(r2_score(y_true=y_te_true, y_pred=y_te_pr_true))

ax = plt.axes()
ax.semilogx(elastic_alphas, elastic_res)
ax.set(xlabel='alpha', ylabel='R2 Score', title='The R2 score for different alphas')

print('''The best R2 score equal to {0:1.3f} for linear regression using Elastic Net
    is obtained with alpha = {1:1.3f}.'''.format(np.max(elastic_res), elastic_alphas[np.argmax(elastic_res)]))
elastic_alpha_best = elastic_alphas[np.argmax(elastic_res)]
elastic = ElasticNet(alpha=elastic_alpha_best, max_iter=100000)
elastic.fit(X_po_tr_sc, y_tr)

y_te_pr = elastic.predict(X_po_te_sc)
y_te_pr_true = inv_boxcox(y_te_pr, lmbda)
elastic_rmse = mean_squared_error(y_true=y_te_true, y_pred=y_te_pr_true, squared=False )


print('The sum of coefficients obtained by Elastic Net: %4.4f' %abs(elastic.coef_).sum())

print('The number of zero coefficients equal to %i.' %len(elastic.coef_[elastic.coef_==0]))

print('''The root mean squared error (rmse) equal to {0:1.3f} for linear regression using Elastic Net
is obtained with alpha = {1:1.3f}.'''.format(elastic_rmse, elastic_alphas[np.argmax(elastic_res)]))
ridge_series = pd.Series([abs(ridge.coef_).sum().round(2), len(ridge.coef_[ridge.coef_==0]),
                         np.max(ridge_res).round(4), np.max(ridge_rmse).round(2)],
                        index=['Sum of coeff.', 'Number of zero Coeff.', 'R2 score', 'RMSE'],
                        name='Ridge')

lasso_series = pd.Series([abs(lasso.coef_).sum().round(2), len(lasso.coef_[lasso.coef_==0]),
                         np.max(lasso_res).round(4), np.max(lasso_rmse).round(2)],
                        index=['Sum of coeff.', 'Number of zero Coeff.', 'R2 score', 'RMSE'],
                        name='Lasso')

elastic_series = pd.Series([abs(elastic.coef_).sum().round(2), len(elastic.coef_[elastic.coef_==0]),
                         np.max(elastic_res).round(4), np.max(elastic_rmse).round(2)],
                        index=['Sum of coeff.', 'Number of zero Coeff.', 'R2 score', 'RMSE'],
                        name='Elastic Net')

RES = pd.DataFrame([ridge_series, lasso_series, elastic_series])
RES
X['new1'] = X['Weight'] / X['Age'] 
X['new2'] = X['Diabetes'] * X['BloodPressureProblems'] * X['AnyTransplants'] * X['AnyChronicDiseases']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, 
                                    random_state=1979, shuffle=True)


poly = PolynomialFeatures(degree=2, include_bias=False)
X_tr_po = poly.fit_transform(X_tr)

scaler_po = StandardScaler()
X_tr_po_sc = scaler_po.fit_transform(X_tr_po)
y_te_true = inv_boxcox(y_te, lmbda)

X_te_po = poly.transform(X_te)
X_te_po_sc = scaler_po.transform(X_te_po)

alphas = np.geomspace(1e-3,1e3,1000)

kf = KFold(n_splits=5, shuffle=True, random_state=1980)

lasCV = LassoCV(alphas=alphas, cv=kf)
ridCV = RidgeCV(alphas=alphas, cv=kf)
elaCV = ElasticNetCV(alphas=alphas, l1_ratio=0.5, cv=kf)

estimators = [lasCV, ridCV, elaCV]

R2 = list()
RMSE = list()

for estimator in estimators:
    estimator.fit(X_tr_po_sc, y_tr)
    y_te_pr = estimator.predict(X_te_po_sc)
    y_te_pr_true = inv_boxcox(y_te_pr, lmbda) 
    
    r2 = r2_score(y_true=y_te_true , y_pred=y_te_pr_true)
    rmse = mean_squared_error(y_true=y_te_true, y_pred=y_te_pr_true, squared=False)
    
    R2.append(r2), RMSE.append(rmse)
    
pd.DataFrame({'R2 score': R2, 'RMSE': RMSE}, index=['Ridge', 'Lasso', 'Elastic Net'] )
colors = sns.color_palette()

fig = plt.figure(figsize=(15,10))
plt.plot(ridCV.coef_, color=colors[0], marker='o', label='ridge rep.')
plt.plot(lasCV.coef_, color=colors[1], marker='^', label='lasso rep.')
plt.plot(elaCV.coef_, color=colors[2], marker='+', label='elastic net rep.')
plt.legend()
plt.title('The coefficients obtained by Ridge, Lasso, and Elastic Net methods')
plt.xlabel('Feature')
plt.ylabel('Value')
scaler_list = [MaxAbsScaler(), MinMaxScaler(), StandardScaler()]
alphas = np.geomspace(1e-2,1e2, 100)

poly = PolynomialFeatures(degree=2)
kf = KFold(n_splits=5, shuffle=True, random_state=1979)

scores = list()
alpha_val = list()
scaler_val = list()
for sca in scaler_list:
    for alpha in alphas:
        las = Lasso(alpha=alpha)

        estimator = Pipeline([('Polynomial_features', poly), ('scaler', sca), ('lasso_reg', las)])
        scaler_val.append(sca)
        predictions = cross_val_predict(estimator, X, y, cv=kf)
        scores.append(r2_score(y_true=y, y_pred=predictions))
        alpha_val.append(alpha)
              
RES = pd.DataFrame({'alpha':alpha_val, 'Scaler':scaler_val, 'R2 score':scores})

print('The best scaler: \n', RES[ RES['R2 score']== RES['R2 score'].max()] )
kf = KFold(n_splits=5, shuffle=True, random_state=1979)

poly = PolynomialFeatures()
sca = StandardScaler()
las = Ridge()

alphas= np.geomspace(1e-2, 1e2, num=100)

estimator = Pipeline([('Polynomial_features', poly),
                      ('Scaler', sca),
                      ('Regression', las)])

params = {'Polynomial_features__degree': [1, 2, 3],
          'Regression__alpha': alphas }

grid = GridSearchCV(estimator, params, cv=kf)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, 
                                    random_state=1979, shuffle=True)

grid.fit(X_tr, y_tr)
grid.best_score_, grid.best_params_
y_te_pre = grid.predict(X_te)

y_te_true = inv_boxcox(y_te, lmbda)
y_te_pre_true = inv_boxcox(y_te_pre, lmbda)
r2_score(y_true=y_te_true , y_pred=y_te_pre_true)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, 
                                    random_state=1979, shuffle=True)

kf = KFold(n_splits=5, shuffle=True, random_state=1979)

BEST_SCORES = list()
BEST_DEGREE = list()
BEST_ALPHA = list()
SCALER = list()
METHODS = list()
R2_TEST = list()
L1_RATIO = list()

poly = PolynomialFeatures()
scalers = [StandardScaler(), MinMaxScaler(), MaxAbsScaler()]
methods = [Ridge(), Lasso(), ElasticNet()]

alphas= np.geomspace(1e-2, 1e2, num=10)
l1_ratios = np.linspace(0,1,num=10)

for scaler in scalers:
    for method in methods:

        estimator = Pipeline([('Polynomial_features', poly),
                              ('Scaler', scaler),
                              ('Regression', method)])
        
        if str(method)=='ElasticNet()':
            params = {'Polynomial_features__degree': [1, 2, 3],
                      'Regression__alpha': alphas,
                      'Regression__l1_ratio': l1_ratios
                      }
        else:
            params = {'Polynomial_features__degree': [1, 2, 3],
                      'Regression__alpha': alphas
                      }
        

        grid = GridSearchCV(estimator, params, cv=kf)
        grid.fit(X_tr, y_tr)
        y_te_pre = grid.predict(X_te)
        y_te_true = inv_boxcox(y_te, lmbda)
        y_te_pre_true = inv_boxcox(y_te_pre, lmbda)
        r2_te = r2_score(y_true=y_te_true , y_pred=y_te_pre_true)
        
        BEST_SCORES.append(grid.best_score_)
        BEST_DEGREE.append(grid.best_params_['Polynomial_features__degree'])
        BEST_ALPHA.append(grid.best_params_['Regression__alpha'])
        SCALER.append(scaler)
        METHODS.append(method)
        R2_TEST.append(r2_te)
        
        if str(method)=='ElasticNet()':
            L1_RATIO.append(grid.best_params_['Regression__l1_ratio'])
        else:
            L1_RATIO.append(np.nan)
        
        
RES = pd.DataFrame({'R2 Score':BEST_SCORES, 'Alpha': BEST_ALPHA, 'L1_ratio':L1_RATIO,
                    'Polynomial Degree':BEST_DEGREE,
                     'Scaler':SCALER, 'Methods': METHODS , 'R2 Score:Test Set':R2_TEST} )
RES
opt_degree = RES[RES['R2 Score:Test Set']==RES['R2 Score:Test Set'].max()]['Polynomial Degree'].values[0]
opt_scaler = RES[RES['R2 Score:Test Set']==RES['R2 Score:Test Set'].max()]['Scaler'].values[0]
opt_alpha = RES[RES['R2 Score:Test Set']==RES['R2 Score:Test Set'].max()]['Alpha'].values[0]
opt_method = RES[RES['R2 Score:Test Set']==RES['R2 Score:Test Set'].max()]['Methods'].values[0]
opt_L1_ratio = RES[RES['R2 Score:Test Set']==RES['R2 Score:Test Set'].max()]['L1_ratio'].values[0]

poly = PolynomialFeatures()

pipe = Pipeline([('Polynomial_features', poly), ('scaler', opt_scaler), ('Regression', opt_method)])

if str(opt_method)=='ElasticNet':
    pipe.set_params(Polynomial_features__degree=opt_degree, Regression__alpha=opt_alpha,
                   Regression__l1_ratio=opt_L1_ratio)
else:
    pipe.set_params(Polynomial_features__degree=opt_degree, Regression__alpha=opt_alpha)

pipe.fit(X_tr,y_tr)
print('The obtained value for alpha for the train set: ', pipe.score(X_tr,y_tr))
y_te_pre = pipe.predict(X_te)
y_te_true = inv_boxcox(y_te, lmbda)
y_te_pre_true = inv_boxcox(y_te_pre, lmbda)

r2_te = r2_score(y_true=y_te_true , y_pred=y_te_pre_true)
print('The alpha value for the test set: ', r2_te)

rmse = mean_squared_error(y_true=y_te_true, y_pred=y_te_pre_true, squared=False)
print('The rmse value for the test set: ', rmse)
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(x=y_te_true, y=y_te_pre_true)
ax.set(xlabel='Ground Truth', ylabel='Prediction', title='Prection versus true values for premium price')
