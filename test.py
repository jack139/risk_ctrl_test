from first_load import *
from tqdm import tqdm


data_path = './data'
file_name = 'test20k.csv'

df_1 = pd.read_csv(os.path.join(data_path, file_name), header=0, sep=',', low_memory=False)
raw_columns_list = list(df_1.columns)

print('原始数据样本量:',df_1.shape)
print('='*80)
print('查看原始数据前 5 行：')
df_1.head()


### 数据清洗 ########################################################

## 好坏样本定义，做贷后标签状态映射
list(df_1["loan_status"].unique()) # 查看标签变量下的状态取值名
print('查看不同 loan_status （贷后状态），样本的分布情况:')
df_1.groupby(["loan_status"])[['int_rate']].count()


df_1.rename(columns={'loan_status':'target'},inplace = True) # 将 loan_status 名称改为 target（目标）
df_1 = df_1.loc[~(df_1.target.isnull()),] # 取出目标变量非空的样本
df_1["target"] = df_1["target"].map(target_mapping(df_1["target"].unique())) # 调用之前定义的子函数，进行好坏样本映射
df_1.target.unique() # 查看映射的取值种类（0和1）
df_1 = df_1.loc[df_1.target<=1,] # 双重确定，取出的样本为目标 target = 0/1 的样本
## 样本不均衡严重
print('好样本 / 坏样本，比值：')
print(sum(df_1.target==0)/df_1.target.sum())


## 1.删除贷后数据（非 target 的其余贷后变量）
var_del = [ 'collection_recovery_fee','initial_list_status','last_credit_pull_d','last_pymnt_amnt',
           'last_pymnt_d','next_pymnt_d','out_prncp','out_prncp_inv','recoveries','total_pymnt',
           'total_pymnt_inv','total_rec_int','total_rec_late_fee','total_rec_prncp','settlement_percentage' ]
df_1 = df_1.drop(var_del, axis=1)


## 2 .删除LC公司信用评估的结果,利率也是LC公司的结果，且利率越高风险越大，也是数据泄露的变量
var_del_1 = ['grade','sub_grade','int_rate']
df_1 = df_1.drop(var_del_1, axis=1)
#    df_1.isnull().any()

print('删除贷后数据、评估数据后，数据样本量：')
df_1.shape

## 3.查看缺失值情况
print('查看数据缺失情况')
## 缺失值绘图
var_list = list(df_1.columns)
for i in range(1,4): # 取值 1,2,3 次循环制图，将 126个变量分层 3*40  显示缺失值
    start = (i-1)*40
    stop = i*40
    plt.figure(figsize=(15,4))
    msno.bar(df_1[var_list[start:stop]],labels=True, fontsize=17)
    plt.xticks(rotation=30,fontsize=17)


## 删除缺失值比率超过 95% 的变量
df_1,na_del = del_na(df_1,list(df_1.columns),rate=0.95) # 调用定义好的子函数
print('删除缺失值比率超过95%的变量,共',len(na_del),'条')
    
## 删除行全为缺失值的本样本
df_1.dropna(axis=0,how='all',inplace=True)
print('删除缺失值后，样本数据量为：',df_1.shape)

## 4.删除只有一种状态的变量
cols_name = list(df_1.columns) 
cols_name.remove('target')
df_1,dele_list = constant_del(df_1, cols_name, ignore=['issue_d']) # 调用定义好的子函数

## 5.删除长尾数据
cols_name_1 = list(df_1.columns)
cols_name_1.remove('target')
df_1,dele_list = tail_del(df_1,cols_name_1,rate=0.9, ignore=['issue_d']) # 调用定义好的子函数

## 6.删除一些明显无关的变量
## emp_title工作岗级，可以做一个等级划分，这里直接删除，离散程度较大删除，
## zip_code邮编信息，离散程度太大
## title与purpose一致，直接删除
len(df_1.emp_title.unique())
var_del_2 = ['emp_title','zip_code','title'] # 人为介入，专家经验
df_1 = df_1.drop(var_del_2, axis=1)


## 7.数据格式规范化
## 设置全部显示列信息
pd.set_option('display.max_columns', None)
print('数据格式规范化之前，显示前 5 行：')
df_1.head(5)

print('数据格式规范化之前，已有的数据类型：')
np.unique(df_1.dtypes)

## revol_util数据格式规约
#df_1['revol_util']=df_1['revol_util'].str.replace('%','').astype('float')

## 8.日期变量处理
#var_date = ['issue_d','earliest_cr_line','sec_app_earliest_cr_line' ]
var_date = ['issue_d','earliest_cr_line' ]
## 时间格式转化
df_1['issue_d'] = df_1['issue_d'].apply(trans_format,args=('%b-%Y','%Y-%m',))
df_1['earliest_cr_line'] = df_1['earliest_cr_line'].apply(trans_format,args=('%b-%Y','%Y-%m',))
#df_1['sec_app_earliest_cr_line'] = df_1['sec_app_earliest_cr_line'].apply(trans_format,args=('%b-%Y','%Y-%m',))

print('查看数据清理后，总数据量：',df_1.shape)
print('='*80)
print('数据格式规范化之后，查看数据前 5 行：')
df_1.head()


### 特征工程 ########################################################

## 将时间差值转为月份
df_1['mth_interval']=df_1['issue_d']-df_1['earliest_cr_line']
#df_1['sec_mth_interval']=df_1['issue_d']-df_1['sec_app_earliest_cr_line']
    
df_1['mth_interval'] = df_1['mth_interval'].apply(lambda x: round(x.days/30,0))
#df_1['sec_mth_interval'] = df_1['sec_mth_interval'].apply(lambda x: round(x.days/30,0))
df_1['issue_m']=df_1['issue_d'].apply(lambda x: x.month)

## 删除原始日期变量
df_1 = df_1.drop(var_date, axis=1)

## 年还款总额占年收入百分比
index_1 = df_1.annual_inc ==0
if sum(index_1) > 0:
    df_1.loc[index_1,'annual_inc'] = 10
df_1['pay_in_rate'] = df_1.installment*12/df_1.annual_inc
index_s1 = (df_1['pay_in_rate'] >=1) & (df_1['pay_in_rate'] <2) 
if sum(index_s1)>0:
    df_1.loc[index_s1,'pay_in_rate'] = 1
index_s2 = df_1['pay_in_rate'] >=2
if sum(index_s2)>0:
    df_1.loc[index_s2,'pay_in_rate'] = 2 
    
## 信用借款账户数与总的账户数比
df_1['credit_open_rate'] = df_1.open_acc/df_1.total_acc
## 周转余额与所有账户余额比
df_1['revol_total_rate'] = df_1.revol_bal/df_1.tot_cur_bal
## 欠款总额和本次借款比
df_1['coll_loan_rate'] = df_1.tot_coll_amt/df_1.installment
index_s3 = df_1['coll_loan_rate'] >=1
if sum(index_s3)>0:
    df_1.loc[index_s3,'coll_loan_rate'] = 1
## 银行卡状态较好的个数与总银行卡数的比
df_1['good_bankcard_rate'] = df_1.num_bc_sats/df_1.num_bc_tl
## 余额大于零的循环账户数与所有循环账户数的比
df_1['good_rev_accts_rate'] = df_1.num_rev_tl_bal_gt_0/df_1.num_rev_accts



### 变量分箱 ########################################################

## 离散变量与连续变量区分
categorical_var,numerical_var = category_continue_separation(df_1,list(df_1.columns))
print('查看变量中的数据类型，若为浮点数 or 整数，则判定为:数值变量/连续变量，否则为离散变量。')
print('='*80)
print('初步数值变量/连续变量有',len(numerical_var),'个,离散变量有',len(categorical_var),'个')
print('='*80)
print('进一步，数值变量中取值少于 10 种的，我们重新认定为离散变量，如：')
print('='*80)
for s in set(numerical_var):
    if len(df_1[s].unique())<=10:
        print('变量'+s+'可能取值'+str(len(df_1[s].unique())))
        categorical_var.append(s)
        numerical_var.remove(s)
        ## 同时将后加的数值变量转为字符串
        index_1 = df_1[s].isnull()
        if sum(index_1) > 0:
            df_1.loc[~index_1,s] = df_1.loc[~index_1,s].astype('str')
        else:
            df_1[s] = df_1[s].astype('str')


## 划分测试集与训练集
data_train, data_test = train_test_split(df_1,  test_size=0.2,stratify=df_1.target,random_state=25)
print('训练集中，好信用 / 坏信用，比值：',sum(data_train.target==0)/data_train.target.sum())
print('测试集中，好信用 / 坏信用，比值：',sum(data_test.target==0)/data_test.target.sum())


## 连续变量分箱
dict_cont_bin = {}
for i in tqdm(numerical_var):
    dict_cont_bin[i],gain_value_save , gain_rate_save = varbin_meth.cont_var_bin(data_train[i], data_train.target, method=2, mmin=4, mmax=12,
                                     bin_rate=0.01, stop_limit=0.05, bin_min_num=20)
## 离散变量分箱
dict_disc_bin = {}
del_key = []
for i in tqdm(categorical_var):
    dict_disc_bin[i],gain_value_save , gain_rate_save ,del_key_1 = varbin_meth.disc_var_bin(data_train[i], data_train.target, method=2, mmin=4,
                                     mmax=10, stop_limit=0.05, bin_min_num=20)
    if len(del_key_1)>0 :
        del_key.extend(del_key_1)
        
## 删除分箱数只有1个的变量
if len(del_key) > 0:
    for j in del_key:
        del dict_disc_bin[j]


## 训练数据分箱
## 连续变量分箱映射
df_cont_bin_train = pd.DataFrame()
for i in dict_cont_bin.keys():
    df_cont_bin_train = pd.concat([ df_cont_bin_train , varbin_meth.cont_var_bin_map(data_train[i], dict_cont_bin[i]) ], axis = 1)
## 离散变量分箱映射
#    ss = data_train[list( dict_disc_bin.keys())]
df_disc_bin_train = pd.DataFrame()
for i in dict_disc_bin.keys():
    df_disc_bin_train = pd.concat([ df_disc_bin_train , varbin_meth.disc_var_bin_map(data_train[i], dict_disc_bin[i]) ], axis = 1)

## 测试数据分箱
## 连续变量分箱映射
df_cont_bin_test = pd.DataFrame()
for i in dict_cont_bin.keys():
    df_cont_bin_test = pd.concat([ df_cont_bin_test , varbin_meth.cont_var_bin_map(data_test[i], dict_cont_bin[i]) ], axis = 1)
## 离散变量分箱映射
#    ss = data_test[list( dict_disc_bin.keys())]
df_disc_bin_test = pd.DataFrame()
for i in dict_disc_bin.keys():
    df_disc_bin_test = pd.concat([ df_disc_bin_test , varbin_meth.disc_var_bin_map(data_test[i], dict_disc_bin[i]) ], axis = 1)
    
## 组成分箱后的训练集与测试集
df_disc_bin_train['target'] = data_train.target
data_train_bin = pd.concat([df_cont_bin_train,df_disc_bin_train],axis=1)
df_disc_bin_test['target'] = data_test.target
data_test_bin = pd.concat([df_cont_bin_test,df_disc_bin_test],axis=1)

data_train_bin.reset_index(inplace=True,drop=True)
data_test_bin.reset_index(inplace=True,drop=True)

var_all_bin = list(data_train_bin.columns)
var_all_bin.remove('target')

print('训练集变量分箱结果：')
data_train_bin



## WOE编码
## 训练集WOE编码
df_train_woe, dict_woe_map, dict_iv_values ,var_woe_name = var_encode.woe_encode(data_train_bin,
                            data_path,var_all_bin, data_train_bin.target,'dict_woe_map', flag='train')
## 测试集WOE编码
df_test_woe, var_woe_name = var_encode.woe_encode(data_test_bin,
                            data_path,var_all_bin, data_test_bin.target, 'dict_woe_map',flag='test')


### 特征选择 ########################################################


## IV值初步筛选，选择iv大于等于0.01的变量
df_train_woe = iv_selection_func(df_train_woe,dict_iv_values,iv_low=0.01)
    
## 相关性分析，相关系数即皮尔逊相关系数大于0.8的，删除IV值小的那个变量。
sel_var = list(df_train_woe.columns)
sel_var.remove('target') 
## 循环，变量与多个变量相关系数大于0.8，则每次只删除IV值最小的那个，直到没有大于0.8的变量为止
while True:
    pearson_corr = (np.abs(df_train_woe[sel_var].corr()) >= 0.8)
    if pearson_corr.sum().sum() <= len(sel_var):
        break
    del_var = []
    for i in sel_var:
        var_1 = list(pearson_corr.index[pearson_corr[i]].values)
        if len(var_1)>1 :
            df_temp = pd.DataFrame({'value':var_1,'var_iv':[ dict_iv_values[x.split(sep='_woe')[0]] for x in var_1 ]})
            del_var.extend(list(df_temp.value.loc[df_temp.var_iv == df_temp.var_iv.min(),].values))
    del_var1 = list(np.unique(del_var) )      
    ## 删除这些，相关系数大于0.8的变量
    sel_var = [s for s in sel_var if s not in del_var1]
print('='*80)
print('IV值筛选后，剩余变量个数',len(sel_var),'个；较筛选前少了',95-len(sel_var),'个')


#        ## 多重共线筛选，vif 方差膨胀银子筛选
#        df_vif = pd.DataFrame({'value':sel_var,
#                               'vif':[variance_inflation_factor(np.array(df_train_woe[sel_var]), i) for i in range(len(sel_var))]})
#        ## 删除 vif 大于10的变量
#        index_1 = df_vif.vif > 10
#        if sum(index_1)>0:
#            df_vif = df_vif.loc[~index_1,]
#        sel_var = list(df_vif.value)


## 随机森林排序
## 特征选择
fs = FeatureSelector(data = df_train_woe[sel_var], labels = data_train_bin.target)
## 一次性去除所有的不满足特征
fs.identify_all(selection_params = {'missing_threshold': 0.9, 
                                    'correlation_threshold': 0.8, 
                                    'task': 'classification', 
                                    'eval_metric': 'binary_error',
                                    'max_depth':2,
                                    'cumulative_importance': 0.90})
df_train_woe = fs.remove(methods = 'all')
df_train_woe['target'] = data_train_bin.target


### 模型训练 ########################################################


print('原始训练集中，好/坏样本比例为',int(sum(data_train.target==0)/data_train.target.sum()),': 1，需要做样本均衡处理')

var_woe_name = list(df_train_woe.columns)
var_woe_name.remove('target')
    
## 随机抽取一些好样本，与坏样本合并，再用 SMOTE 生成一个新的样本训练集
df_temp_normal = df_train_woe[df_train_woe.target==0] # 筛出好样本
df_temp_normal.reset_index(drop=True,inplace=True)
index_1 = np.random.randint( low = 0,high = df_temp_normal.shape[0]-1,size=20000)
index_1 = np.unique(index_1)
# print('在含有',df_train_woe.shape[0],'个样本的训练集中，随机抽取 20000 个 target = 0 的好样本，去重后剩余的随机好样本量为：',len(index_1))

df_temp =  df_temp_normal.loc[index_1]
index_2 = [x for x in range(df_temp_normal.shape[0]) if x not in index_1 ] # 剩余没有被随机抽到的好样本
df_temp_other = df_temp_normal.loc[index_2]
df_temp = pd.concat([df_temp,df_train_woe[df_train_woe.target==1]],axis=0,ignore_index=True)
# print('用剩余好样本 + 坏样本 = 新样本集，样本量为：',df_temp.shape[0])

## 用随机抽取的样本做样本生成
sm_sample_1 = SMOTE(random_state=10,sampling_strategy=1,k_neighbors=5)# kind='borderline1'，ratio=0.5
x_train, y_train = sm_sample_1.fit_resample(df_temp[var_woe_name], df_temp.target)
print('用 SMOTE 在随机样本集中，构建 K=5 的近邻领域，并在其中生成少数样本后，样本总量为：',x_train.shape[0])

## 合并数据
x_train = np.vstack([x_train, np.array(df_temp_other[var_woe_name])])
y_train = np.hstack([y_train, np.array(df_temp_other.target)])

print('最终进行训练的样本集中，好/坏样本比例为',int(sum(y_train==0)/sum(y_train)),': 1')


#    var_woe_name = sel_var
#    x_train = df_train_woe[var_woe_name]
#    x_train = np.array(x_train)
#    y_train = np.array(data_train_bin.target)

del_list = []
for s in var_woe_name:
    index_s = df_test_woe[s].isnull()
    if sum(index_s)> 0:
        del_list.extend(list(df_test_woe.index[index_s]))
if len(del_list)>0:
    list_1 = [x for x in list(df_test_woe.index) if x not in del_list ]
    df_test_woe = df_test_woe.loc[list_1]
        
    x_test = df_test_woe[var_woe_name]
    x_test = np.array(x_test)
    y_test = np.array(df_test_woe.target.loc[list_1])
else:
    x_test = df_test_woe[var_woe_name]
    x_test = np.array(x_test)
    y_test = np.array(df_test_woe.target)


### logistic 逻辑回归模型建模
## 设置待优化的超参数
lr_param = {'C': [0.01, 0.1, 0.2, 0.5, 1, 1.5, 2],
            'class_weight': [{1: 1, 0: 1},  {1: 2, 0: 1}, {1: 3, 0: 1}, {1: 5, 0: 1}]}
## 初始化网格搜索
lr_gsearch = GridSearchCV(
        estimator=LogisticRegression(random_state=0, fit_intercept=True, penalty='l2', solver='saga'),
        param_grid=lr_param, cv=3, scoring='f1', n_jobs=-1, verbose=2)
## 执行超参数优化
lr_gsearch.fit(x_train, y_train)
print('LR逻辑回归模型最优得分 {0},\n最优参数{1}'.format(lr_gsearch.best_score_,lr_gsearch.best_params_))

## 用最优参数，初始化logistic模型
LR_model = LogisticRegression(C=lr_gsearch.best_params_['C'], penalty='l2', solver='saga',
                                class_weight=lr_gsearch.best_params_['class_weight'])
## 训练logistic模型
#    LR_model = LogisticRegression(C=0.01, penalty='l2', solver='saga',
#                                    class_weight={1: 3, 0: 1})
LR_model_fit = LR_model.fit(x_train, y_train)


## 模型评估
y_pred = LR_model_fit.predict(x_test)
## 计算混淆矩阵与recall、precision
cnf_matrix = pd.DataFrame(data=confusion_matrix(y_test, y_pred),
                          columns=['预测结果为正例','预测结果为反例'],index=['真实样本为正例','真实样本为反例'])
recall_value = recall_score(y_test, y_pred)
precision_value = precision_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print('LR 逻辑回归模型的召回率：',recall_value)
print('LR 逻辑回归模型的精准率：',precision_value)
print('LR 逻辑回归模型预测的正确率:',acc)
print('='*80)
print('测试集的混淆矩阵：')
cnf_matrix

## 给出概率预测结果
y_score_test = LR_model_fit.predict_proba(x_test)[:, 1]
## 计算fpr与tpr
fpr, tpr, thresholds = roc_curve(y_test, y_score_test)  
## 计算AR、gini等
roc_auc = auc(fpr, tpr)
ks = max(tpr - fpr)
ar = 2*roc_auc-1
gini = ar
print('测试集的 AR/基尼系数为:',ar)
print('测试集的KS值为:',ks)
print('测试集的AUC值为:',roc_auc)


## ks曲线    
plt.figure(figsize=(10,6))
plt.plot(np.linspace(0,1,len(tpr)),tpr,'--',color='red', label='正样本洛伦兹曲线')
plt.plot(np.linspace(0,1,len(tpr)),fpr,':',color='blue', label='负样本洛伦兹曲线')
plt.plot(np.linspace(0,1,len(tpr)),tpr - fpr,'-',color='green')
plt.grid()
plt.xticks( fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel('概率分组',fontsize=16)
plt.ylabel('累积占比%',fontsize=16)
plt.legend(fontsize=16)
print('最大KS值:',max(tpr - fpr))


## ROC曲线
plt.figure(figsize=(10,6))
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks( fontsize=16)
plt.yticks( fontsize=16)
plt.xlabel('FPR',fontsize=16)
plt.ylabel('TPR',fontsize=16)
plt.title('ROC',fontsize=16)
plt.legend(loc="lower right",fontsize=16)



# 生成评分卡 ########################################################


## 保存模型的参数用于计算评分
var_woe_name.append('intercept')
## 提取权重
weight_value = list(LR_model_fit.coef_.flatten())
## 提取截距项
weight_value.extend(list(LR_model_fit.intercept_))
dict_params = dict(zip(var_woe_name,weight_value))

## 提取 训练集、测试集 样本分数
y_score_train = LR_model_fit.predict_proba(x_train)[:, 1]
y_score_test = LR_model_fit.predict_proba(x_test)[:, 1]

## 生成评分卡
df_score,dict_bin_score,params_A,params_B,score_base = create_score(dict_woe_map,
                                                dict_params,dict_cont_bin,dict_disc_bin)
print('参数 A 取值:',params_A)
print('='*80)
print('参数 B 取值:',params_B)
print('='*80)
print('基准分数:',score_base)


var_bin_score = pd.DataFrame(dict_bin_score)
print('全部',var_bin_score.shape[1],'个变量，不同取值 bins(分箱) 所对应的分数：')
var_bin_score.sort_index()


## 计算样本评分
df_all = pd.concat([data_train,data_test],axis = 0)
df_all_score = cal_score(df_all,dict_bin_score,dict_cont_bin,dict_disc_bin,score_base)
df_all_score.score[df_all_score.score >900] = 900
print('样本最高分：',df_all_score.score.max())
print('样本最低分：',df_all_score.score.min())
print('样本平均分：',df_all_score.score.mean())
print('样本中位数得分：',df_all_score.score.median())
print('='*80)
print('全部样本的变量得分情况：')
df_all_score


## 评分卡区间分数统计
good_total = sum(df_all_score.target == 0)
bad_total = sum(df_all_score.target == 1)
score_bin = np.arange(300,950,50)
bin_rate = []
bad_rate = []
ks = []
good_num = []
bad_num = []
for i in range(len(score_bin)-1):
    ## 取出分数区间的样本
    if score_bin[i+1] == 900:
        index_1 = (df_all_score.score >= score_bin[i]) & (df_all_score.score <= score_bin[i+1]) 
    else:
        index_1 = (df_all_score.score >= score_bin[i]) & (df_all_score.score < score_bin[i+1]) 
    df_temp = df_all_score.loc[index_1,['target','score']]
    ## 计算该分数区间的指标
    good_num.append(sum(df_temp.target==0))
    bad_num.append(sum(df_temp.target==1))
    ## 区间样本率
    bin_rate.append(df_temp.shape[0]/df_all_score.shape[0]*100)
    ## 坏样本率
    bad_rate.append(df_temp.target.sum()/df_temp.shape[0]*100)
    ## 以该分数为注入分数的ks值
    ks.append(sum(bad_num[0:i+1])/bad_total - sum(good_num[0:i+1])/good_total )


index_range = ['[ 300-350 )','[ 350-400 )','[ 400-450 )','[ 450-500 )','[ 500-550 )','[ 550-600 ) ','[ 600-650 ) ',
               '[ 650-700 )','[ 700-750 )','[ 750-800 )','[ 800-850 )','[ 850-900 ]']
df_result = pd.DataFrame({'好信用数量':good_num,'坏信用数量':bad_num,'区间样本率':bin_rate,
                            '坏信用率':bad_rate,'KS值(真正率-假正率)':ks},index=index_range)
print('评分卡12个区间分数统计结果如下：')
df_result

score_all = df_all_score.loc[:,['score','target']]
target_0_score = score_all.loc[score_all['target']==0]
target_1_score = score_all.loc[score_all['target']==1]


print('观察不同分数段中，好坏信用样本频数分布:\n好样本采用左侧纵轴刻度，坏样本采用右侧纵轴刻度')

bar_width = 0.3
fig, ax1 = plt.subplots(figsize=(15,6))
plt.bar(np.arange(0,12)+ bar_width,df_result.iloc[:,0],bar_width,alpha=0.7,color='blue', label='好信用样本量') 
ax1.set_ylabel('好信用样本频数',fontsize=17)
ax1.set_ylim([0,60000])
plt.grid(True)
plt.xlabel('分值区间',fontsize=17)
plt.xticks(np.arange(0,12),index_range,rotation=35,fontsize=15)
plt.yticks(fontsize=17)

# 共享横轴，双纵轴
ax2 = ax1.twinx()
ax2.bar(np.arange(0,12),df_result.iloc[:,1],bar_width,alpha=0.7,color='red', label='坏信用样本量')
ax2.set_ylabel('坏信用样本频数',fontsize=17)
ax2.set_ylim([0,500])
plt.yticks(fontsize=17)
plt.xlabel('分值区间',fontsize=17)

# 合并图例
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
plt.legend(handles1+handles2, labels1+labels2, loc='upper right',fontsize=17)
plt.show()


print('观察不同分数段中，好坏信用样本的概率密度分布：')
plt.figure(figsize=(15,6))
plt.hist(target_0_score.iloc[:,0],bins=200,alpha=0.5,label='好信用',
         color='blue',range=(300,900),density=True,rwidth=0.3,histtype='stepfilled')
plt.hist(target_1_score.iloc[:,0],bins=200,alpha=0.5,label='坏信用',
         color='red',range=(300,900),density=True,rwidth=0.3,histtype='stepfilled')
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlabel('分值区间',fontsize=17)
plt.ylabel('概率密度',fontsize=17)
plt.legend(fontsize=17)  
plt.show()

