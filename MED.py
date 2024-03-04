import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense, Dropout, Masking
from tensorflow.keras import Input
from tensorflow.keras import Model

#%% Load data
df = pd.read_csv('Field_App_Full_S50_2.csv')
df = df.drop('CDWater_BBLPerDAY',axis=1)
# Extract
feature_no = len(list(df))
data_cols = list(list(df)[i] for i in list(range(1,feature_no)))
data = df[data_cols].astype(float).to_numpy()

# qg index
qg_idx = data_cols.index('CDGas_MCFPerDAY')
shut_idx = data_cols.index('ShutIns')
peak_idx = data_cols.index('Max_time')
peakt_idx = data_cols.index('Max_Value')

#%% Data Normalozation
norm_data = data
scaler = MinMaxScaler()
scaler.fit(data)
norm_data = scaler.transform(norm_data)

max_qg = np.max(data[:,qg_idx])
min_qg = np.min(data[:,qg_idx])
print(max_qg)
print(min_qg)
print(norm_data.shape)

t_steps = 36 #reshaped time steps
n_wells = int(len(df)/36)
data_total = norm_data.reshape(n_wells,t_steps,data.shape[1])
data_total_list = data_total.tolist()
random.Random(99).shuffle(data_total_list)
data_total = np.array(data_total_list)

# split training and testing data
tr_por = 2499
data_train = data_total[:tr_por,:,:]
data_test = data_total[tr_por:,:,:]
print(data_train.shape)
print(data_test.shape)

data_total = data.reshape(n_wells,t_steps,data.shape[1])
data_total_list = data_total.tolist()
random.Random(99).shuffle(data_total_list)
data_total = np.array(data_total_list)
data_test_org = data_total[tr_por:,:,:]

# Test Data Normalozation
data_test_org_trans = data_test_org.reshape(200*t_steps,data.shape[1])
print(data_test_org_trans.shape)
scaler = MinMaxScaler()
scaler.fit(data)
norm_data_test_org_trans = scaler.transform(data_test_org_trans)
norm_data_test_org = norm_data_test_org_trans.reshape(200,t_steps,data.shape[1])

#%% Process data: Delay and segmentation
feature_list = list(range(len(list(df))-1))
exampt_list = [i for i in feature_list if i not in [shut_idx]]
exampt_list2 = [i for i in feature_list if i not in [qg_idx]]
exampt_list3 = [i for i in feature_list if i not in [peak_idx,peakt_idx,qg_idx]]

def delay_embedding_MRA_de(train,m,d,exampt_list,shut_idx,max_len):
    en_trainX = []
    de_trainX = []
    trainY = []
    i = 0
    en_trainx_val = train[i:i+m*d:d]
    en_trainx_val_padded = nan_padding(en_trainx_val,max_len)

    en_trainX.append(en_trainx_val_padded)
    
    de_trainX.append(train[:, exampt_list3])
    
    trainY.append(train[:, [qg_idx]]) # colume want to predict

    return np.array(en_trainX), np.array(de_trainX), np.array(trainY)

def nan_padding(sub_list,max_length):
    padded_array = np.empty((max_length,sub_list.shape[1]))
    padded_array[:] = np.nan
    current_t = sub_list.shape[0]
    padded_array[:current_t,:] = sub_list
        
    return padded_array

def zero_padding(sub_list,max_length):
    padded_array = np.zeros((max_length,sub_list.shape[1]))
    current_t = sub_list.shape[0]
    padded_array[:current_t,:] = sub_list
        
    return padded_array

# Training data
d = 1
m_all = list(range(0,t_steps))
max_len = t_steps
en_trainXX = []
de_trainXX = []
trainYY = []
x = data_train

# 2499*36 = 89964
for i_delay in range(x.shape[0]):
    for m in m_all:
        train_i = x[i_delay,:,:] 
        en_trainX, de_trainX, trainY = delay_embedding_MRA_de(train_i,m,d,exampt_list,shut_idx,max_len)

        en_trainXX.append(en_trainX.tolist()[0])
        de_trainXX.append(de_trainX.tolist()[0])
        trainYY.append(trainY.tolist()[0])

# re-format
encoder_inputs = np.array(en_trainXX)   
decoder_inputs = np.array(de_trainXX)  
labels = np.array(trainYY)  
print('Encoder Input shape == {}'.format(encoder_inputs.shape))
print('Decoder Input shape == {}'.format(decoder_inputs.shape))
print('Label shape == {}'.format(labels.shape))

# Testing data
d = 1
m_all = list(range(t_steps))
max_len = t_steps
en_trainXX_te = []
de_trainXX_te = []
trainYY_te = []
x = data_test

# 200*36 = 7200
for i_delay in range(x.shape[0]):
    for m in m_all:
        train_i = x[i_delay,:,:] 
        en_trainX, de_trainX, trainY = delay_embedding_MRA_de(train_i,m,d,exampt_list,shut_idx,max_len)

        en_trainXX_te.append(en_trainX.tolist()[0])
        de_trainXX_te.append(de_trainX.tolist()[0])
        trainYY_te.append(trainY.tolist()[0])
        
print(len(en_trainXX_te))
print(len(de_trainXX_te))   
print(len(trainYY_te))

# re-format
encoder_inputs_te = np.array(en_trainXX_te)   
decoder_inputs_te = np.array(de_trainXX_te)  
labels_te = np.array(trainYY_te)  
print('Encoder Input shape == {}'.format(encoder_inputs_te.shape))
print('Decoder Input shape == {}'.format(decoder_inputs_te.shape))
print('Label shape == {}'.format(labels_te.shape))

# Pad with special value
mask_special_val = -1e9
nn_inputs = np.nan_to_num(encoder_inputs, nan=mask_special_val)
nn_inputs_te = np.nan_to_num(encoder_inputs_te, nan=mask_special_val)

#%% Build the neural network
# Encoder
encoder_input = Input(shape=(encoder_inputs.shape[1], encoder_inputs.shape[2]))
mask_input = Masking(mask_value=mask_special_val)(encoder_input)
encoder_gru1 = GRU(64, return_sequences=True)(mask_input)
encoder_gru1 = Dropout(0.1)(encoder_gru1)
encoder_gru2 = GRU(64, return_sequences=False)(encoder_gru1)
encoder_gru2 = Dropout(0.1)(encoder_gru2)

# Decoder
decoder_input = Input(shape=(decoder_inputs.shape[1], decoder_inputs.shape[2]))
decoder_gru1 = GRU(64, return_sequences=True)(decoder_input, initial_state=encoder_gru2)
decoder_gru1 = Dropout(0.1)(decoder_gru1)
decoder_gru2 = GRU(64, return_sequences=True)(decoder_gru1, initial_state=encoder_gru2)
decoder_gru2 = Dropout(0.1)(decoder_gru2)
dense = Dense(64, activation='tanh')(decoder_gru2)
decoder_output = Dense(1, activation='tanh')(dense)

# Define the model
model = Model([encoder_input, decoder_input], decoder_output)

# Compile the model
model.compile(loss='mse', optimizer='adam')
model.summary()

#%% Train the model
start_time = time.time() 
history =  model.fit([nn_inputs,decoder_inputs],
                     labels, epochs=50, batch_size= 200, 
                     verbose=1,validation_split = 0.1)
print("--- %s seconds ---" % (time.time() - start_time))

# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#%% Make a Prediction
def MED_Process(predict_num,prod_period,curr_model):
    filename="MD_0714_cloud_weight"+str(curr_model+1)
    filepath = filename+".hdf5"
    model.load_weights(filepath)
    
    y_pred = model.predict([nn_inputs_te,decoder_inputs_te])
    qg_pred = y_pred[predict_num*36+prod_period,:,:]
    
    return qg_pred

def local_RMSE(testY_org,testPredict):
    # Local RMSE
    scaler = MinMaxScaler()
    scaler.fit(testY_org.reshape(-1,1))
    norm_y_label = scaler.transform(testY_org.reshape(-1,1))
    norm_y_pred = scaler.transform(testPredict.reshape(-1,1))
    testScore = mean_squared_error(norm_y_label, norm_y_pred,squared=False)
    return testScore

plot_num = 66 # pick the case want to plot 
row = 1
col = 4
model_num = 5
fig, ax = plt.subplots(row,col,figsize=(17,2.5))
if isinstance(ax, np.ndarray):
    ax = ax.flatten()
    
for i, subplot_ax in enumerate(ax):
    if i == 0:
        plot_t = i
    else:
        plot_t = i*2-1
    testScore_all = []
    subplot_ax.plot(list(range(plot_t)), labels_te[plot_num*36,:plot_t,:]*(max_qg - min_qg) + min_qg,'ro-',label='Production Data',alpha=0.3)
    subplot_ax.plot(list(range(36)), data_test[plot_num,:,qg_idx]*(max_qg - min_qg) + min_qg,'k-',label='Label',alpha=0.1)
    
    for j in range(model_num):
        qg_pred = MED_Process(plot_num,plot_t,j)
        label_predict = 'Prediction'+ str(j+1)
        subplot_ax.plot(list(range(plot_t,36)), qg_pred[plot_t:,:]*(max_qg - min_qg) + min_qg,label=label_predict)
        qg_pred = qg_pred.reshape(-1)
        qg_pred[:plot_t] = data_test[plot_num,:plot_t,qg_idx]

        testScore = local_RMSE(qg_pred,data_test[plot_num,:,qg_idx])
        testScore_all.append(testScore)
    
    if i == 0:
        legend = subplot_ax.legend(loc='upper right',ncol=1,labelspacing=0.1, handlelength=1.0)
        for text in legend.get_texts():
            text.set_fontsize(10)
        ymin = 0
        ymax = np.max(qg_pred)*((max_qg - min_qg) + min_qg)*2.4
        subplot_ax.set_ylim([ymin, ymax])
    else:
        ymax = np.max(data_test[plot_num,:,qg_idx])*((max_qg - min_qg) + min_qg)*1.3
        subplot_ax.set_ylim([ymin, ymax])
    
    testScore_ave = sum(testScore_all)/len(testScore_all)
    subplot_ax.set_title('RMSE(ave): %.6f\n' % (testScore_ave)+ ' Producing Time(month): %d' %(plot_t))
    subplot_ax.set_xlabel('Time (Month)')
    subplot_ax.set_ylabel('Gas Rate (MCF/day)')
    
plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.95, wspace=0.32, hspace=0.6)

#%% Evaluation
def MED_Process_group(curr_model):
    filename="MD_0714_cloud_weight"+str(curr_model+1)
    filepath = filename+".hdf5"
    model.load_weights(filepath)
    
    y_pred = model.predict([nn_inputs_te,decoder_inputs_te])
    
    return y_pred

def find_RMSE_MED(model_num,prod_period):
    testScore_all = []
    for j in range(model_num):
        qg_pred_all = MED_Process_group(j)
        
        testScore_store = []
        for i in range(data_test.shape[0]):
            qg_pred = qg_pred_all[i*36+prod_period].reshape(-1)
            qg_pred[:prod_period] = data_test[i,:prod_period,qg_idx] # ignore the error form the kown period
            testScore = local_RMSE(qg_pred,data_test[i,:,qg_idx])
            testScore_store.append(testScore)
        testScore_arr = np.array(testScore_store)
        testScore_all.append(testScore_arr)
        
    testScore_ave = sum(testScore_all)/len(testScore_all)
    return testScore_ave

def calculate_average_without_outliers(data):
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the lower and upper bounds to identify outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers from the data
    data_without_outliers = [x for x in data if lower_bound <= x <= upper_bound]

    # Calculate the average of the data without outliers
    average_without_outliers = np.mean(data_without_outliers)

    return average_without_outliers

qua_idx_store = []
quantile_store = []
maxx = max_qg
minn = min_qg

for i in [0,1,3,5]:
    prod_period = i
    model_num = 5
    RMSE_qg = find_RMSE_MED(model_num,prod_period)
    RMSE_qg_ave = np.mean(RMSE_qg)
    #RMSE_qg_ave = calculate_average_without_outliers(RMSE_qg)
    print('RMSE:',RMSE_qg_ave)
    
    quantiles = np.percentile(RMSE_qg,(10,50,90))
    qua_idx = [(np.abs(RMSE_qg - ii)).argmin() for ii in quantiles]
    quantile_store.append(quantiles)
    qua_idx_store.append(qua_idx)
    
    print(str(i),':',quantiles)
    print(str(i),':',qua_idx)