# **DATA PREPROCESSING**

Importing the libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
```


```python

import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='bM-UBMSwFCa7R0jKoO6AyQaUYNWQfm0p3Oqyqmh7so4x',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'crudeoilpriceprediction-donotdelete-pr-ikcsbsjvdluquo'
object_key = 'Crude Oil Prices Daily.xlsx'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']

data = pd.read_excel(body.read())
data.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Closing Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1986-01-02</td>
      <td>25.56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1986-01-03</td>
      <td>26.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1986-01-06</td>
      <td>26.53</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1986-01-07</td>
      <td>25.85</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1986-01-08</td>
      <td>25.87</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Closing Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1986-01-02</td>
      <td>25.56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1986-01-03</td>
      <td>26.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1986-01-06</td>
      <td>26.53</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1986-01-07</td>
      <td>25.85</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1986-01-08</td>
      <td>25.87</td>
    </tr>
  </tbody>
</table>
</div>



# **Handling missing values**


```python
data.isnull().any()
```




    Date             False
    Closing Value     True
    dtype: bool




```python
data.isnull().sum()
```




    Date             0
    Closing Value    7
    dtype: int64




```python
data.dropna(axis=0,inplace=True)
```


```python
data_oil=data.reset_index()['Closing Value']
```


```python
data_oil
```




    0       25.56
    1       26.00
    2       26.53
    3       25.85
    4       25.87
            ...  
    8211    73.89
    8212    74.19
    8213    73.05
    8214    73.78
    8215    73.93
    Name: Closing Value, Length: 8216, dtype: float64




```python
data.isnull().any()
```




    Date             False
    Closing Value    False
    dtype: bool



# **Feature Scaling**


```python
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
data_oil=scaler.fit_transform(np.array(data_oil).reshape(-1,1))
```

## **Data Visualization**


```python
plt.title('Crude oil price')
plt.plot(data_oil)
```




    [<matplotlib.lines.Line2D at 0x7fe83b272160>]




    
![png](output_14_1.png)
    


## **Splitting data into Train and Test Data**


```python
training_size=int(len(data_oil)*0.65)
test_size=len(data_oil)-training_size
train_data,test_data=data_oil[0:training_size,:],data_oil[training_size:len(data_oil),:1]
```


```python
training_size,test_size
```




    (5340, 2876)




```python
train_data.shape
```




    (5340, 1)



## **Creating a dataset with sliding windows**


```python
def create_dataset (dataset, time_step=1):
    dataX, dataY = [], []

    for i in range(len(dataset)-time_step-1):

        a = dataset[i:(i+time_step), 0] 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])

    return np.array(dataX),np.array(dataY)
```


```python
time_step = 10

X_train, y_train=create_dataset(train_data,time_step)

X_test, y_test = create_dataset(test_data,time_step)
```


```python
print(X_train.shape),print(y_train.shape)
```

    (5329, 10)
    (5329,)





    (None, None)




```python
print(X_test.shape),print(y_test.shape)
```

    (2865, 10)
    (2865,)





    (None, None)




```python
X_train
```




    array([[0.11335703, 0.11661484, 0.12053902, ..., 0.10980305, 0.1089886 ,
            0.11054346],
           [0.11661484, 0.12053902, 0.11550422, ..., 0.1089886 , 0.11054346,
            0.10165852],
           [0.12053902, 0.11550422, 0.1156523 , ..., 0.11054346, 0.10165852,
            0.09906708],
           ...,
           [0.36731823, 0.35176958, 0.36080261, ..., 0.36391234, 0.37042796,
            0.37042796],
           [0.35176958, 0.36080261, 0.35354657, ..., 0.37042796, 0.37042796,
            0.37879461],
           [0.36080261, 0.35354657, 0.35295424, ..., 0.37042796, 0.37879461,
            0.37916482]])




```python
X_train.shape
```




    (5329, 10)




```python
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
```


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
```


```python
model=Sequential()
```


```python
model.add(LSTM(50,return_sequences=True,input_shape=(10,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
```


```python
model.add(Dense(1))
```


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lstm (LSTM)                 (None, 10, 50)            10400     
                                                                     
     lstm_1 (LSTM)               (None, 10, 50)            20200     
                                                                     
     lstm_2 (LSTM)               (None, 50)                20200     
                                                                     
     dense (Dense)               (None, 1)                 51        
                                                                     
    =================================================================
    Total params: 50,851
    Trainable params: 50,851
    Non-trainable params: 0
    _________________________________________________________________



```python
model.compile(loss='mean_squared_error',optimizer='adam')
```


```python
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=3,batch_size=64,verbose=1)
```

    Epoch 1/3
    84/84 [==============================] - 6s 27ms/step - loss: 0.0020 - val_loss: 9.2766e-04
    Epoch 2/3
    84/84 [==============================] - 1s 18ms/step - loss: 1.2612e-04 - val_loss: 7.7579e-04
    Epoch 3/3
    84/84 [==============================] - 1s 15ms/step - loss: 1.2418e-04 - val_loss: 8.0377e-04





    <keras.callbacks.History at 0x7fe81be4baf0>




```python
##Transformback to original form
train_predict=scaler.inverse_transform(train_data) 
test_predict=scaler.inverse_transform(test_data)
### Calculate RMSE performance metrics
import math 
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(train_data,train_predict))
```




    29.347830443269938




```python
from tensorflow.keras.models import load_model
```


```python
model.save("crude_oil.h5")
```


```python
### Plotting
look_back=10
trainpredictPlot = np.empty_like(data_oil)
trainpredictPlot[:, :]= np.nan
trainpredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictplot = np.empty_like(data_oil)
testPredictplot[:,: ] = np.nan
testPredictplot[look_back:len(test_predict)+look_back, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(data_oil))
plt.show()
```


    
![png](output_37_0.png)
    



```python
len(test_data)
```




    2876




```python
x_input=test_data[2866:].reshape(1,-1)
x_input.shape
```




    (1, 10)




```python
temp_input=list(x_input) 
temp_input=temp_input[0].tolist()
```


```python
temp_input
```




    [0.44172960165852215,
     0.48111950244335855,
     0.49726047682511476,
     0.4679401747371539,
     0.4729749740855915,
     0.47119798608026064,
     0.47341922108692425,
     0.4649785280616022,
     0.4703835332444839,
     0.47149415074781587]




```python
lst_output=[]
n_steps=10
i=0
while(i<10):
    if(len(temp_input)>10):
#print(temp_input)
       x_input=np.array(temp_input[1:]) 
       print("{} day input {}".format(i,x_input))
       x_input=x_input.reshape(1,-1)
       x_input = x_input.reshape((1, n_steps, 1)) #print(x_input)
       yhat = model.predict(x_input, verbose=0)
       print("{} day output {}".format(i,yhat))
       temp_input.extend(yhat[0].tolist())
       temp_input=temp_input[1:] #print(temp_input)
       lst_output.extend(yhat.tolist())
       i=i+1
    else:
       x_input = x_input.reshape((1, n_steps,1))
       yhat = model.predict(x_input, verbose=0)
       print(yhat[0])
       temp_input.extend(yhat[0].tolist()) 
       print(len(temp_input))
       lst_output.extend(yhat.tolist())
       i=i+1
```

    [0.47180176]
    11
    1 day input [0.4811195  0.49726048 0.46794017 0.47297497 0.47119799 0.47341922
     0.46497853 0.47038353 0.47149415 0.47180176]
    1 day output [[0.47561172]]
    2 day input [0.49726048 0.46794017 0.47297497 0.47119799 0.47341922 0.46497853
     0.47038353 0.47149415 0.47180176 0.47561172]
    2 day output [[0.47417307]]
    3 day input [0.46794017 0.47297497 0.47119799 0.47341922 0.46497853 0.47038353
     0.47149415 0.47180176 0.47561172 0.47417307]
    3 day output [[0.47077382]]
    4 day input [0.47297497 0.47119799 0.47341922 0.46497853 0.47038353 0.47149415
     0.47180176 0.47561172 0.47417307 0.47077382]
    4 day output [[0.47133595]]
    5 day input [0.47119799 0.47341922 0.46497853 0.47038353 0.47149415 0.47180176
     0.47561172 0.47417307 0.47077382 0.47133595]
    5 day output [[0.47125107]]
    6 day input [0.47341922 0.46497853 0.47038353 0.47149415 0.47180176 0.47561172
     0.47417307 0.47077382 0.47133595 0.47125107]
    6 day output [[0.47140375]]
    7 day input [0.46497853 0.47038353 0.47149415 0.47180176 0.47561172 0.47417307
     0.47077382 0.47133595 0.47125107 0.47140375]
    7 day output [[0.47126076]]
    8 day input [0.47038353 0.47149415 0.47180176 0.47561172 0.47417307 0.47077382
     0.47133595 0.47125107 0.47140375 0.47126076]
    8 day output [[0.47219065]]
    9 day input [0.47149415 0.47180176 0.47561172 0.47417307 0.47077382 0.47133595
     0.47125107 0.47140375 0.47126076 0.47219065]
    9 day output [[0.47236827]]



```python
day_new=np.arange(1,11) 
day_pred=np.arange(11,21)
len(data_oil)
plt.plot(day_new, scaler.inverse_transform(data_oil[8206:])) 
plt.plot(day_pred, scaler.inverse_transform(lst_output))
```




    [<matplotlib.lines.Line2D at 0x7fe808591130>]




    
![png](output_43_1.png)
    



```python
df3=data_oil.tolist() 
df3.extend(lst_output) 
plt.plot(df3[8100:])
```




    [<matplotlib.lines.Line2D at 0x7fe81bc04280>]




    
![png](output_44_1.png)
    



```python
df3=scaler.inverse_transform(df3).tolist()
```


```python
plt.plot(scaler.inverse_transform(data_oil))
```




    [<matplotlib.lines.Line2D at 0x7fe81bce5700>]




    
![png](output_46_1.png)
    



```python

# @hidden_cell
# The following code contains the credentials for a file in your IBM Cloud Object Storage.
# You might want to remove those credentials before you share your notebook.
metadata_1 = {
    'IAM_SERVICE_ID': 'iam-ServiceId-b9c18a80-9780-4270-bf88-fc8afee2ca22',
    'IBM_API_KEY_ID': 'bM-UBMSwFCa7R0jKoO6AyQaUYNWQfm0p3Oqyqmh7so4x',
    'ENDPOINT': 'https://s3.private.us.cloud-object-storage.appdomain.cloud',
    'IBM_AUTH_ENDPOINT': 'https://iam.cloud.ibm.com/oidc/token',
    'BUCKET': 'crudeoilpriceprediction-donotdelete-pr-ikcsbsjvdluquo',
    'FILE': 'Crude Oil Prices Daily.xlsx'
}

```


```python
!tar -zcvf trainedModel.tgz crude_oil.h5
```

    crude_oil.h5



```bash
%%bash
ls -ll
```

    total 1224
    -rw-rw---- 1 wsuser wscommon 667616 Nov 10 05:44 crude_oil.h5
    -rw-rw---- 1 wsuser wscommon 583825 Nov 10 05:46 trainedModel.tgz



```python
!pip install watson-machine-learning-client --upgrade
```

    Collecting watson-machine-learning-client
      Downloading watson_machine_learning_client-1.0.391-py3-none-any.whl (538 kB)
    [K     |████████████████████████████████| 538 kB 21.8 MB/s eta 0:00:01
    [?25hRequirement already satisfied: tqdm in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (4.62.3)
    Requirement already satisfied: pandas in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (1.3.4)
    Requirement already satisfied: certifi in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (2022.9.24)
    Requirement already satisfied: tabulate in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (0.8.9)
    Requirement already satisfied: ibm-cos-sdk in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (2.11.0)
    Requirement already satisfied: boto3 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (1.18.21)
    Requirement already satisfied: urllib3 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (1.26.7)
    Requirement already satisfied: lomond in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (0.3.3)
    Requirement already satisfied: requests in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (2.26.0)
    Requirement already satisfied: botocore<1.22.0,>=1.21.21 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from boto3->watson-machine-learning-client) (1.21.41)
    Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from boto3->watson-machine-learning-client) (0.10.0)
    Requirement already satisfied: s3transfer<0.6.0,>=0.5.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from boto3->watson-machine-learning-client) (0.5.0)
    Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from botocore<1.22.0,>=1.21.21->boto3->watson-machine-learning-client) (2.8.2)
    Requirement already satisfied: six>=1.5 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from python-dateutil<3.0.0,>=2.1->botocore<1.22.0,>=1.21.21->boto3->watson-machine-learning-client) (1.15.0)
    Requirement already satisfied: ibm-cos-sdk-core==2.11.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from ibm-cos-sdk->watson-machine-learning-client) (2.11.0)
    Requirement already satisfied: ibm-cos-sdk-s3transfer==2.11.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from ibm-cos-sdk->watson-machine-learning-client) (2.11.0)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from requests->watson-machine-learning-client) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from requests->watson-machine-learning-client) (3.3)
    Requirement already satisfied: pytz>=2017.3 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from pandas->watson-machine-learning-client) (2021.3)
    Requirement already satisfied: numpy>=1.17.3 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from pandas->watson-machine-learning-client) (1.20.3)
    Installing collected packages: watson-machine-learning-client
    Successfully installed watson-machine-learning-client-1.0.391



```python
from ibm_watson_machine_learning import APIClient
wml_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "bM-UBMSwFCa7R0jKoO6AyQaUYNWQfm0p3Oqyqmh7so4x"
}

client = APIClient(wml_credentials)
```


```python
def guid_from_space_name(client, space_name):
    space = client.spaces.get_details()
    return (next(item for item in space['resources'] if item['entity']["name"] == space_name)['metadata']['id'])
```


```python
space_uid = guid_from_space_name(client, 'crudeoilspace')
print("Space UID : ", space_uid)
```

    Space UID :  ce7bbf85-759f-4579-bfc5-9af2c9338c9b



```python
client.set.default_space(space_uid)
```




    'SUCCESS'




```python
client.software_specifications.list()
```

    -----------------------------  ------------------------------------  ----
    NAME                           ASSET_ID                              TYPE
    default_py3.6                  0062b8c9-8b7d-44a0-a9b9-46c416adcbd9  base
    kernel-spark3.2-scala2.12      020d69ce-7ac1-5e68-ac1a-31189867356a  base
    pytorch-onnx_1.3-py3.7-edt     069ea134-3346-5748-b513-49120e15d288  base
    scikit-learn_0.20-py3.6        09c5a1d0-9c1e-4473-a344-eb7b665ff687  base
    spark-mllib_3.0-scala_2.12     09f4cff0-90a7-5899-b9ed-1ef348aebdee  base
    pytorch-onnx_rt22.1-py3.9      0b848dd4-e681-5599-be41-b5f6fccc6471  base
    ai-function_0.1-py3.6          0cdb0f1e-5376-4f4d-92dd-da3b69aa9bda  base
    shiny-r3.6                     0e6e79df-875e-4f24-8ae9-62dcc2148306  base
    tensorflow_2.4-py3.7-horovod   1092590a-307d-563d-9b62-4eb7d64b3f22  base
    pytorch_1.1-py3.6              10ac12d6-6b30-4ccd-8392-3e922c096a92  base
    tensorflow_1.15-py3.6-ddl      111e41b3-de2d-5422-a4d6-bf776828c4b7  base
    runtime-22.1-py3.9             12b83a17-24d8-5082-900f-0ab31fbfd3cb  base
    scikit-learn_0.22-py3.6        154010fa-5b3b-4ac1-82af-4d5ee5abbc85  base
    default_r3.6                   1b70aec3-ab34-4b87-8aa0-a4a3c8296a36  base
    pytorch-onnx_1.3-py3.6         1bc6029a-cc97-56da-b8e0-39c3880dbbe7  base
    kernel-spark3.3-r3.6           1c9e5454-f216-59dd-a20e-474a5cdf5988  base
    pytorch-onnx_rt22.1-py3.9-edt  1d362186-7ad5-5b59-8b6c-9d0880bde37f  base
    tensorflow_2.1-py3.6           1eb25b84-d6ed-5dde-b6a5-3fbdf1665666  base
    spark-mllib_3.2                20047f72-0a98-58c7-9ff5-a77b012eb8f5  base
    tensorflow_2.4-py3.8-horovod   217c16f6-178f-56bf-824a-b19f20564c49  base
    runtime-22.1-py3.9-cuda        26215f05-08c3-5a41-a1b0-da66306ce658  base
    do_py3.8                       295addb5-9ef9-547e-9bf4-92ae3563e720  base
    autoai-ts_3.8-py3.8            2aa0c932-798f-5ae9-abd6-15e0c2402fb5  base
    tensorflow_1.15-py3.6          2b73a275-7cbf-420b-a912-eae7f436e0bc  base
    kernel-spark3.3-py3.9          2b7961e2-e3b1-5a8c-a491-482c8368839a  base
    pytorch_1.2-py3.6              2c8ef57d-2687-4b7d-acce-01f94976dac1  base
    spark-mllib_2.3                2e51f700-bca0-4b0d-88dc-5c6791338875  base
    pytorch-onnx_1.1-py3.6-edt     32983cea-3f32-4400-8965-dde874a8d67e  base
    spark-mllib_3.0-py37           36507ebe-8770-55ba-ab2a-eafe787600e9  base
    spark-mllib_2.4                390d21f8-e58b-4fac-9c55-d7ceda621326  base
    xgboost_0.82-py3.6             39e31acd-5f30-41dc-ae44-60233c80306e  base
    pytorch-onnx_1.2-py3.6-edt     40589d0e-7019-4e28-8daa-fb03b6f4fe12  base
    default_r36py38                41c247d3-45f8-5a71-b065-8580229facf0  base
    autoai-ts_rt22.1-py3.9         4269d26e-07ba-5d40-8f66-2d495b0c71f7  base
    autoai-obm_3.0                 42b92e18-d9ab-567f-988a-4240ba1ed5f7  base
    pmml-3.0_4.3                   493bcb95-16f1-5bc5-bee8-81b8af80e9c7  base
    spark-mllib_2.4-r_3.6          49403dff-92e9-4c87-a3d7-a42d0021c095  base
    xgboost_0.90-py3.6             4ff8d6c2-1343-4c18-85e1-689c965304d3  base
    pytorch-onnx_1.1-py3.6         50f95b2a-bc16-43bb-bc94-b0bed208c60b  base
    autoai-ts_3.9-py3.8            52c57136-80fa-572e-8728-a5e7cbb42cde  base
    spark-mllib_2.4-scala_2.11     55a70f99-7320-4be5-9fb9-9edb5a443af5  base
    spark-mllib_3.0                5c1b0ca2-4977-5c2e-9439-ffd44ea8ffe9  base
    autoai-obm_2.0                 5c2e37fa-80b8-5e77-840f-d912469614ee  base
    spss-modeler_18.1              5c3cad7e-507f-4b2a-a9a3-ab53a21dee8b  base
    cuda-py3.8                     5d3232bf-c86b-5df4-a2cd-7bb870a1cd4e  base
    autoai-kb_3.1-py3.7            632d4b22-10aa-5180-88f0-f52dfb6444d7  base
    pytorch-onnx_1.7-py3.8         634d3cdc-b562-5bf9-a2d4-ea90a478456b  base
    spark-mllib_2.3-r_3.6          6586b9e3-ccd6-4f92-900f-0f8cb2bd6f0c  base
    tensorflow_2.4-py3.7           65e171d7-72d1-55d9-8ebb-f813d620c9bb  base
    spss-modeler_18.2              687eddc9-028a-4117-b9dd-e57b36f1efa5  base
    -----------------------------  ------------------------------------  ----
    Note: Only first 50 records were displayed. To display more use 'limit' parameter.



```python
software_spec_uid = client.software_specifications.get_uid_by_name("tensorflow_rt22.1-py3.9")
software_spec_uid
```




    'acd9c798-6974-5d2f-a657-ce06e986df4d'




```python
model_details = client.repository.store_model(model="trainedModel.tgz", meta_props={
    client.repository.ModelMetaNames.NAME: "sequential",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid,
    client.repository.ModelMetaNames.TYPE: "tensorflow_2.7"})
model_id = client.repository.get_model_uid(model_details)
```

    This method is deprecated, please use get_model_id()



```python
model_id
```




    '915fd04b-978d-4eee-8a33-1d6e4aca0d52'




```python
X_train[0]
```




    array([[0.11335703],
           [0.11661484],
           [0.12053902],
           [0.11550422],
           [0.1156523 ],
           [0.11683696],
           [0.1140234 ],
           [0.10980305],
           [0.1089886 ],
           [0.11054346]])




```python
model_id
```




    '915fd04b-978d-4eee-8a33-1d6e4aca0d52'




```python
model.predict(([[0.11335703],
       [0.11661484],
       [0.12053902],
       [0.11550422],
       [0.1156523 ],
       [0.11683696],
       [0.1140234 ],
       [0.10980305],
       [0.1089886 ],
       [0.11054346]]))
```

    WARNING:tensorflow:Model was constructed with shape (None, 10, 1) for input KerasTensor(type_spec=TensorSpec(shape=(None, 10, 1), dtype=tf.float32, name='lstm_input'), name='lstm_input', description="created by layer 'lstm_input'"), but it was called on an input with incompatible shape (None, 1, 1).





    array([[0.01902138],
           [0.01907121],
           [0.01913123],
           [0.01905422],
           [0.01905649],
           [0.0190746 ],
           [0.01903157],
           [0.01896702],
           [0.01895457],
           [0.01897835]], dtype=float32)




```python

```
