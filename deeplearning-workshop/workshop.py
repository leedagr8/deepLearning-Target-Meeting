from tensorflow.keras.layers import Dense #Dense Layer
from tensorflow.keras.models import Sequential #whatever we do to the model will happen in sequence
import panda as pd # can call panda as pd in my code

#get data
data = pd.read_csv('linear.csv', header=0, index_col=0)
#print(data.head()) #verify and make sure it's printing the correct info

indicies = data.index.values 
#print(indices)
values = data['value'.values] #we want the column value and its values
print(values)

model = Sequential()

model.add(Dense(8, input_shape=(1, ))) #adding a layer (input + hidden)
#8= nodes , (1, TBD (however many rows))

model.add(Dense(32))
#add another layer with 32 nodes. Grabs the shape of the previous layer

#LSTM - model that allows you to pass in states...

model.add(Dense(1))
#OUTPUT LAYER = number of predictions you want

model.compile(optimizer ='adam', loss='mae') 
#compile the model. can add optimizer(adam- minimizes errors) and loss function (mae - Mean absolute error)
model.fit(indicies, values, epochs=1, batch_size=1)
#epochs = 1 iteration through the entire data set
#batch size - send 1 data at a time and update your weights
# weights are the estimation
# weights initialize randomly

test_index = [5000, 5001, 5002]

for index in test_index:

    prediction = model.predict(test_index)

    print(int(prediction))