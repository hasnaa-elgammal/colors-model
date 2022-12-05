import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import plotly.offline as py
import plotly.graph_objs as go

#Getting data and processing
dataset = pd.read_csv('final_data.csv')
dataset = pd.get_dummies(dataset, columns=['label'])
dataset = dataset[['red', 'green', 'blue', 'label_Red', 'label_Green', 'label_Blue', 'label_Yellow', 'label_Orange', 'label_Pink', 'label_Purple', 'label_Brown', 'label_Grey', 'label_Black', 'label_White']]

# Split the data into train and test
train_dataset = dataset.sample(frac=0.8, random_state=9)
test_dataset = dataset.drop(train_dataset.index)

#Split features: `red`, `green`, `blue` and labels
train_labels = pd.DataFrame([train_dataset.pop(x) for x in ['label_Red', 'label_Green', 'label_Blue', 'label_Yellow', 'label_Orange', 'label_Pink', 'label_Purple', 'label_Brown', 'label_Grey', 'label_Black', 'label_White']]).T

test_labels = pd.DataFrame([test_dataset.pop(x) for x in ['label_Red', 'label_Green', 'label_Blue', 'label_Yellow', 'label_Orange', 'label_Pink', 'label_Purple', 'label_Brown', 'label_Grey', 'label_Black', 'label_White']]).T

#Model
model = keras.Sequential([
    layers.Dense(3, activation='relu', input_shape=[len(train_dataset.keys())]), #inputshape=[3]
    layers.Dense(32, activation='relu'),
    layers.Dense(11)
  ])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=['accuracy'])
model.summary()

# Train the model
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0.01, patience=100)
history = model.fit(x=train_dataset, y=train_labels, 
                    validation_split=0.2, 
                    epochs=1000, 
                    batch_size=32, 
                    verbose=0,
                    callbacks=[early_stop,tfdocs.modeling.EpochDots()], 
                    shuffle=True)

#Plot epochs with accuracy & loss function
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric = "accuracy")
plt.ylim([0, 1])
plt.ylabel('accuracy [Color]')
plotter.plot({'Basic': history}, metric = "loss")
plt.ylim([0, 1])
plt.ylabel('loss [Color]')

#Save model
model.save('colormodel.h5')

# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('colormodel.h5') #very important

# Show the model architecture
model.summary()

"""# Make Prediction
The prediction by the model is an array of 11 numbers. 
They represent the model's "confidence" that the RGB color corresponds to each of the 11 different classes of color as follows:

* 0 for Red
* 1 for Green
* 2 for Blue 
* 3 for Yellow
* 4 for Orange
* 5 for Pink
* 6 for Purple
* 7 for Brown
* 8 for Grey
* 9 for Black
* 10 for White 

You can see which out of the 11 labels has the highest confidence value

## Train Dataset Prediction
"""

train_predictions = model.predict(train_dataset)
print(train_predictions)

"""### Selecting Class with highest confidence"""

actual_encoded_train_labels = np.argmax(train_labels.to_numpy(), axis=1) #train_labels were originally in one-hot
print(actual_encoded_train_labels)

predicted_encoded_train_labels = np.argmax(train_predictions, axis=1)
print(predicted_encoded_train_labels)

"""### Converting numpy array to pandas dataframe"""

actual_encoded_train_labels = pd.DataFrame(actual_encoded_train_labels, columns=['Labels'])
print(actual_encoded_train_labels)

predicted_encoded_train_labels = pd.DataFrame(predicted_encoded_train_labels, columns=['Labels'])
print(predicted_encoded_train_labels)

"""### Visualize Prediction for Train Dataset"""

#Plot Actual vs Predicted Class for Training Dataset
actual_chart = go.Scatter(x=actual_encoded_train_labels.index, y=actual_encoded_train_labels.Labels, name= 'Actual Label')
predict_chart = go.Scatter(x=actual_encoded_train_labels.index, y=predicted_encoded_train_labels.Labels, name= 'Predicted Label')
py.iplot([predict_chart, actual_chart])

"""## Test Dataset Prediction"""

test_predictions = model.predict(test_dataset)
print(test_predictions)

"""### Selecting Class with highest confidence"""

actual_encoded_test_labels = np.argmax(test_labels.to_numpy(), axis=1) 
print(actual_encoded_test_labels)

predicted_encoded_test_labels = np.argmax(test_predictions, axis=1)
print(predicted_encoded_test_labels)

"""### Converting numpy array to pandas dataframe"""

actual_encoded_test_labels = pd.DataFrame(actual_encoded_test_labels, columns=['Labels'])
print(actual_encoded_test_labels)

predicted_encoded_test_labels = pd.DataFrame(predicted_encoded_test_labels, columns=['Labels'])
print(predicted_encoded_test_labels)

"""### Visualize Prediction for Test Dataset"""

#Plot Actual vs Predicted Class for Test Dataset
actual_chart = go.Scatter(x=actual_encoded_test_labels.index, y=actual_encoded_test_labels.Labels, name= 'Actual Label')
predict_chart = go.Scatter(x=actual_encoded_test_labels.index, y=predicted_encoded_test_labels.Labels, name= 'Predicted Label')
py.iplot([predict_chart, actual_chart])

"""# Evaluate Model

## Evaluating for Training Dataset
"""

model.evaluate(x=train_dataset, y=train_labels)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
confusion_matrix_train = confusion_matrix(actual_encoded_train_labels, predicted_encoded_train_labels)
confusion_matrix_train

f,ax = plt.subplots(figsize=(16,12))
categories = ['Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Pink', 'Purple', 'Brown', 'Grey', 'Black', 'White']
sns.heatmap(confusion_matrix_train, annot=True, cmap='Blues', fmt='d',
            xticklabels = categories,
            yticklabels = categories)
plt.show()

target_names = ['Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Pink', 'Purple', 'Brown', 'Grey', 'Black', 'White']
print(classification_report(actual_encoded_train_labels, predicted_encoded_train_labels, target_names=target_names))
print(accuracy_score(actual_encoded_train_labels, predicted_encoded_train_labels))
"""## Evaluating for Test Dataset"""

model.evaluate(x=test_dataset, y=test_labels)

confusion_matrix_test = confusion_matrix(actual_encoded_test_labels, predicted_encoded_test_labels)
confusion_matrix_test

f,ax = plt.subplots(figsize=(16,12))
categories = ['Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Pink', 'Purple', 'Brown', 'Grey', 'Black', 'White']
sns.heatmap(confusion_matrix_test, annot=True, cmap='Blues', fmt='d',
            xticklabels = categories,
            yticklabels = categories)
plt.show()

target_names = ['Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Pink', 'Purple', 'Brown', 'Grey', 'Black', 'White']
print(classification_report(actual_encoded_test_labels, predicted_encoded_test_labels, target_names=target_names))
print(accuracy_score(actual_encoded_test_labels, predicted_encoded_test_labels))