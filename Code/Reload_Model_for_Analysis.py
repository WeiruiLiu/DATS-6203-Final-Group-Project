# I. ====================Load Libraries====================================================
import os
import cv2
import keras
import random
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn.metrics
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, InputLayer
from datetime import datetime

# The time of program's execution
start_time = datetime.now()          # Duration: 0:07:10.215383

# II. ====================Load Dataset====================================================
data = pd.read_csv('age_gender.csv')
data.dtypes    # Check the data type of each variable
data.head()    # Show the first 5 row of dataset

# III. ====================Data Preprocessing (for Exploratory Data Analysis)==============
## Convert the type of pixels into np.array
data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(),dtype='float32'))
print('The pixels of each image is', len(data['pixels'][0]))        # The length of pixels of each image: 2304 (48x48)
print('The image width is', int(np.sqrt(len(data['pixels'][0]))))   # 48
print('The image height is', int(np.sqrt(len(data['pixels'][0]))))  # 48

# IV. ==================================Description of dataset===============================
data['Number of Image'] = 1

# Age : 1-116, missing some age
Age_group = data.groupby(by='age')['Number of Image'].sum().reset_index(name='Number of Image')
plt.figure()           # Plot the distribution of Age Group
plt.bar(Age_group['age'],Age_group['Number of Image'])
plt.title('The Distribution of Age Group')
plt.xlabel('Age')
plt.ylabel('Number of Image')
plt.show()

# Ethnicity:
# 0: White, 1:Black, 2:Asian, 3:Indian, 4:Others(like Hispanic,Latino,Middle Eastern)
Ethnicity_group = data.groupby(by='ethnicity')['Number of Image'].sum().reset_index(name='Number of Image')
label_ethnicity = ['White', 'Black', 'Asian', 'Indian', 'Others']
plt.figure()            # Plot the distribution of Ethnicity Group
plt.bar(Ethnicity_group['ethnicity'],Ethnicity_group['Number of Image'])
plt.title('The Distribution of Ethnicity Group')
plt.xlabel('Ethnicity')
plt.ylabel('Number of Image')
plt.xticks(Ethnicity_group['ethnicity'], label_ethnicity)
plt.show()

# Gender
# 0: Male, 1:Female
Gender_group = data.groupby(by='gender')['Number of Image'].sum().reset_index(name='Number of Image')
label_gender = ['Male', 'Female']
plt.figure()            # Plot the distribution of Gender Group
plt.bar(Gender_group['gender'],Gender_group['Number of Image'])
plt.title('The Distribution of Gender Group')
plt.xlabel('Gender')
plt.ylabel('Number of Image')
plt.xticks(Gender_group['gender'],label_gender)
plt.show()

# V. ----Show some sample images randomly------
def sample_images(data):
  fig, axs = plt.subplots(4,4,figsize=(16,16))
  df = data.sample(n=16).reset_index(drop=True)
  axs = axs.ravel()
  j = 0
  for i in range(len(df)):
    if j < 16:
      pixels = df['pixels'][i].reshape(48,48)
      axs[j].imshow(pixels, cmap='gray')
      axs[j].get_xaxis().set_ticks([])
      axs[j].get_yaxis().set_ticks([])
      axs[j].set_xlabel(f"Age: {df['age'].iloc[i]}, Ethnicity: {df['ethnicity'].iloc[i]}, Gender: {'F' if df['gender'].iloc[i]==1 else 'M'}", fontsize=14)
      j += 1
    else:
      break
  fig.suptitle('Sample Images',fontsize=40)
  plt.show()

# Plot 16 images with label of age, ethnicity, and gender randomly
sample_images(data)



# VI. ================================ Two Models ==================================================
# Setting up
SEED = 41
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --------------------------------- 1. Age Predict Model--------------------------------------------
# Data Preparation
Image = np.array(data['pixels'].tolist())
Image = Image.reshape(Image.shape[0],48,48,1)      # Reshapes to (n_examples, n_channels, height_pixels, width_pixels)
Age_target = np.array(data['age']).reshape(-1,1)
Image = Image/255                                  # Normalizing pixels data
print('Input Image shape:',Image.shape)            # (23705, 48, 48, 1)
print('Age Target shape:',Age_target.shape)        # (23705, 1)


# Split Data into training set (70%), validation set (15%), and testing set (15%)
age_x_train, age_x_test, age_y_train, age_y_test = train_test_split(Image, Age_target, test_size=0.3, random_state=SEED, shuffle=True)  # stratify=Age_target (ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.)
age_x_val, age_x_test, age_y_val, age_y_test = train_test_split(age_x_test, age_y_test, test_size=0.5, random_state=SEED, shuffle=True)
print('Age train samples:',age_x_train.shape[0])       # 16593
print('Age validation samples:',age_x_val.shape[0])    # 3556
print()

#  --------------------------------- 2. Ethnicity & Gender Predict Model----------------------------
# Data Preparation
Ethnicity_target = np.array(data['ethnicity']).reshape(-1,1)
Gender_target = np.array(data['gender']).reshape(-1,1)
Ethnicity_class = keras.utils.to_categorical(Ethnicity_target,5)
Gender_class = keras.utils.to_categorical(Gender_target,2)
# Concatenate Ethnicity and Gender target in one-hot-encoded (eg. White:0,Female:1 is [1,0,0,0,0,0,1])
Ethnicity_Gender_target = np.concatenate((Ethnicity_class,Gender_class),axis=1)     # one-hot-encoded
print('Input Image shape:',Image.shape)                                             # (23705, 48, 48, 1)
print('Ethnicity & Gender Target shape:',Ethnicity_Gender_target.shape)             # (23705, 7)


# Split Data into training set (70%) and validation set (15%) testing set (15%)
eg_x_train, eg_x_test, eg_y_train, eg_y_test = train_test_split(Image, Ethnicity_Gender_target, test_size=0.3, random_state=SEED, stratify=Ethnicity_Gender_target) # stratify=Age_target (ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.)
eg_x_val, eg_x_test, eg_y_val, eg_y_test = train_test_split(eg_x_test, eg_y_test, test_size=0.5, random_state=SEED, stratify=eg_y_test)
print('Ethnicity & Gender train sample:',eg_x_train.shape[0])          # 16593
print('Ethnicity & Gender validation sample:',eg_x_val.shape[0])       # 3556


# VIII. ========================== Reload Saved Two Models =============================

restore_model_age = load_model('age_model_best.hdf5')
restore_model_eg = load_model('Eg_model_best1.hdf5')
print("Best MAE of Age:", restore_model_age.evaluate(age_x_test, age_y_test)[1])
print("Best Accuracy of Ethnicity & Gender:", 100*restore_model_eg.evaluate(eg_x_test, eg_y_test)[1], "%")

# ------------ Predict age, ethnicity, and gender of all images-----------------------
age_prediction = np.round(restore_model_age.predict(Image)).astype(int)
ethnicity_gender_prediction = np.round(restore_model_eg.predict(Image))
ethnicity_prediction = ethnicity_gender_prediction[:,:-2].argmax(axis=1)
gender_prediction = ethnicity_gender_prediction[:,-2:].argmax(axis=1)

data_pred = data.copy()   # Create a DataFrame contain the prediction and predict error
data_pred['age pred'] = age_prediction
data_pred['ethnicity pred'] = ethnicity_prediction
data_pred['gender pred'] = gender_prediction
data_pred['age error'] = data_pred['age']-data_pred['age pred']
data_pred['ethnicity error'] = data_pred['ethnicity']-data_pred['ethnicity pred']
data_pred['gender error'] = data_pred['gender']-data_pred['gender pred']

# Plot some images with real label and predict label randomly
def predict_images(data):
  fig, axs = plt.subplots(4,4,figsize=(16,16))
  df = data_pred.sample(n=16).reset_index(drop=True)
  axs = axs.ravel()
  j = 0
  for i in range(len(df)):
    if j < 16:
      pixels = df['pixels'][i].reshape(48,48)
      axs[j].imshow(pixels, cmap='gray')
      axs[j].get_xaxis().set_ticks([])
      axs[j].get_yaxis().set_ticks([])
      axs[j].set_xlabel('[Real]: Age:'+str(df['age'].iloc[i])+
                        ' Ethnicity:'+str(df['ethnicity'].iloc[i])+
                        ' Gender:'+str(df['gender'].iloc[i])+
                        '\n[Pred]: Age:'+str(df['age pred'].iloc[i])+
                        ' Ethnicity:'+str(df['ethnicity pred'].iloc[i])+
                        ' Gender:'+str(df['gender pred'].iloc[i]))
      j += 1
    else:
      break
  fig.suptitle('Sample Prediction Images',fontsize=40)
  plt.show()

predict_images(data_pred)


# IX. =====================Prediction Accuracy Analysis==================================

# --------------------------------1. Age------------ --------
Age_pred_group = data_pred[data_pred['age']==data_pred['age pred']]
Age_correct_group = Age_pred_group.groupby(by='age')['Number of Image'].sum().reset_index(name='Sum of Correct Image')
age_result = pd.merge(Age_group, Age_correct_group, on='age')
age_result['Accuracy'] = 100*age_result['Sum of Correct Image']/age_result['Number of Image']
conditions = [(age_result['age'] <= 10),
              (age_result['age'] <= 20) & (age_result['age'] > 10),
              (age_result['age'] <= 30) & (age_result['age'] > 20),
              (age_result['age'] <= 40) & (age_result['age'] > 30),
              (age_result['age'] <= 50) & (age_result['age'] > 40),
              (age_result['age'] <= 60) & (age_result['age'] > 50),
              (age_result['age'] <= 70) & (age_result['age'] > 60),
              (age_result['age'] <= 80) & (age_result['age'] > 70),
              (age_result['age'] <= 90) & (age_result['age'] > 80),
              (age_result['age'] <= 100) & (age_result['age'] > 90),
              (age_result['age'] > 100)]
values = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60','61-70','71-80','81-90','91-100','100+']
age_result['Age Groups'] = np.select(conditions, values)
Age_Group_avg = age_result.groupby(by='Age Groups')['Accuracy'].mean().reset_index(name='Average of Accuracy')

# Plot the Distribution of Correctly Prediction of Whole Age Group
plt.figure()
plt.bar(Age_Group_avg['Age Groups'],Age_Group_avg['Average of Accuracy'],color=['palevioletred'])            # change x to age_result['age'] could see the distribution of correctly prediction of age
plt.title('The Distribution of Correctly Prediction of Whole Age Group')
plt.xlabel('Age Groups')
plt.ylabel('Percentage of Correct Prediction Image (%)')
for a,b in zip(Age_Group_avg['Age Groups'],Age_Group_avg['Average of Accuracy']):
    plt.text(a, b+0.05, '%.3f' % b, ha='center', va= 'bottom',fontsize=7)
plt.show()

# Histogram of prediction error of age
plt.figure()
plt.hist(data_pred['age error'],bins=40)    # age_result['age'],age_result['Accuracy']
plt.title('The Histogram of Prediction Error of Age')
plt.xlabel('Error')
plt.ylabel('Number of Images')
plt.xlim(-20,20)
plt.show()


# -------------------- 2. Ethnicity----------------------------------
Ethnicity_pred_group = data_pred[data_pred['ethnicity']==data_pred['ethnicity pred']]
Ethnicity_correct_group = Ethnicity_pred_group.groupby(by='ethnicity')['Number of Image'].sum().reset_index(name='Sum of Correct Image')
ethnicity_result = pd.merge(Ethnicity_group, Ethnicity_correct_group, on='ethnicity')
ethnicity_result['Accuracy'] = 100*ethnicity_result['Sum of Correct Image']/ethnicity_result['Number of Image']

# Plot the Distribution of Correctly Prediction of Ethnicity Group
label_ethnicity = ['White', 'Black', 'Asian', 'Indian', 'Others']
plt.figure()
plt.bar(ethnicity_result['ethnicity'],ethnicity_result['Accuracy'],color = ['lightskyblue'])
plt.title('The Distribution of Correctly Prediction of Ethnicity Group')
plt.xlabel('Ethnicity')
plt.ylabel('Percentage of Correct Prediction Image (%)')
plt.xticks(ethnicity_result['ethnicity'], label_ethnicity)
for a,b in zip(ethnicity_result['ethnicity'],ethnicity_result['Accuracy']):
    plt.text(a, b+0.05, '%.3f' % b, ha='center', va= 'bottom',fontsize=7)
plt.show()

# Plot Confusion Matrix of Ethnicity
cm = sklearn.metrics.confusion_matrix(data_pred['ethnicity'],data_pred['ethnicity pred'])
df_cm = pd.DataFrame(cm, range(5), range(5))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)         # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},cmap="Blues")    # font size
plt.xlabel("Predicted Ethnicity")
plt.ylabel("True Ethnicity")
plt.title('Confusion Matrix of Ethnicity')
plt.show()

# -------------------------3. Gender -----------------------------------
Gender_pred_group = data_pred[data_pred['gender']==data_pred['gender pred']]
Gender_correct_group = Gender_pred_group.groupby(by='gender')['Number of Image'].sum().reset_index(name='Sum of Correct Image')
gender_result = pd.merge(Gender_group, Gender_correct_group, on='gender')
gender_result['Accuracy'] = 100*gender_result['Sum of Correct Image']/gender_result['Number of Image']

# Plot the Distribution of Correctly Prediction of Gender Group
label_gender = ['Male', 'Female']
plt.figure(figsize=(10,7))
plt.bar(gender_result['gender'],gender_result['Accuracy'],color = ['lightblue','pink'])
plt.title('The Distribution of Correctly Prediction of Gender Group')
plt.xlabel('Gender')
plt.ylabel('Percentage of Correct Prediction Image (%)')
plt.xticks(gender_result['gender'],label_gender)
for a,b in zip(gender_result['gender'],gender_result['Accuracy']):
    plt.text(a, b+0.05, '%.3f' % b, ha='center', va= 'bottom',fontsize=7)
plt.show()

# Plot Confusion Matrix of Gender
cm = sklearn.metrics.confusion_matrix(data_pred['gender'],data_pred['gender pred'])
df_cm = pd.DataFrame(cm, range(2), range(2))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)         # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},cmap="Blues")    # font size
plt.xlabel("Predicted Gender")
plt.ylabel("True Gender")
plt.title('Confusion Matrix of Gender')
plt.show()


# X.================================ Error Analysis ==============================

# Show Incorrectly Age Images
def Incorrectly_images_age(x_test,y_test):
    fig, axs = plt.subplots(4,4,figsize=(16,16))
    age_y_pred = np.round(restore_model_age.predict(x_test))
    axs = axs.ravel()
    x_test = x_test * 255
    j = 0
    for i in range(len(x_test)):
        if j < 16:
            if age_y_pred[i] != y_test[i]:
                pixels = x_test[i].reshape((48,48))
                axs[j].imshow(pixels, cmap='gray')
                axs[j].get_xaxis().set_ticks([])
                axs[j].get_yaxis().set_ticks([])
                axs[j].set_xlabel('Real Age:'+str(y_test[i])+
                                  ' Pred Age:'+str(age_y_pred[i]), fontsize=14)
                j += 1
        else:
            break
    fig.suptitle('Incorrectly Age Images',fontsize=40)
    plt.show()

Incorrectly_images_age(age_x_test,age_y_test)

# Show Incorrectly Ethnicity and Gender Images
eg_y_pred = restore_model_eg.predict(eg_x_test)
eg_y_pred = np.round(eg_y_pred)
ethnicity_pred = eg_y_pred[:,:-2].argmax(axis=1)
gender_pred = eg_y_pred[:,-2:].argmax(axis=1)

ethnicity_y_test = eg_y_test[:,:-2].argmax(axis=1)
gender_y_test = eg_y_test[:,-2:].argmax(axis=1)

def Incorrectly_images_eg(x_test,y_test):
    fig, axs = plt.subplots(4,4,figsize=(16,16))
    axs = axs.ravel()
    x_test = x_test * 255
    j = 0
    for i in range(len(x_test)):
        if j < 16:
            if ethnicity_pred[i] != ethnicity_y_test[i] and gender_pred[i] != gender_y_test[i]:
                pixels = x_test[i].reshape((48,48))
                axs[j].imshow(pixels, cmap='gray')
                axs[j].get_xaxis().set_ticks([])
                axs[j].get_yaxis().set_ticks([])
                axs[j].set_xlabel('[Real] Ethnicity:'+str(ethnicity_y_test[i])+
                                  ' Gender:'+str(gender_y_test[i])+
                                  '\n [Pred] Ethnicity:'+str(ethnicity_pred[i])+
                                  ' Gender:'+str(gender_pred[i]), fontsize=14)
                j += 1
        else:
            break
    fig.suptitle('Incorrectly Ethnicity & Gender Images',fontsize=40)
    plt.show()

Incorrectly_images_eg(eg_x_test,eg_y_test)


# XI.================================ Application ======================
# Predict the age,ethnicity, and gender of the image which randomly choose from Google
# Load the Test_Images Folder
direction = os.getcwd() + "/Test_Images/"        # get the image path
image_path = []
for im in [f for f in os.listdir(direction)]:
    image_path.append(direction+im)

# Reshape the Images
RESIZE_TO = 48
x = []
pixels = []
for png in image_path:
    pixel = cv2.imread(png)
    pixels.append(pixel)
    image = cv2.resize(pixel, (RESIZE_TO, RESIZE_TO))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x.append(image)
x = np.array(x)
x = x.reshape(len(x), RESIZE_TO, RESIZE_TO, 1)
x = x / 255

# Age Prediction
age_pred_apply = np.round(restore_model_age.predict(x))

# Ethnicity & Gender Prediction
ethnicity_gender_pred_apply = np.round(restore_model_eg.predict(x))
ethnicity_pred_apply = ethnicity_gender_pred_apply[:,:-2].argmax(axis=1)
gender_pred_apply = ethnicity_gender_pred_apply[:,-2:].argmax(axis=1)

ethnicity_pred_label = []
for i in range(len(ethnicity_pred_apply)):
    if ethnicity_prediction[i] == 0:
        ethnicity_pred_label.append('White')
    elif ethnicity_prediction[i] == 1:
        ethnicity_pred_label.append('Black')
    elif ethnicity_prediction[i] == 2:
        ethnicity_pred_label.append('Asian')
    elif ethnicity_prediction[i] == 3:
        ethnicity_pred_label.append('Indian')
    else:
        ethnicity_pred_label.append('Other')

gender_pred_label = []
for i in range(len(gender_pred_apply)):
    if gender_prediction[i] == 0:
        gender_pred_label.append('M')
    else:
        gender_pred_label.append('F')

# Plot the Prediction
fig, axs = plt.subplots(3,2,figsize=(16,16))
axs = axs.ravel()
for i in range(len(image_path)):
    image = pixels[i]
    axs[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[i].get_xaxis().set_ticks([])
    axs[i].get_yaxis().set_ticks([])
    axs[i].set_xlabel('[Pred]: Age:'+str(age_pred_apply[i])+
                    ' Ethnicity:'+str([ethnicity_pred_label[i]])+
                    ' Gender:'+str([gender_pred_label[i]]), fontsize=18)
fig.suptitle('Prediction Images from Google Image',fontsize=40)
plt.show()

# End time
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


