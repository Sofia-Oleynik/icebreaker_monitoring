import numpy as np 
np.random.seed(666) 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, 
Dropout 
from tensorflow.keras.layers import Flatten, Dense 
from tensorflow.keras.models import Model, load_model 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, 
EarlyStopping 
import matplotlib.pyplot as plt 
from scipy.ndimage.filters import uniform_filter 
from scipy.ndimage.measurements import variance 
data = pd.read_json(r'~\Iceberg-Ship\Classification-master\Models\train.json') 
print(len(data)) 
print(round(np.sum(data.is_iceberg)/len(data)*100,3)) 
data.inc_angle = data.inc_angle.replace('na', np.nan) 
data = data.dropna(axis=0, how='any') 
print(len(data)) 
print(round(np.sum(data.is_iceberg)/len(data)*100,3)) 
def normalize(sig_nought, inc_angle): 
    """ 
    :type sig_nought: np.ndarray(np.float) 
    :type inc_angle: float 
    """ 

    sig_nought_n = (sig_nought + 0.766 * inc_angle - 31.638) / 2 
 
    return sig_nought_n 
 
def lee_filter(img, size): 
    img_mean = uniform_filter(img, (size, size)) 
    img_sqr_mean = uniform_filter(img**2, (size, size)) 
    img_variance = img_sqr_mean - img_mean**2 
 
    overall_variance = variance(img) 
 
    img_weights = img_variance / (img_variance + overall_variance) 
    img_output = img_mean + img_weights * (img - img_mean) 
    return img_output 
 
 
def normalize_data(hh, hv, inc_angle): 
    # normalize hh and hv using inc_angle (x_angle[i]) 
    hh = normalize(hh, inc_angle) 
    hv = normalize(hv, inc_angle) 
 
    # apply Lee filter for all bands 
    hh = lee_filter(hh, 20) 
    hv = lee_filter(hv, 20) 
 
    # total backscatter = hh + hv 
    b3 = hh + hv 
    # cross polarisation ratio = hv / (hh+hv) 
    c3 = hv / 2 * (hh + hv) 

    # Rescale images between 0 and 1 for faster convergence rate 
    hh = (hh - hh.min()) / (hh.max() - hh.min()) 
    hv = (hv - hv.min()) / (hv.max() - hv.min()) 
    b3 = (b3 - b3.min()) / (b3.max() - b3.min()) 
    c3 = (c3 - np.nanmin(c3)) / (np.nanmax(c3) - np.nanmin(c3)) 
    return hh, hv, b3, c3 
 
 
def prepare_data(data): 
    x_angle = np.array(data["inc_angle"]) 
 
    imgs = [] 
    labels = [] 
 
    for i, row in data.iterrows(): 
        if not np.isnan(row["inc_angle"]): 
            labels.append(row["is_iceberg"]) 
            # Reshape list to image 
            hh = np.reshape(row["band_1"], (75, 75)) 
            hv = np.reshape(row["band_2"], (75, 75)) 
 
            hh, hv, b3, c3 = normalize_data(hh, hv, row["inc_angle"]) 
 
            # Объединение диапазонов в изображения 
            imgs.append(np.dstack((hh, hv, b3, c3))) 
 
    labels = np.array(labels) 

    # Разбитие датасета на обучение (70%) and проверку (30 %) 
    x_train, x_val, y_train, y_val = train_test_split(imgs, labels, 
                                                      test_size=0.3, 
                                                      random_state=0) 
    # Then split validation dataset into validation (20 %) and testing (10 %) 
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, 
                                                    test_size=(1 / 3), 
                                                    random_state=0) 
    x_train = np.array(x_train) 
    x_test = np.array(x_test) 
    x_val = np.array(x_val) 
 
    return x_train, x_val, x_test, y_train, y_val, y_test 
 
x_train, x_val, x_test, y_train, y_val, y_test = prepare_data(data) 
print("Number of samples for training: %i (%.2f%%)"%(len(x_train), 
round(len(x_train)/len(data)*100))) 
print("Number of samples for validation: %i (%.2f%%)"%(len(x_val), 
round(len(x_val)/len(data)*100))) 
print("Number of samples for testing: %i (%.2f%%)"%(len(x_test), 
round(len(x_test)/len(data)*100))) 
 
batch_size = 32 
gen = ImageDataGenerator( 
    rotation_range = 90, 
    width_shift_range = 0.2, 
    height_shift_range = 0.2, 
    shear_range = 0.2, 
    zoom_range = 0.2, 
    horizontal_flip = True, 
    fill_mode = 'nearest') 
 
gen_flow = gen.flow(x_train, y_train, batch_size = batch_size, seed = 
666) 
 
 
def create_model(optimizer): 
    input_img = Input(shape=(75, 75, 4), name="X_img") 
 
    cnn = Conv2D(16, kernel_size=(3, 3), activation="relu")(input_img) 
    cnn = MaxPooling2D((2, 2))(cnn) 
    cnn = Dropout(0.1)(cnn) 
 
    cnn = Conv2D(32, kernel_size=(3, 3), activation="relu")(cnn) 
    cnn = MaxPooling2D((2, 2))(cnn) 
    cnn = Dropout(0.1)(cnn) 
 
    cnn = Conv2D(64, kernel_size=(3, 3), activation="relu")(cnn) 
    cnn = MaxPooling2D((2, 2))(cnn) 
    cnn = Dropout(0.1)(cnn) 
 
    cnn = Conv2D(128, kernel_size=(3, 3), activation="relu")(cnn) 
    cnn = Conv2D(128, kernel_size=(3, 3), activation="relu")(cnn) 
    cnn = MaxPooling2D((2, 2))(cnn) 
    cnn = Dropout(0.1)(cnn) 
 
    cnn = Flatten()(cnn) 
150 
 
 
    dense = Dense(64, activation="relu")(cnn) 
    dense = Dense(64, activation="relu")(dense) 
    dense = Dense(64, activation="relu")(dense) 
 
    output = Dense(1, activation="sigmoid")(dense) 
 
    model = Model(input_img, output) 
 
    model.compile(loss="binary_crossentropy", optimizer=optimizer, 
metrics=["accuracy"]) 
    return model 
 
 
model = create_model(optimizer=Adam()) 
model.summary() 
 
def get_callbacks(patience=5): 
    es = EarlyStopping('val_loss', patience=patience, mode="auto", 
restore_best_weights=True) 
    return es 
 
callbacks = get_callbacks(patience=50) 
 
epochs = 100 
 
history = model.fit(gen_flow, 
                    validation_data=(x_val, y_val), 
                    steps_per_epoch=len(x_train) / batch_size, 
                    epochs=epochs, 
                    callbacks=callbacks) 
 
def plot_graphs(history, string): 
  plt.plot(history.history[string]) 
  plt.plot(history.history['val_'+string]) 
  plt.xlabel("Epochs") 
  plt.ylabel(string) 
  plt.legend([string, 'val_'+string], loc = 'best') 
  plt.ylim([0,1]) 
  plt.grid() 
  plt.show() 
 
plot_graphs(history, "accuracy") 
plot_graphs(history, "loss") 
 
results = model.evaluate(x_test, y_test) 
 
model.save("~\model_100epcs.hd5")