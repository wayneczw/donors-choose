import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import random

from argparse import ArgumentParser
from glob import glob
from keras.applications import *
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.regularizers import *
from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle as sk_shuffle
from tqdm import tqdm

WIDTH = 299
N_CLASS = 120


def plot_history(h, path):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.subplot(1, 2, 2)
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.legend(['acc', 'val_acc'])
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.savefig(path)
#end def


def batch_iter(X_featurized, Y_binarized, batch_size=128):
    data_size = Y_binarized.shape[0]
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1

    def data_generator():
        while True:
            # Shuffle the data at each epoch
            shuffled_indices = np.random.permutation(np.arange(data_size, dtype=np.int))

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)

                X_batch = [X_featurized[i] for i in shuffled_indices[start_index:end_index]]
                Y_batch = [Y_binarized[i] for i in shuffled_indices[start_index:end_index]]

                yield ({'input': np.asarray(X_batch)}, {'output': np.asarray(Y_batch)})
            #end for
        #end while
    #end def

    return num_batches_per_epoch, data_generator()
#end def


def build_model(input_shape, output_shape):
    x_input = Input(input_shape, name='input')
    x = Dropout(0.5)(x_input)
    output = Dense(output_shape, activation='softmax', name='output')(x)
    model = Model(inputs=[x_input], outputs=[output])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
#end def


def predict_iter(X_data, batch_size=128):
    data_size = X_data.shape[0]
    num_batches = int((data_size - 1) / batch_size) + 1

    def _data_generator():
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, data_size)
            X_batch = [d for d in X_data[start_index:end_index]]

            yield ({'input': np.asarray(X_batch)})
        #end for
    #end def

    return num_batches, _data_generator()
#end def


def get_features(MODEL, data, width=299):
    cnn_model = MODEL(include_top=False, input_shape=(width, width, 3), weights='imagenet')
    
    x_input = Input((width, width, 3), name='input')
    x = Lambda(preprocess_input, name='preprocessing')(x_input)
    x = cnn_model(x)  # Transfer Learning
    output = GlobalAveragePooling2D(name='output')(x)  # GAP Technique
    model = Model(inputs=[x_input], outputs=[output])

    data_steps, data_batches = predict_iter(data, batch_size=128)
    features = model.predict_generator(generator=data_batches, steps=data_steps)
    return features
#end def


def read(root_dir, stanford=False, seed=0, width=299, n_class=120):
    df = pd.read_csv('./data/labels.csv')
    breed = set(df['breed'])

    class_to_num = dict(zip(breed, range(n_class)))
    num_to_class = dict(zip(range(n_class), breed))

    if stanford:
        image_dir = root_dir[:-1] if root_dir.endswith('/') else root_dir
        n = len(glob(image_dir + '/*/*.jpg'))
        # n = 256  # set to small value for testing only

        X = np.zeros((n, width, width, 3), dtype=np.uint8)
        y = np.zeros((n, n_class), dtype=np.uint8)

        for i, file_name in tqdm(enumerate(glob(image_dir + '/*/*.jpg')), total=n):
            if i == n: break
            y_label = file_name.split('/')[6][10:].lower()
            img = cv2.imread(file_name)
            X[i] = cv2.resize(img, (width, width))
            y[i][class_to_num[y_label]] = 1
        #end for
    else:
        n = len(df)
        # n = 256  # set to small value for testing only

        X = np.zeros((n, width, width, 3), dtype=np.uint8)
        y = np.zeros((n, n_class), dtype=np.uint8)
        for i in range(n):
            img = cv2.imread(root_dir + '/%s.jpg' % df['id'][i])
            X[i] = cv2.resize(img, (width, width))
            y[i][class_to_num[df['breed'][i]]] = 1
        #end for
    #end if

    return X, y, class_to_num, num_to_class, breed
#end def


def plot_sample_figure(X, y, num_to_class, path):
    plt.figure(figsize=(12, 6))
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.imshow(X[i][:, :, ::-1])
        plt.title(num_to_class[y[i].argmax()])
    #end for

    plt.savefig(path)
#end def


def main():
    parser = ArgumentParser(description='Run machine learning experiment.')
    parser.add_argument('-i', '--train_image', type=str, metavar='<train_set_image>', required=True, help='Training data image set.')
    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    parser.add_argument('--stanford', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=True, help='Whether to use stanford data set or not')
    
    A = parser.parse_args()

    # set seed
    np.random.seed(A.seed)
    X_299, y, class_to_num, num_to_class, breed = read(A.train_image, stanford=A.stanford, seed=A.seed, width=WIDTH, n_class=N_CLASS)

    plot_sample_figure(X_299, y, num_to_class, './images/sample.png')

    inception_features = get_features(InceptionResNetV2, data=X_299, width=WIDTH)
    del X_299

    X_331, y, class_to_num, num_to_class, breed = read(A.train_image, stanford=A.stanford, seed=A.seed, width=331, n_class=N_CLASS)
    nas_net_features = get_features(NASNetLarge, data=X_331, width=331)
    del X_331

    features = np.concatenate([inception_features, nas_net_features], axis=-1)
    del inception_features
    del nas_net_features

    train_features, val_features, train_y, val_y = train_test_split(features, y, test_size=0.1)
    
    model = build_model(input_shape=features.shape[1:], output_shape=N_CLASS)
    del features

    train_steps, train_batches = batch_iter(train_features, train_y, batch_size=128)
    val_steps, val_batches = batch_iter(val_features, val_y, batch_size=128)

    # define early stopping callback
    callbacks_list = []
    early_stopping = dict(monitor='val_loss', patience=10, min_delta=0.0005, verbose=1)
    model_checkpoint = dict(filepath='./weights/{val_loss:.5f}_{loss:.5f}_{epoch:04d}.weights.h5',
                            save_best_only=True,
                            save_weights_only=True,
                            mode='auto',
                            period=1,
                            verbose=1)

    earlystop = EarlyStopping(**early_stopping)
    callbacks_list.append(earlystop)

    checkpoint = ModelCheckpoint(**model_checkpoint)
    callbacks_list.append(checkpoint)

    epochs = 32
    history = model.fit_generator(
        epochs=epochs,
        generator=train_batches,
        steps_per_epoch=train_steps,
        validation_data=val_batches,
        validation_steps=val_steps,
        callbacks=callbacks_list
    )

    del train_features
    del val_features
    del train_y
    del val_y
    del train_batches
    del val_batches

    plot_history(h=history, path='./images/train_cost.png')

    df2 = pd.read_csv('./data/sample_submission.csv')
    n_test = len(df2)
    X_299_test = np.zeros((n_test, WIDTH, WIDTH, 3), dtype=np.uint8)
    X_331_test = np.zeros((n_test, 331, 331, 3), dtype=np.uint8)
    for i in tqdm(range(n_test)):
        img = cv2.imread('./data/test/%s.jpg' % df2['id'][i])
        X_299_test[i] = cv2.resize(img, (WIDTH, WIDTH))
        X_331_test[i] = cv2.resize(img, (331, 331))
    #end for
    
    inception_features_test = get_features(InceptionResNetV2, data=X_299_test, width=WIDTH)
    del X_299_test
    nas_net_features_test = get_features(NASNetLarge, data=X_331_test, width=331)
    del X_331_test
    features_test = np.concatenate([inception_features_test, nas_net_features_test], axis=-1)
    del inception_features_test
    del nas_net_features_test
    
    test_steps, test_batches = predict_iter(features_test)
    y_pred = model.predict_generator(generator=test_batches, steps=test_steps)
    for b in breed:
        df2[b] = y_pred[ : ,class_to_num[b]]
    df2.to_csv('./data/pred.csv', index=None)
#end def



if __name__ == '__main__': main()
