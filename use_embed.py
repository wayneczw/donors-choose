import calendar
import gc
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import regex as re
import tensorflow as tf
import tensorflow_hub as hub
import warnings

from argparse import ArgumentParser

from category_encoders.target_encoder import TargetEncoder

from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.engine import Layer
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import GRU
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Reshape
from keras.layers import concatenate
from keras.models import Model

from nltk import sent_tokenize
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight


logger = logging.getLogger(__name__)


target_feature = ['project_is_approved']

raw_categorical_features = [
    'month_year', 'dayofweek',
    'teacher_prefix', 'school_state',
    'project_grade_category',
    'project_subject_categories', 'project_subject_subcategories']

raw_continuous_features = [
    'teacher_number_of_previously_posted_projects',
    'total_price', 'quantity']

old_raw_string_features = [
    'project_title', 'project_essay_1',
    'project_essay_2', 'project_essay_3',
    'project_essay_4', 'project_resource_summary',
    'description']

new_raw_string_features = [
    'project_title', 'project_essay_1',
    'project_essay_2', 'project_resource_summary',
    'description']

categorical_features = [
    'month_year', 'dayofweek',
    'teacher_prefix', 'school_state',
    'project_grade_category',
    'project_subject_categories', 'project_subject_subcategories']

old_continuous_features = [
    'teacher_number_of_previously_posted_projects',
    'total_price', 'quantity', 'polarity', 'subjectivity',
    'project_essay_1_len', 'project_essay_2_len',
    'project_essay_3_len', 'project_essay_4_len']

new_continuous_features = [
    'teacher_number_of_previously_posted_projects',
    'total_price', 'quantity', 'polarity', 'subjectivity',
    'project_essay_1_len', 'project_essay_2_len']

old_string_features = [
    'project_title', 'project_essay_1',
    'project_essay_2', 'project_essay_3',
    'project_essay_4', 'project_resource_summary', 'description']

new_string_features = [
    'project_title', 'project_essay_1',
    'project_essay_2', 'project_resource_summary', 'description']

USE_MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/2"
USE_EMBED = hub.Module(USE_MODULE_URL, trainable=True)
# ELMO_MODULE_URL = "https://tfhub.dev/google/elmo/2"
# ELMO_EMBED = hub.Module(ELMO_MODULE_URL, trainable=True)


def USE_Embedding(x):
    return USE_EMBED(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
#end def


# def ELMO_Embedding(x):
#     return ELMO_EMBED(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
# #end def


def print_header(text, width=30, char='='):
    print('\n' + char * width)
    print(text)
    print(char * width)
#end def


def display_null_percentage(data):
    print_header('Percentage of Nulls')
    df = data.isnull().sum().reset_index().rename(columns={0: 'Count', 'index': 'Column'})
    df['Frequency'] = df['Count'] / data.shape[0] * 100
    pd.options.display.float_format = '{:.2f}%'.format
    print(df)
    pd.options.display.float_format = None
#end def


def display_category_counts(data, categorical_features):
    print_header('Category Counts for Categorical Features')
    for categorical_feature in categorical_features:
        print('-' * 30)
        print(categorical_feature)
        print(data[categorical_feature].value_counts(dropna=False))
#end def


def analyse(df, categorical_features, target_feature, continuous_features):
    print(df.info())
    print('='*100)
    # <class 'pandas.core.frame.DataFrame'>
    # Int64Index: 182080 entries, 0 to 182079
    # Data columns (total 17 columns):
    # teacher_prefix                                  182080 non-null category
    # school_state                                    182080 non-null category
    # project_grade_category                          182080 non-null category
    # project_subject_categories                      182080 non-null category
    # project_subject_subcategories                   182080 non-null category
    # project_title                                   182080 non-null object
    # project_resource_summary                        182080 non-null object
    # teacher_number_of_previously_posted_projects    182080 non-null int64
    # project_is_approved                             182080 non-null category
    # description                                     182080 non-null object
    # total_price                                     182080 non-null float64
    # year                                            182080 non-null category
    # month                                           182080 non-null category
    # hour                                            182080 non-null category
    # dayofweek                                       182080 non-null category
    # project_essays                                  182080 non-null object
    # dtypes: category(10), float64(1), int64(1), object(5)
    # memory usage: 13.1+ MB
    # None

    print(df.head())
    print('='*100)
    #   teacher_prefix school_state  ... dayofweek                                     project_essays
    # 0            Ms.           NV  ...         4  Most of my kindergarten students come from low...
    # 1           Mrs.           GA  ...         2  Our elementary school is a culturally rich sch...
    # 2            Ms.           UT  ...         6  Hello;\r\nMy name is Mrs. Brotherton. I teach ...
    # 3            Mr.           NC  ...         4  My students are the greatest students but are ...
    # 4            Mr.           CA  ...         5  My students are athletes and students who are ...

    # [5 rows x 16 columns]

    print(df.describe())
    print('='*100)
    #        teacher_number_of_previously_posted_projects    total_price
    # count                                 182080.000000  182080.000000
    # mean                                      11.237055     545.748958
    # std                                       28.016086     548.198713
    # min                                        0.000000     100.000000
    # 25%                                        0.000000     245.997500
    # 50%                                        2.000000     397.750000
    # 75%                                        9.000000     691.920000
    # max                                      451.000000   15299.690000

    print(df.nunique())
    print('='*100)
    # teacher_prefix                                       5
    # school_state                                        51
    # project_grade_category                               4
    # project_subject_categories                          51
    # project_subject_subcategories                      407
    # project_title                                   164282
    # project_resource_summary                        179730
    # teacher_number_of_previously_posted_projects       401
    # project_is_approved                                  2
    # description                                     138458
    # total_price                                      93689
    # year                                                 2
    # month                                               12
    # hour                                                24
    # dayofweek                                            7
    # project_essays                                  181402
    # dtype: int64

    display_null_percentage(df)  # No missing hahaha :) :) :)
    print('='*100)
    # ==============================
    # Percentage of Nulls
    # ==============================
    #                                           Column  Count  Frequency
    # 0                                 teacher_prefix      0      0.00%
    # 1                                   school_state      0      0.00%
    # 2                         project_grade_category      0      0.00%
    # 3                     project_subject_categories      0      0.00%
    # 4                  project_subject_subcategories      0      0.00%
    # 5                                  project_title      0      0.00%
    # 6                       project_resource_summary      0      0.00%
    # 7   teacher_number_of_previously_posted_projects      0      0.00%
    # 8                            project_is_approved      0      0.00%
    # 9                                   description      0      0.00%
    # 10                                   total_price      0      0.00%
    # 11                                          year      0      0.00%
    # 12                                         month      0      0.00%
    # 13                                          hour      0      0.00%
    # 14                                     dayofweek      0      0.00%
    # 15                                project_essays      0      0.00%

    display_category_counts(data=df, categorical_features=categorical_features)
    # Many repeated project titles/essays
    # Can consider reducing dimension of subject {categories, subcategories}
    display_category_counts(data=df, categorical_features=target_feature)
#end def


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


def read(resource_df_path, df_path, old=False, quick=False):
    def _month_year_to_str(month_num, year_num):
        return calendar.month_abbr[month_num] + '_' + str(year_num)
    #end def

    def _dayofweek_to_str(day_num):
        return calendar.day_name[day_num]
    #end def

    def get_time_features(_df):
        time_df = pd.to_datetime(_df['project_submitted_datetime'])
        _df['month_year'] = time_df.apply(lambda x: _month_year_to_str(x.month, x.year))
        _df['dayofweek'] = time_df.apply(lambda x: _dayofweek_to_str(x.dayofweek))
        return _df
    #end def

    def get_polarity(text):
        textblob = TextBlob(text)
        pol = textblob.sentiment.polarity
        return round(pol, 2)
    #end def

    def get_subjectivity(text):
        textblob = TextBlob(text)
        subj = textblob.sentiment.subjectivity
        return round(subj, 2)
    #end def

    def _count_sent(text):
        try:
            return len(sent_tokenize(text))
        except:
            return 0
    #end def

    def get_essay_length(_df, essay_cols):
        for f in essay_cols:
            _df[f + '_len'] = _df[f].apply(_count_sent)
        return _df
    #end def

    def clean(text):
        return re.sub('[!@#$:]', '', ' '.join(re.findall('\w{3,}', str(text).lower())))
    #end def

    logger.info("Reading in data from {}....".format(resource_df_path))

    resource_df = pd.read_csv(resource_df_path)
    resource_df['id'] = resource_df['id'].astype('category')
    resource_df[['price', 'quantity']] = resource_df[['price', 'quantity']].apply(pd.to_numeric)
    resource_df['description'] = resource_df['description'].astype(str)
    
    resource_df['total_price'] = resource_df['price'] * resource_df['quantity']

    resource_df_des = resource_df.groupby('id')['description'].apply('\n'.join).reset_index()
    resource_df_price = resource_df.groupby('id')['total_price'].sum()
    resource_df_qty = resource_df.groupby('id')['quantity'].sum()    
    resource_df = pd.merge(resource_df_des, resource_df_price, on='id', how='left')
    resource_df = pd.merge(resource_df, resource_df_qty, on='id', how='left')

    logger.info("Reading in data from {}....".format(df_path))

    df = pd.read_csv(df_path)
    df = pd.merge(df, resource_df, on='id', how='left')
    df = get_time_features(df)
    df = df.drop(['teacher_id', 'project_submitted_datetime'], axis=1)

    if quick: df = df[:1024]

    df[raw_categorical_features] = df[raw_categorical_features].apply(lambda x: x.astype('category'))
    df[raw_continuous_features] = df[raw_continuous_features].apply(pd.to_numeric)
    try: df['project_is_approved'] = df['project_is_approved'].astype('category')
    except KeyError: pass

    if old:
        df = df[~df['project_essay_3'].isnull()]
        df[old_raw_string_features] = df[old_raw_string_features].fillna('').apply(lambda x: x.astype(str))
        # get length of essays
        df = get_essay_length(df, ['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4'])

        for f in old_raw_string_features:
            df[f] = df[f].apply(lambda x: clean(x))
        df['all_essays'] = df['project_essay_1'].str.cat(df[['project_essay_2', 'project_essay_3', 'project_essay_4']], sep='. ', na_rep='nan')
        df['polarity'] = df['all_essays'].apply(get_polarity)
        df['subjectivity'] = df['all_essays'].apply(get_subjectivity)
        df = df.drop(['all_essays'], axis=1)
        df[old_continuous_features] = df[old_continuous_features].apply(pd.to_numeric)
    else:
        df = df[df['project_essay_3'].isnull()].drop(['project_essay_3', 'project_essay_4'], axis=1)
        df[new_raw_string_features] = df[new_raw_string_features].fillna('').apply(lambda x: x.astype(str))
        # get length of essays
        df = get_essay_length(df, ['project_essay_1', 'project_essay_2'])

        for f in new_raw_string_features:
            df[f] = df[f].apply(lambda x: clean(x))
        df['all_essays'] = df['project_essay_1'].str.cat(df['project_essay_2'], sep='. ', na_rep='nan')
        df['all_essays'] = df['project_essay_1'].str.cat(df[['project_essay_2']], sep='. ', na_rep='nan')
        df['polarity'] = df['all_essays'].apply(get_polarity)
        df['subjectivity'] = df['all_essays'].apply(get_subjectivity)
        df = df.drop(['all_essays'], axis=1)
        df[new_continuous_features] = df[new_continuous_features].apply(pd.to_numeric)
    #end if

    logger.info("Done processing in {} data....".format(df.shape[0]))

    return df
#end def


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale
#end def


class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)
    #end def

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
    #end def

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs
    #end def

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
    #end def
#end class


def auc(y_true, y_pred):
    def _binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
        # PFA, prob false alert for binary classifier
        y_pred = K.cast(y_pred >= threshold, 'float32')
        # N = total number of negative labels
        N = K.sum(1 - y_true)
        # FP = total number of false alerts, alerts from the negative class labels
        FP = K.sum(y_pred - y_pred * y_true)    
        return FP/N
    #end def

    def _binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
        # P_TA prob true alerts for binary classifier
        y_pred = K.cast(y_pred >= threshold, 'float32')
        # P = total number of positive labels
        P = K.sum(y_true)
        # TP = total number of correct alerts, alerts from the positive class labels
        TP = K.sum(y_pred * y_true)    
        return TP/P
    #end def

    # AUC
    ptas = tf.stack([_binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([_binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1, )), pfas], axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes

    return K.sum(s, axis=0)
#end def


def build_model(cat_input_shape, cont_input_shape, output_shape, dropout_rate=0, kernel_regularizer=0, activity_regularizer=0, bias_regularizer=0, old=True):
    
    cat_input = Input(cat_input_shape, name='cat_input')
    cat = Dense(
        16,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(cat_input)  # down size the learnt representation
    cat = Dropout(dropout_rate)(cat)

    cont_input = Input(cont_input_shape, name='cont_input')
    cont = Dense(
        16,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(cont_input)  # down size the learnt representation
    cont = Dropout(dropout_rate)(cont)

    title_input = Input(shape=(1,), dtype=tf.string, name='title_input')
    title = Lambda(USE_Embedding, output_shape=(512,))(title_input)
    # title = Reshape((16, 32), input_shape=(512,))(title)
    # title = Bidirectional(GRU(gru_len, activation='relu', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True))(title)
    # title = Capsule(num_capsule=10, dim_capsule=16, routings=5, share_weights=True)(title)
    # title = Flatten()(title)
    title = Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(title)
    title = Dropout(dropout_rate)(title)

    essay1_input = Input(shape=(1,), dtype=tf.string, name='essay1_input')
    essay1 = Lambda(USE_Embedding, output_shape=(512,))(essay1_input)
    # essay1 = Reshape((16, 32), input_shape=(512,))(essay1)
    # essay1 = Bidirectional(GRU(gru_len, activation='relu', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True))(essay1)
    # essay1 = Capsule(num_capsule=10, dim_capsule=16, routings=5, share_weights=True)(essay1)
    # essay1 = Flatten()(essay1)
    essay1 = Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(essay1)
    essay1 = Dropout(dropout_rate)(essay1)

    essay2_input = Input(shape=(1,), dtype=tf.string, name='essay2_input')
    essay2 = Lambda(USE_Embedding, output_shape=(512,))(essay2_input)
    # essay2 = Reshape((16, 32), input_shape=(512,))(essay2)
    # essay2 = Bidirectional(GRU(gru_len, activation='relu', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True))(essay2)
    # essay2 = Capsule(num_capsule=10, dim_capsule=16, routings=5, share_weights=True)(essay2)
    # essay2 = Flatten()(essay2)
    essay2 = Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(essay2)
    essay2 = Dropout(dropout_rate)(essay2)

    if old:
        essay3_input = Input(shape=(1,), dtype=tf.string, name='essay3_input')
        essay3 = Lambda(USE_Embedding, output_shape=(512,))(essay3_input)
        # essay3 = Reshape((16, 32), input_shape=(512,))(essay3)
        # essay3 = Bidirectional(GRU(gru_len, activation='relu', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True))(essay3)
        # essay3 = Capsule(num_capsule=10, dim_capsule=16, routings=5, share_weights=True)(essay3)
        # essay3 = Flatten()(essay3)
        essay3 = Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(kernel_regularizer),
            bias_regularizer=regularizers.l2(bias_regularizer))(essay3)
        essay3 = Dropout(dropout_rate)(essay3)

        essay4_input = Input(shape=(1,), dtype=tf.string, name='essay4_input')
        essay4 = Lambda(USE_Embedding, output_shape=(512,))(essay4_input)
        # essay4 = Reshape((16, 32), input_shape=(512,))(essay4)
        # essay4 = Bidirectional(GRU(gru_len, activation='relu', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True))(essay4)
        # essay4 = Capsule(num_capsule=10, dim_capsule=16, routings=5, share_weights=True)(essay4)
        # essay4 = Flatten()(essay4)
        essay4 = Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(kernel_regularizer),
            bias_regularizer=regularizers.l2(bias_regularizer))(essay4)
        essay4 = Dropout(dropout_rate)(essay4)

    resource_summary_input = Input(shape=(1,), dtype=tf.string, name='resource_summary_input')
    resource_summary = Lambda(USE_Embedding, output_shape=(512,))(resource_summary_input)
    # resource_summary = Reshape((16, 32), input_shape=(512,))(resource_summary)
    # resource_summary = Bidirectional(GRU(gru_len, activation='relu', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True))(resource_summary)
    # resource_summary = Capsule(num_capsule=10, dim_capsule=16, routings=5, share_weights=True)(resource_summary)
    # resource_summary = Flatten()(resource_summary)
    resource_summary = Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(resource_summary)
    resource_summary = Dropout(dropout_rate)(resource_summary)

    description_input = Input(shape=(1,), dtype=tf.string, name='description_input')
    description = Lambda(USE_Embedding, output_shape=(512,))(description_input)
    # description = Reshape((16, 32), input_shape=(512,))(description)
    # description = Bidirectional(GRU(gru_len, activation='relu', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True))(description)
    # description = Capsule(num_capsule=10, dim_capsule=16, routings=5, share_weights=True)(description)
    # description = Flatten()(description)
    description = Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(description)
    description = Dropout(dropout_rate)(description)

    if old:
        x = concatenate([cat, cont, title, essay1, essay2, essay3, essay4, resource_summary, description])
    else:
        x = concatenate([cat, cont, title, essay1, essay2, resource_summary, description])

    x = Dense(
        256,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(x)  # down size the learnt representation
    x = Dropout(dropout_rate)(x)
    x = Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(x)  # down size the learnt representation
    x = Dropout(dropout_rate)(x)

    output = Dense(output_shape, activation='sigmoid', name='output')(x)

    if old:
        model = Model(inputs=[cat_input, cont_input, title_input, essay1_input, essay2_input, essay3_input, essay4_input, resource_summary_input, description_input], outputs=[output])
    else:
        model = Model(inputs=[cat_input, cont_input, title_input, essay1_input, essay2_input, resource_summary_input, description_input], outputs=[output])

    model.compile(optimizer=optimizers.Adam(0.0005, decay=1e-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy', auc])

    return model
#end def


def batch_iter(
    X_cat, X_cont, X_title,
    X_essay1, X_essay2,
    X_resource_summary, X_description, y,
    X_essay3=None, X_essay4=None,
    batch_size=128, old=False, **kwargs):

    data_size = X_cat.shape[0]
    num_batches = int((data_size - 1) / batch_size) + 1

    def _data_generator():
        while True:
            # Shuffle the data at each epoch
            shuffled_indices = np.random.permutation(np.arange(data_size, dtype=np.int))

            for i in range(num_batches):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, data_size)

                X_cat_batch = [X_cat[i] for i in shuffled_indices[start_index:end_index]]
                X_cont_batch = [X_cont[i] for i in shuffled_indices[start_index:end_index]]

                X_title_batch = [X_title[i] for i in shuffled_indices[start_index:end_index]]
                X_essay1_batch = [X_essay1[i] for i in shuffled_indices[start_index:end_index]]
                X_essay2_batch = [X_essay2[i] for i in shuffled_indices[start_index:end_index]]
                X_resource_summary_batch = [X_resource_summary[i] for i in shuffled_indices[start_index:end_index]]
                X_description_batch = [X_description[i] for i in shuffled_indices[start_index:end_index]]
                y_batch = [y[i] for i in shuffled_indices[start_index:end_index]]

                if old:
                    X_essay3_batch = [X_essay3[i] for i in shuffled_indices[start_index:end_index]]
                    X_essay4_batch = [X_essay4[i] for i in shuffled_indices[start_index:end_index]]
                    yield ({
                        'cat_input': np.asarray(X_cat_batch), 'cont_input': np.asarray(X_cont_batch),
                        'title_input': np.asarray(X_title_batch), 'essay1_input': np.asarray(X_essay1_batch),
                        'essay2_input': np.asarray(X_essay2_batch), 'essay3_input': np.asarray(X_essay3_batch),
                        'essay4_input': np.asarray(X_essay4_batch), 'resource_summary_input': np.asarray(X_resource_summary_batch),
                        'description_input': np.asarray(X_description_batch)},
                        {'output': np.asarray(y_batch)})
                else:
                    yield ({
                        'cat_input': np.asarray(X_cat_batch), 'cont_input': np.asarray(X_cont_batch),
                        'title_input': np.asarray(X_title_batch), 'essay1_input': np.asarray(X_essay1_batch),
                        'essay2_input': np.asarray(X_essay2_batch), 'resource_summary_input': np.asarray(X_resource_summary_batch),
                        'description_input': np.asarray(X_description_batch)},
                        {'output': np.asarray(y_batch)})
                #end if
            #end for
        #end while
    #end def

    return num_batches, _data_generator()
#end def


def train(
    model, X_cat_train, X_cont_train, X_title_train,
    X_essay1_train, X_essay2_train,
    X_resource_summary_train, X_description_train, y_train,
    X_essay3_train=None, X_essay4_train=None,
    X_cat_val=None, X_cont_val=None, X_title_val=None,
    X_essay1_val=None, X_essay2_val=None, X_essay3_val=None, X_essay4_val=None,
    X_resource_summary_val=None, X_description_val=None, y_val=None,
    class_weight=None, batch_size=128, epochs=32, old=False, **kwargs):

    tf_session = tf.Session()
    K.set_session(tf_session)
    K.get_session().run(tf.tables_initializer())

    init_op = tf.global_variables_initializer()
    tf_session.run(init_op)

    if old:
        train_steps, train_batches = batch_iter(
            X_cat_train, X_cont_train, X_title_train,
            X_essay1_train, X_essay2_train,
            X_resource_summary_train, X_description_train, y_train,
            X_essay3_train, X_essay4_train,
            batch_size=batch_size, old=old)

        if y_val is not None:
            val_steps, val_batches = batch_iter(
                X_cat_val, X_cont_val, X_title_val,
                X_essay1_val, X_essay2_val,
                X_resource_summary_val, X_description_val, y_val,
                X_essay3_val, X_essay4_val,
                batch_size=batch_size, old=old)
    else:
        train_steps, train_batches = batch_iter(
            X_cat_train, X_cont_train, X_title_train,
            X_essay1_train, X_essay2_train,
            X_resource_summary_train, X_description_train, y_train,
            batch_size=batch_size, old=old)

        if y_val is not None:
            val_steps, val_batches = batch_iter(
                X_cat_val, X_cont_val, X_title_val,
                X_essay1_val, X_essay2_val,
                X_resource_summary_val, X_description_val, y_val,
                batch_size=batch_size, old=old)

    # define early stopping callback
    callbacks_list = []
    if y_val is not None:
        early_stopping = dict(monitor='val_loss', patience=3, min_delta=0.0001, verbose=1)
        model_checkpoint = dict(filepath='./weights/{val_loss:.5f}_{loss:.5f}_{epoch:04d}.weights.h5',
                                save_best_only=True,
                                save_weights_only=True,
                                mode='auto',
                                period=1,
                                verbose=1)
    else:
        early_stopping = dict(monitor='loss', patience=1, min_delta=0.001, verbose=1)
        model_checkpoint = dict(filepath='./weights/{loss:.5f}_{epoch:04d}.weights.h5',
                                save_best_only=True,
                                save_weights_only=True,
                                mode='auto',
                                period=1,
                                verbose=1)

    earlystop = EarlyStopping(**early_stopping)
    callbacks_list.append(earlystop)

    checkpoint = ModelCheckpoint(**model_checkpoint)
    callbacks_list.append(checkpoint)

    if y_val is not None:
        model.fit_generator(
            epochs=epochs,
            generator=train_batches,
            steps_per_epoch=train_steps,
            validation_data=val_batches,
            validation_steps=val_steps,
            callbacks=callbacks_list,
            class_weight=class_weight)
    else:
        model.fit_generator(
            epochs=epochs,
            generator=train_batches,
            steps_per_epoch=train_steps,
            callbacks=callbacks_list,
            class_weight=class_weight)

    return model
#end def


def predict_iter(
    X_cat, X_cont, X_title,
    X_essay1, X_essay2,
    X_resource_summary, X_description,
    X_essay3=None, X_essay4=None,
    batch_size=128, old=False, **kwargs):

    data_size = X_cat.shape[0]
    num_batches = int((data_size - 1) / batch_size) + 1

    def _data_generator():
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, data_size)

            X_cat_batch = [x for x in X_cat[start_index:end_index]]
            X_cont_batch = [x for x in X_cont[start_index:end_index]]

            X_title_batch = [x for x in X_title[start_index:end_index]]
            X_essay1_batch = [x for x in X_essay1[start_index:end_index]]
            X_essay2_batch = [x for x in X_essay2[start_index:end_index]]
            X_resource_summary_batch = [x for x in X_resource_summary[start_index:end_index]]
            X_description_batch = [x for x in X_description[start_index:end_index]]

            if old:
                X_essay3_batch = [x for x in X_essay3[start_index:end_index]]
                X_essay4_batch = [x for x in X_essay4[start_index:end_index]]
                yield ({
                    'cat_input': np.asarray(X_cat_batch), 'cont_input': np.asarray(X_cont_batch),
                    'title_input': np.asarray(X_title_batch), 'essay1_input': np.asarray(X_essay1_batch),
                    'essay2_input': np.asarray(X_essay2_batch), 'essay3_input': np.asarray(X_essay3_batch),
                    'essay4_input': np.asarray(X_essay4_batch), 'resource_summary_input': np.asarray(X_resource_summary_batch),
                    'description_input': np.asarray(X_description_batch)})
            else:
                yield ({
                    'cat_input': np.asarray(X_cat_batch), 'cont_input': np.asarray(X_cont_batch),
                    'title_input': np.asarray(X_title_batch), 'essay1_input': np.asarray(X_essay1_batch),
                    'essay2_input': np.asarray(X_essay2_batch), 'resource_summary_input': np.asarray(X_resource_summary_batch),
                    'description_input': np.asarray(X_description_batch)})
        #end for
    #end def

    return num_batches, _data_generator()
#end def


def test(
    model, X_cat_test, X_cont_test, X_title_test,
    X_essay1_test, X_essay2_test,
    X_resource_summary_test, X_description_test,
    X_essay3_test=None, X_essay4_test=None,
    batch_size=128, old=False, **kwargs):

    test_steps, test_batches = predict_iter(
        X_cat_test, X_cont_test, X_title_test,
        X_essay1_test, X_essay2_test,
        X_resource_summary_test, X_description_test,
        X_essay3=X_essay3_test, X_essay4=X_essay4_test,
        batch_size=batch_size, old=old, **kwargs)

    return model.predict_generator(generator=test_batches, steps=test_steps)
#end def


def main():
    parser = ArgumentParser(description='Run machine learning experiment.')
    parser.add_argument('-i', '--train', type=str, metavar='<train_set>', required=True, help='Training data set.')
    parser.add_argument('-t', '--test', type=str, metavar='<test_set>', required=True, help='Test data set.')
    parser.add_argument('-r', '--resource', type=str, metavar='<resource_data>', required=True, help='Resource data.')    
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--cat_encode', type=str, default='mean', const='mean', nargs='?', choices=['mean', 'onehot'],help='Choose encoding method for categorical features.')
    A = parser.parse_args()

    log_level = 'DEBUG'
    log_format = '%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    # set seed
    np.random.seed(A.seed)
    random.seed(A.seed)
    tf.set_random_seed(A.seed)

    ######### old ##################
    old = True
    quick = False
    validate = False
    batch_size = 256
    epochs = 4
    threshold = 5 # Anything that occurs less than this will be removed.

    # read in data
    train_df = read(A.resource, A.train, old=old, quick=quick)

    # simple data exploration and na detection
    # analyse(train_df, categorical_features, target_feature, old_continuous_features)    # # 4 NaN detected in teacher_prefix
    train_df['teacher_prefix'] = train_df['teacher_prefix'].fillna('Teacher')

    value_counts = train_df['project_subject_subcategories'].value_counts()  # Specific column 
    to_replace = value_counts[value_counts <= threshold].index.values
    to_replace_arr = list(train_df['project_subject_subcategories'].loc[train_df['project_subject_subcategories'].isin(to_replace)].unique())
    train_df['project_subject_subcategories'].replace(to_replace, 'Others', inplace=True)
    # analyse(train_df, categorical_features, target_feature, old_continuous_features)

    if validate: train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=A.seed)

    # target encode categorical features
    if A.cat_encode == 'mean':
        cat_encoder = TargetEncoder()
        train_cat = cat_encoder.fit_transform(train_df[categorical_features].values, pd.to_numeric(train_df['project_is_approved']).values).values
        if validate: val_cat = cat_encoder.transform(val_df[categorical_features].values).values
    elif A.cat_encode == 'onehot':
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        train_cat = cat_encoder.fit_transform(train_df[categorical_features].values)
        if validate: val_cat = cat_encoder.transform(val_df[categorical_features].values)
    else:
        warnings.warn("Unknown categorical encoding method....")
    #end if

    # normalize continuous features
    scaler = MinMaxScaler()
    train_cont = scaler.fit_transform(train_df[old_continuous_features])
    y_train = train_df['project_is_approved'].values
    if validate:
        val_cont = scaler.transform(val_df[old_continuous_features])
        y_val = val_df['project_is_approved'].values

    model = build_model(cat_input_shape=train_cat.shape[1:], cont_input_shape=train_cont.shape[1:], output_shape=1, old=old)
    print(model.summary())

    class_weights = compute_class_weight('balanced', np.asarray([0, 1]), y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    train_input_dict = dict(
        model=model,
        X_cat_train=train_cat,
        X_cont_train=train_cont,
        X_title_train=train_df['project_title'].values,
        X_essay1_train=train_df['project_essay_1'].values,
        X_essay2_train=train_df['project_essay_2'].values,
        X_essay3_train=train_df['project_essay_3'].values,
        X_essay4_train=train_df['project_essay_4'].values,
        X_resource_summary_train=train_df['project_resource_summary'].values,
        X_description_train=train_df['description'].values,
        y_train=y_train,
        class_weight=class_weight_dict,
        batch_size=batch_size,
        epochs=epochs,
        old=old)
    if validate:
        train_input_dict.update(
            dict(
                X_cat_val=val_cat,
                X_cont_val=val_cont,
                X_title_val=val_df['project_title'].values,
                X_essay1_val=val_df['project_essay_1'].values,
                X_essay2_val=val_df['project_essay_2'].values,
                X_essay3_val=val_df['project_essay_3'].values,
                X_essay4_val=val_df['project_essay_4'].values,
                X_resource_summary_val=val_df['project_resource_summary'].values,
                X_description_val=val_df['description'].values,
                y_val=y_val))
    #end if

    model = train(**train_input_dict)
    del train_df, train_cat, train_cont, y_train, train_input_dict
    if validate: del val_df, val_cat, val_cont, y_val
    gc.collect()

    old_test_df = read(A.resource, A.test, old=old)
    old_test_df['teacher_prefix'] = old_test_df['teacher_prefix'].fillna('Teacher')
    old_test_df['project_subject_subcategories'].replace(to_replace_arr, 'Others', inplace=True)

    if A.cat_encode == 'mean': test_cat = cat_encoder.transform(old_test_df[categorical_features].values).values
    elif A.cat_encode == 'onehot': test_cat = cat_encoder.transform(old_test_df[categorical_features].values)

    test_cont = scaler.transform(old_test_df[old_continuous_features])

    test_input_dict = dict(
        model=model,
        X_cat_test=test_cat,
        X_cont_test=test_cont,
        X_title_test=old_test_df['project_title'].values,
        X_essay1_test=old_test_df['project_essay_1'].values,
        X_essay2_test=old_test_df['project_essay_2'].values,
        X_essay3_test=old_test_df['project_essay_3'].values,
        X_essay4_test=old_test_df['project_essay_4'].values,
        X_resource_summary_test=old_test_df['project_resource_summary'].values,
        X_description_test=old_test_df['description'].values,
        batch_size=64,
        old=old)

    old_y_pred = test(**test_input_dict)

    logger.info("Done with old dataset....")

    ######### new ##################
    old = False
    quick = False
    validate = False
    batch_size = 256
    epochs = 4
    threshold = 50 # Anything that occurs less than this will be removed.

    # read in data
    train_df = read(A.resource, A.train, old=old, quick=quick)
    train_df['teacher_prefix'] = train_df['teacher_prefix'].fillna('Teacher')

    value_counts = train_df['project_subject_subcategories'].value_counts()  # Specific column 
    to_replace = value_counts[value_counts <= threshold].index.values
    to_replace_arr = list(train_df['project_subject_subcategories'].loc[train_df['project_subject_subcategories'].isin(to_replace)].unique())
    train_df['project_subject_subcategories'].replace(to_replace, 'Others', inplace=True)

    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=A.seed)

    # target encode categorical features
    if A.cat_encode == 'mean':
        cat_encoder = TargetEncoder()
        train_cat = cat_encoder.fit_transform(train_df[categorical_features].values, pd.to_numeric(train_df['project_is_approved']).values).values
        if validate: val_cat = cat_encoder.transform(val_df[categorical_features].values).values
    elif A.cat_encode == 'onehot':
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        train_cat = cat_encoder.fit_transform(train_df[categorical_features].values)
        if validate: val_cat = cat_encoder.transform(val_df[categorical_features].values)
    else:
        warnings.warn("Unknown categorical encoding method....")
    #end if

    # normalize continuous features
    scaler = MinMaxScaler()
    train_cont = scaler.fit_transform(train_df[new_continuous_features])
    y_train = train_df['project_is_approved'].values
    if validate:
        val_cont = scaler.transform(val_df[new_continuous_features])
        y_val = val_df['project_is_approved'].values

    model = build_model(cat_input_shape=train_cat.shape[1:], cont_input_shape=train_cont.shape[1:], output_shape=1, old=old)
    print(model.summary())

    class_weight = compute_class_weight('balanced', np.asarray([0, 1]), y_train)
    class_weight = {i: weight for i, weight in enumerate(class_weight)}

    train_input_dict = dict(
        model=model,
        X_cat_train=train_cat,
        X_cont_train=train_cont,
        X_title_train=train_df['project_title'].values,
        X_essay1_train=train_df['project_essay_1'].values,
        X_essay2_train=train_df['project_essay_2'].values,
        X_resource_summary_train=train_df['project_resource_summary'].values,
        X_description_train=train_df['description'].values,
        y_train=y_train,
        class_weight=class_weight,
        batch_size=batch_size,
        epochs=epochs,
        old=old)
    if validate:
        train_input_dict.update(
            dict(
                X_cat_val=val_cat,
                X_cont_val=val_cont,
                X_title_val=val_df['project_title'].values,
                X_essay1_val=val_df['project_essay_1'].values,
                X_essay2_val=val_df['project_essay_2'].values,
                X_resource_summary_val=val_df['project_resource_summary'].values,
                X_description_val=val_df['description'].values,
                y_val=y_val))
    #end if

    model = train(**train_input_dict)
    del train_df, train_cat, train_cont, y_train, train_input_dict
    if validate: del val_df, val_cat, val_cont, y_val
    gc.collect()

    new_test_df = read(A.resource, A.test, old=old, quick=quick)
    new_test_df['teacher_prefix'] = new_test_df['teacher_prefix'].fillna('Teacher')
    new_test_df['project_subject_subcategories'].replace(to_replace_arr, 'Others', inplace=True)

    if A.cat_encode == 'mean': test_cat = cat_encoder.transform(new_test_df[categorical_features].values).values
    elif A.cat_encode == 'onehot': test_cat = cat_encoder.transform(new_test_df[categorical_features].values)

    test_cont = scaler.transform(new_test_df[new_continuous_features])

    test_input_dict = dict(
        model=model,
        X_cat_test=test_cat,
        X_cont_test=test_cont,
        X_title_test=new_test_df['project_title'].values,
        X_essay1_test=new_test_df['project_essay_1'].values,
        X_essay2_test=new_test_df['project_essay_2'].values,
        X_resource_summary_test=new_test_df['project_resource_summary'].values,
        X_description_test=new_test_df['description'].values,
        batch_size=64,
        old=old)

    new_y_pred = test(**test_input_dict)

    ######### merge ##################
    old_test_df['project_is_approved'] = old_y_pred
    new_test_df['project_is_approved'] = new_y_pred

    test_df = pd.concat([old_test_df, new_test_df], ignore_index=True)
    out_df = test_df[['id', 'project_is_approved']]
    out_df.to_csv('./data/mean_oldnew_use_submission.csv', index=False)
#end def


if __name__ == '__main__': main()
