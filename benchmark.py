import gc
import lightgbm as lgb
import logging
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import tensorflow_hub as hub
import yaml

from tqdm import tqdm
from category_encoders.target_encoder import TargetEncoder

from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional
from keras.layers import Convolution1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GRU
from keras.layers import GlobalMaxPool1D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import SpatialDropout1D
from keras.layers import concatenate
from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing import text

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)

target_feature = ['project_is_approved']

raw_string_features = [
    'project_title', 'project_essay_1',
    'project_essay_2', 'project_essay_3',
    'project_essay_4', 'project_resource_summary',
    'description']

config_path = 'benchmark.yaml'

# Load in config file
with open(config_path, 'r') as f:
    config = yaml.load(f)
#end with

print('='*50)
for k, v in config.items():
    if isinstance(v, dict):
        print("{}:".format(k))
        one = False
        for (_k, _v) in v.items():
            print("    {}: {}".format(_k, _v))
            if _v is True and (not one):
                one = True
            elif _v is not True:
                pass
            else:
                raise Exception("Should only enable one of the parameters under <{}>".format(k))
        continue
    else:
        print("{}:{}".format(k,v))
print('='*50)


Tags = ['CC', 'CD', 'DT', 'IN', 'JJ', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 
        'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 
        'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']


Keywords = ['!', '\?', '@', '#', '\$', '%', '&', '\*', '\(', '\[', '\{', '\|', '-', '_', '=', '\+',
            '\.', ':', ';', ',', '/', '\\\\r', '\\\\t', '\\"', '\.\.\.', 'etc', 'http', 'poor',
            'military', 'traditional', 'charter', 'head start', 'magnet', 'year-round', 'alternative',
            'art', 'book', 'basics', 'computer', 'laptop', 'tablet', 'kit', 'game', 'seat',
            'food', 'cloth', 'hygiene', 'instraction', 'technolog', 'lab', 'equipment',
            'music', 'instrument', 'nook', 'desk', 'storage', 'sport', 'exercise', 'trip', 'visitor',
            'my students', 'our students', 'my class', 'our class']


# Preparation for type of embedding
if config['embedding']['word_vector']:
    logger.info('Reading in embedding....')
    embeddings_index = {}
    with open(config['embedding']['word_path'], encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        #end for
    #end with
    logger.info('Done reading in embedding....')
#end if

USE_MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/2"
USE_EMBED = hub.Module(USE_MODULE_URL, trainable=True)


def USE_Embedding(x):
    return USE_EMBED(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
#end def


def read(df_path, old=True, quick=False):
    string_features = []
    categorical_features = [
        'teacher_prefix', 'school_state',
        'project_grade_category',
        'project_subject_categories', 'project_subject_subcategories',
        'dayofweek', 'monthofyear',
        ]
    continuous_features = ['teacher_number_of_previously_posted_projects']

    logger.info("Reading in data from {}....".format(df_path))

    df = pd.read_csv(df_path)

    if quick:
        df = df[:40000]

    if old:
        if config['embedding']['use']:
            df = df[~df['project_essay_3'].isna()].reset_index()

        # get sent count of essays
        if config['sent_count']:
            continuous_features.extend([
                'project_essay_1_sent_len', 'project_essay_2_sent_len',
                'project_essay_3_sent_len', 'project_essay_4_sent_len'])
        #end if

        # get word count of essays
        if config['word_count']:
            continuous_features.extend([
                'project_essay_1_word_len', 'project_essay_2_word_len',
                'project_essay_3_word_len', 'project_essay_4_word_len'])
        #end if

        # get polarity
        if config['polarity']:
            continuous_features.append('polarity')
        #end if

        # get subjectivity
        if config['subjectivity']:
            continuous_features.append('subjectivity')
        #end if

        # if tfidf/wordvector, merge all text into one
        if config['embedding']['tfidf'] or config['embedding']['word_vector']:
            string_features.append('all_text')
        elif config['embedding']['use']:
            string_features.extend([
                'project_title', 'project_essay_1',
                'project_essay_2', 'project_essay_3',
                'project_essay_4', 'project_resource_summary',
                'description'])
        #end if
    else:
        df = df[df['project_essay_3'].isna()].reset_index()

        # get sent count of essays
        if config['sent_count']:
            continuous_features.extend([
                'project_essay_1_sent_len', 'project_essay_2_sent_len'])
        #end if

        # get word count of essays
        if config['word_count']:
            continuous_features.extend([
                'project_essay_1_word_len', 'project_essay_2_word_len'])
        #end if

        # get polarity
        if config['polarity']:
            continuous_features.append('polarity')
        #end if

        # get subjectivity
        if config['subjectivity']:
            continuous_features.append('subjectivity')
        #end if

        # if tfidf/wordvector, merge all text into one
        if config['embedding']['tfidf'] or config['embedding']['word_vector']:
            string_features.append('all_text')
        elif config['embedding']['use']:
            string_features.extend([
                'project_title', 'project_essay_1',
                'project_essay_2', 'project_resource_summary',
                'description'])
        #end if
    #end if

    # get pos count for each text attribute
    if config['pos_count']:
        continuous_features.extend([f + '_' + t + '_count' for f in raw_string_features for t in Tags])
    #end if

    # get keyword count for each text attribute
    if config['keyword_count']:
        continuous_features.extend([f + '_' + t + '_count' for f in raw_string_features for t in Keywords])
    #end if

    # get common word count for each text attribute pair
    if config['common_word_count']:
        continuous_features.extend(['%s_%s_common' % (f1, f2) for i, f1 in enumerate(raw_string_features[:-1]) for f2 in raw_string_features[i+1:]])
    #end if

    logger.info("Done reading in {} data....".format(df.shape[0]))

    df['teacher_prefix'] = df['teacher_prefix'].fillna('Teacher')

    return df, continuous_features, categorical_features, string_features
#end def


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


def text_tokenize(train, test, col_name, num_words, maxlen, embeddings_index, embed_size):
    logger.info('Computing in embedding matrixes....')
    tokenizer = text.Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(train[col_name].tolist() + test[col_name].tolist())
    train_tokenized = tokenizer.texts_to_sequences(train[col_name].tolist())
    test_tokenized = tokenizer.texts_to_sequences(test[col_name].tolist())
    X_train = sequence.pad_sequences(train_tokenized, maxlen=maxlen)
    X_test = sequence.pad_sequences(test_tokenized, maxlen=maxlen)

    word_index = tokenizer.word_index
    #prepare embedding matrix
    num_words = min(num_words, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embed_size))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    logger.info('Done computing in embedding matrixes....')

    return X_train, X_test, embedding_matrix
#end def


def build_model_word_vector(
    cat_input_shape,
    cont_input_shape,
    all_text_input_shape,
    output_shape,
    text_embedding_matrix,
    text_max_features,
    dropout_rate=0,
    kernel_regularizer=0,
    activity_regularizer=0,
    bias_regularizer=0, **kwargs):
    
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

    all_text_input = Input(all_text_input_shape, name='all_text_input')
    all_text = Embedding(
        text_max_features, 300,
        weights=[text_embedding_matrix],
        trainable=False)(all_text_input)
    all_text = SpatialDropout1D(dropout_rate)(all_text)
    all_text = Bidirectional(GRU(50, return_sequences=True))(all_text)
    all_text = Convolution1D(100, 3, activation="relu")(all_text)
    all_text = GlobalMaxPool1D()(all_text)

    x = concatenate([cat, cont, all_text])

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

    model = Model(inputs=[cat_input, cont_input, all_text_input], outputs=[output])
    model.compile(optimizer=optimizers.Adam(0.0005, decay=1e-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy', auc])

    return model
#end def


def build_model_tfidf(
    cat_input_shape,
    cont_input_shape,
    all_text_input_shape,
    output_shape,
    dropout_rate=0,
    kernel_regularizer=0,
    activity_regularizer=0,
    bias_regularizer=0, **kwargs):
    
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

    all_text_input = Input(all_text_input_shape, name='all_text_input')
    all_text = Dense(
        2048,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(all_text_input)  # down size the learnt representation
    all_text = Dropout(dropout_rate)(all_text)
    all_text = Dense(
        512,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(all_text)  # down size the learnt representation
    all_text = Dropout(dropout_rate)(all_text)

    x = concatenate([cat, cont, all_text])

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

    model = Model(inputs=[cat_input, cont_input, all_text_input], outputs=[output])
    model.compile(optimizer=optimizers.Adam(0.0005, decay=1e-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy', auc])

    return model
#end def


def build_model_use(
    cat_input_shape,
    cont_input_shape,
    output_shape,
    dropout_rate=0,
    kernel_regularizer=0,
    activity_regularizer=0,
    bias_regularizer=0,
    old=False, **kwargs):
    
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
    title = Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(title)
    title = Dropout(dropout_rate)(title)

    essay1_input = Input(shape=(1,), dtype=tf.string, name='essay1_input')
    essay1 = Lambda(USE_Embedding, output_shape=(512,))(essay1_input)
    essay1 = Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(essay1)
    essay1 = Dropout(dropout_rate)(essay1)

    essay2_input = Input(shape=(1,), dtype=tf.string, name='essay2_input')
    essay2 = Lambda(USE_Embedding, output_shape=(512,))(essay2_input)
    essay2 = Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(essay2)
    essay2 = Dropout(dropout_rate)(essay2)

    if old:
        essay3_input = Input(shape=(1,), dtype=tf.string, name='essay3_input')
        essay3 = Lambda(USE_Embedding, output_shape=(512,))(essay3_input)
        essay3 = Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(kernel_regularizer),
            bias_regularizer=regularizers.l2(bias_regularizer))(essay3)
        essay3 = Dropout(dropout_rate)(essay3)

        essay4_input = Input(shape=(1,), dtype=tf.string, name='essay4_input')
        essay4 = Lambda(USE_Embedding, output_shape=(512,))(essay4_input)
        essay4 = Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(kernel_regularizer),
            bias_regularizer=regularizers.l2(bias_regularizer))(essay4)
        essay4 = Dropout(dropout_rate)(essay4)

    resource_summary_input = Input(shape=(1,), dtype=tf.string, name='resource_summary_input')
    resource_summary = Lambda(USE_Embedding, output_shape=(512,))(resource_summary_input)
    resource_summary = Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(resource_summary)
    resource_summary = Dropout(dropout_rate)(resource_summary)

    description_input = Input(shape=(1,), dtype=tf.string, name='description_input')
    description = Lambda(USE_Embedding, output_shape=(512,))(description_input)
    description = Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(description)
    description = Dropout(dropout_rate)(description)

    if old:
        text = concatenate([title, essay1, essay2, essay3, essay4, resource_summary, description])
    else:
        text = concatenate([title, essay1, essay2, resource_summary, description])

    text = Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(kernel_regularizer),
        bias_regularizer=regularizers.l2(bias_regularizer))(text)  # down size the learnt representation
    
    x = concatenate([cat, cont, text])

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


def batch_iter_use(
    X_cat,
    X_cont,
    X_title,
    X_essay1,
    X_essay2,
    X_resource_summary,
    X_description,
    y,
    X_essay3=None,
    X_essay4=None,
    batch_size=128,
    old=False, **kwargs):

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
                        'cat_input': np.asarray(X_cat_batch),
                        'cont_input': np.asarray(X_cont_batch),
                        'title_input': np.asarray(X_title_batch),
                        'essay1_input': np.asarray(X_essay1_batch),
                        'essay2_input': np.asarray(X_essay2_batch),
                        'essay3_input': np.asarray(X_essay3_batch),
                        'essay4_input': np.asarray(X_essay4_batch),
                        'resource_summary_input': np.asarray(X_resource_summary_batch),
                        'description_input': np.asarray(X_description_batch),
                        },
                        {'output': np.asarray(y_batch)})
                else:
                    yield ({
                        'cat_input': np.asarray(X_cat_batch),
                        'cont_input': np.asarray(X_cont_batch),
                        'title_input': np.asarray(X_title_batch),
                        'essay1_input': np.asarray(X_essay1_batch),
                        'essay2_input': np.asarray(X_essay2_batch),
                        'resource_summary_input': np.asarray(X_resource_summary_batch),
                        'description_input': np.asarray(X_description_batch),
                        },
                        {'output': np.asarray(y_batch)})
                #end if
            #end for
        #end while
    #end def

    return num_batches, _data_generator()
#end def


def train_use(
    model,
    X_cat_train,
    X_cont_train,
    X_title_train,
    X_essay1_train,
    X_essay2_train,
    X_resource_summary_train,
    X_description_train,
    y_train,
    X_essay3_train=None,
    X_essay4_train=None,
    batch_size=128,
    epochs=32,
    old=False, **kwargs):

    tf_session = tf.Session()
    K.set_session(tf_session)
    K.get_session().run(tf.tables_initializer())

    init_op = tf.global_variables_initializer()
    tf_session.run(init_op)

    if old:
        train_steps, train_batches = batch_iter_use(
            X_cat_train,
            X_cont_train,
            X_title_train,
            X_essay1_train,
            X_essay2_train,
            X_resource_summary_train,
            X_description_train,
            y_train,
            X_essay3_train,
            X_essay4_train,
            batch_size=batch_size,
            old=old)
    else:
        train_steps, train_batches = batch_iter_use(
            X_cat_train,
            X_cont_train,
            X_title_train,
            X_essay1_train,
            X_essay2_train,
            X_resource_summary_train,
            X_description_train,
            y_train,
            batch_size=batch_size,
            old=old)

    # define early stopping callback
    callbacks_list = []
    early_stopping = dict(monitor='loss', patience=1, min_delta=0.001, verbose=1)
    model_checkpoint = dict(filepath='./weights/{loss:.5f}_{epoch:04d}.weights.h5',
                            save_best_only=False,
                            save_weights_only=True,
                            mode='auto',
                            period=1,
                            verbose=1)

    earlystop = EarlyStopping(**early_stopping)
    callbacks_list.append(earlystop)

    checkpoint = ModelCheckpoint(**model_checkpoint)
    callbacks_list.append(checkpoint)

    # if old:
    #     x_dict = dict(
    #         cat_input=X_cat_train,
    #         cont_input=X_cont_train,
    #         title_input=X_title_train,
    #         essay1_input=X_essay1_train,
    #         essay2_input=X_essay2_train,
    #         essay3_input=X_essay3_train,
    #         essay4_input=X_essay4_train,
    #         resource_summary_input=X_resource_summary_train,
    #         description_input=X_description_train,
    #         )
    # else:
    #     x_dict = dict(
    #         cat_input=X_cat_train,
    #         cont_input=X_cont_train,
    #         title_input=X_title_train,
    #         essay1_input=X_essay1_train,
    #         essay2_input=X_essay2_train,
    #         resource_summary_input=X_resource_summary_train,
    #         description_input=X_description_train,
    #         )

    # y_dict = dict(output=y_train)
    # model.fit(x_dict, y_dict, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

    model.fit_generator(
        epochs=epochs,
        generator=train_batches,
        steps_per_epoch=train_steps,
        callbacks=callbacks_list)

    return model
#end def


def predict_iter_use(
    X_cat,
    X_cont,
    X_title,
    X_essay1,
    X_essay2,
    X_resource_summary,
    X_description,
    X_essay3=None,
    X_essay4=None,
    batch_size=128,
    old=False, **kwargs):

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
                    'cat_input': np.asarray(X_cat_batch),
                    'cont_input': np.asarray(X_cont_batch),
                    'title_input': np.asarray(X_title_batch),
                    'essay1_input': np.asarray(X_essay1_batch),
                    'essay2_input': np.asarray(X_essay2_batch),
                    'essay3_input': np.asarray(X_essay3_batch),
                    'essay4_input': np.asarray(X_essay4_batch),
                    'resource_summary_input': np.asarray(X_resource_summary_batch),
                    'description_input': np.asarray(X_description_batch),
                    })
            else:
                yield ({
                    'cat_input': np.asarray(X_cat_batch),
                    'cont_input': np.asarray(X_cont_batch),
                    'title_input': np.asarray(X_title_batch),
                    'essay1_input': np.asarray(X_essay1_batch),
                    'essay2_input': np.asarray(X_essay2_batch),
                    'resource_summary_input': np.asarray(X_resource_summary_batch),
                    'description_input': np.asarray(X_description_batch),
                    })
        #end for
    #end def

    return num_batches, _data_generator()
#end def


def test_use(
    model,
    X_cat_test,
    X_cont_test,
    X_title_test,
    X_essay1_test,
    X_essay2_test,
    X_resource_summary_test,
    X_description_test,
    X_essay3_test=None,
    X_essay4_test=None,
    batch_size=128,
    old=False, **kwargs):

    test_steps, test_batches = predict_iter_use(
        X_cat_test,
        X_cont_test,
        X_title_test,
        X_essay1_test,
        X_essay2_test,
        X_resource_summary_test,
        X_description_test,
        X_essay3=X_essay3_test,
        X_essay4=X_essay4_test,
        batch_size=batch_size,
        old=old,
        **kwargs)

    return model.predict_generator(generator=test_batches, steps=test_steps)
#end def


def train(
    model,
    X_cat_train,
    X_cont_train,
    X_all_text_train,
    y_train,
    batch_size=128,
    epochs=32, **kwargs):

    tf_session = tf.Session()
    K.set_session(tf_session)
    K.get_session().run(tf.tables_initializer())

    init_op = tf.global_variables_initializer()
    tf_session.run(init_op)

    # define early stopping callback
    callbacks_list = []
    early_stopping = dict(monitor='loss', patience=1, min_delta=0.001, verbose=1)
    model_checkpoint = dict(filepath='./weights/{loss:.5f}_{epoch:04d}.weights.h5',
                            save_best_only=False,
                            save_weights_only=True,
                            mode='auto',
                            period=1,
                            verbose=1)

    earlystop = EarlyStopping(**early_stopping)
    callbacks_list.append(earlystop)

    checkpoint = ModelCheckpoint(**model_checkpoint)
    callbacks_list.append(checkpoint)

    x_dict = dict(
        cat_input=X_cat_train,
        cont_input=X_cont_train,
        all_text_input=X_all_text_train,
        )
    y_dict = dict(output=y_train)
    model.fit(x_dict, y_dict, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
    return model
#end def


def test(
    model,
    X_cat_test,
    X_cont_test,
    X_all_text_test,
    batch_size=128, **kwargs):

    x_dict = dict(
        cat_input=X_cat_test,
        cont_input=X_cont_test,
        all_text_input=X_all_text_test,
        )
    return model.predict(x_dict, batch_size=batch_size)
#end def


def prepare_nn(train_df, test_df, old=False, continuous_features=[], categorical_features=[], string_features=[]):
    ############################
    logger.info("Preparing word embeddings")
    if config['embedding']['tfidf']:
        tfidf_vec = TfidfVectorizer(
            max_features=10000,
            sublinear_tf=True,
            strip_accents='unicode',
            stop_words='english',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            dtype=np.float32,
            norm='l2',
            min_df=5,
            max_df=.9)
        train_tfidf_text = tfidf_vec.fit_transform(train_df['all_text']).toarray()
        test_tfidf_text = tfidf_vec.transform(test_df['all_text']).toarray()    
    elif config['embedding']['word_vector']:
        text_max_features = 100000
        text_max_len = 300
        text_embed_size = 300
        train_word_text, test_word_text, text_embedding_matrix = text_tokenize(
            train_df, test_df, col_name='all_text',
            num_words=text_max_features, maxlen=text_max_len,
            embeddings_index=embeddings_index, embed_size=text_embed_size)
    elif config['embedding']['use']:
        pass
    #end if

    ############################
    logger.info("Encoding categorical features")
    # encode categorical features
    if config['cat_encoding']['one_hot_encoding']:
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        train_cat = cat_encoder.fit_transform(train_df[categorical_features].values)
        test_cat = cat_encoder.transform(test_df[categorical_features].values)
    elif config['cat_encoding']['mean_encoding']:
        cat_encoder = TargetEncoder()
        train_cat = cat_encoder.fit_transform(train_df[categorical_features].values, pd.to_numeric(train_df['project_is_approved']).values).values
        test_cat = cat_encoder.transform(test_df[categorical_features].values).values
    elif config['cat_encoding']['count_encoding']:
        for i, f in enumerate(categorical_features):
            cat_encoder = CountVectorizer(
                binary=True,
                ngram_range=(1, 1),
                tokenizer=lambda x: [a.strip() for a in x.split(',')])
            if i == 0:
                train_cat = cat_encoder.fit_transform(train_df[f]).toarray()
                test_cat = cat_encoder.transform(test_df[f]).toarray()
            else:
                _train_cat = cat_encoder.fit_transform(train_df[f]).toarray()
                _test_cat = cat_encoder.transform(test_df[f]).toarray()
                train_cat = np.hstack((train_cat, _train_cat))
                test_cat = np.hstack((test_cat, _test_cat))
            #end if
        #end for
    #end if

    ############################
    logger.info("Normalizing continuous features")
    # normalize train continuous features
    scaler = MinMaxScaler()
    train_cont = scaler.fit_transform(train_df[continuous_features])
    test_cont = scaler.transform(test_df[continuous_features])

    # get polynomial array of all continuous attributes
    if config['polynomial']:
        poly = PolynomialFeatures()
        train_cont = poly.fit_transform(train_cont)
        test_cont = poly.transform(test_cont)
    #end if

    ############################
    # prepare y
    y_train = pd.to_numeric(train_df['project_is_approved']).values

    ############################
    logger.info("Model building and training on train data")
    # build model
    if config['embedding']['use']:
        model = build_model_use(
            cat_input_shape=train_cat.shape[1:],
            cont_input_shape=train_cont.shape[1:],
            output_shape=1,
            old=old)
        print(model.summary())

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
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            old=old)

        model = train_use(**train_input_dict)
    elif config['embedding']['word_vector']:
        model = build_model_word_vector(
            cat_input_shape=train_cat.shape[1:],
            cont_input_shape=train_cont.shape[1:],
            all_text_input_shape=train_word_text.shape[1:],
            text_embedding_matrix=text_embedding_matrix,
            text_max_features=text_max_features,
            output_shape=1)
        print(model.summary())

        train_input_dict = dict(
            model=model,
            X_cat_train=train_cat,
            X_cont_train=train_cont,
            X_all_text_train=train_word_text,
            y_train=y_train,
            class_weight=None,
            batch_size=config['batch_size'],
            epochs=config['epochs'])

        model = train(**train_input_dict)
    elif config['embedding']['tfidf']:
        model = build_model_tfidf(
            cat_input_shape=train_cat.shape[1:],
            cont_input_shape=train_cont.shape[1:],
            all_text_input_shape=train_tfidf_text.shape[1:],
            output_shape=1)
        print(model.summary())

        train_input_dict = dict(
            model=model,
            X_cat_train=train_cat,
            X_cont_train=train_cont,
            X_all_text_train=train_tfidf_text,
            y_train=y_train,
            class_weight=None,
            batch_size=config['batch_size'],
            epochs=config['epochs'])

        model = train(**train_input_dict)
    #end if
    del train_df, train_cat, train_cont, y_train, train_input_dict
    gc.collect()

    ############################
    logger.info('Model testing on test data')
    if config['embedding']['use']:
        test_input_dict = dict(
            model=model,
            X_cat_test=test_cat,
            X_cont_test=test_cont,
            X_title_test=test_df['project_title'].values,
            X_essay1_test=test_df['project_essay_1'].values,
            X_essay2_test=test_df['project_essay_2'].values,
            X_essay3_test=test_df['project_essay_3'].values,
            X_essay4_test=test_df['project_essay_4'].values,
            X_resource_summary_test=test_df['project_resource_summary'].values,
            X_description_test=test_df['description'].values,
            batch_size=64,
            old=old)

        y_pred = test_use(**test_input_dict)
    elif config['embedding']['word_vector']:
        test_input_dict = dict(
            model=model,
            X_cat_test=test_cat,
            X_cont_test=test_cont,
            X_all_text_test=test_word_text,
            batch_size=64)
        y_pred = test(**test_input_dict)
    elif config['embedding']['tfidf']:
        test_input_dict = dict(
            model=model,
            X_cat_test=test_cat,
            X_cont_test=test_cont,
            X_all_text_test=test_tfidf_text,
            batch_size=64)
        y_pred = test(**test_input_dict)    
    #end if

    return y_pred
#end def


def prepare_lgbm(train_df, test_df, old=False, continuous_features=[], categorical_features=[], string_features=[]):
    df_all = pd.concat([train_df, test_df], axis=0)

    logger.info("Preparing word embeddings")
    if config['embedding']['tfidf']:
        cols = [
            'project_title',
            'all_essays',  # Concatenate 4 essay into all_essay
            # 'project_essay_1',
            # 'project_essay_2',
            # 'project_essay_3',
            # 'project_essay_4',
            'project_resource_summary',
            # 'description',
        ]
        n_features = [400, 5000, 400]
        for c_i, c in tqdm(enumerate(cols)):
            tfidf = TfidfVectorizer(
                max_features=n_features[c_i],
                min_df=3,
                norm='l2',
                ngrams=(1, 2))
            tfidf.fit(df_all[c])
            tfidf_train = np.array(tfidf.transform(train_df[c]).todense(), dtype=np.float16)
            tfidf_test = np.array(tfidf.transform(test_df[c]).todense(), dtype=np.float16)

            for i in range(n_features[c_i]):
                train_df[c + '_tfidf_' + str(i)] = tfidf_train[:, i]
                test_df[c + '_tfidf_' + str(i)] = tfidf_test[:, i]

            del tfidf, tfidf_train, tfidf_test
            gc.collect()
        print('Done.')

    del df_all
    gc.collect()

    # Prepare data
    cols_to_drop = [
        'id',
        'project_title',
        'project_essay',
        'project_resource_summary',
        'project_is_approved',
        'all_text',
        'all_essays',
        'project_essay_1',
        'project_essay_2',
        'project_essay_3',
        'project_essay_4',
        'description'
    ]
    X = train_df.drop(cols_to_drop, axis=1, errors='ignore')
    y = train_df['project_is_approved']
    X_test = test_df.drop(cols_to_drop, axis=1, errors='ignore')
    feature_names = list(X.columns)
    print(X.shape, X_test.shape)

    del train_df, test_df
    gc.collect()

    # Build the model
    cnt = 0
    p_buf = []
    n_splits = 5
    n_repeats = 1
    kf = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=0)
    auc_buf = []

    for train_index, valid_index in kf.split(X):
        print('Fold {}/{}'.format(cnt + 1, n_splits))
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'max_depth': 14,
            'num_leaves': 31,
            'learning_rate': 0.025,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'verbose': 0,
            'num_threads': 44,
            'lambda_l2': 1.0,
            'min_gain_to_split': 0,
        }

        model = lgb.train(
            params,
            lgb.Dataset(X.loc[train_index], y.loc[train_index], feature_name=feature_names),
            num_boost_round=10000,
            valid_sets=[lgb.Dataset(X.loc[valid_index], y.loc[valid_index])],
            early_stopping_rounds=100,
            verbose_eval=100,
        )

        if cnt == 0:
            importance = model.feature_importance()
            model_fnames = model.feature_name()
            tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
            tuples = [x for x in tuples if x[1] > 0]
            print('Important features:')
            print(tuples[:50])

        p = model.predict(X.loc[valid_index], num_iteration=model.best_iteration)
        auc = roc_auc_score(y.loc[valid_index], p)

        print('{} AUC: {}'.format(cnt, auc))

        p = model.predict(X_test, num_iteration=model.best_iteration)
        if len(p_buf) == 0:
            p_buf = np.array(p)
        else:
            p_buf += np.array(p)
        auc_buf.append(auc)

        cnt += 1
        # if cnt > 0:  # Comment this to run several folds
        #     break

        del model
        gc.collect()
    #end for

    preds = p_buf / cnt

    return preds
#end def




def main():
    log_level = 'DEBUG'
    log_format = '%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    # set seed
    seed = config['seed']
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    # Load in train_df test_df
    if config['embedding']['use']:
        # if using USE, then need to load the text cols separately
        old = True
        old_train_df, continuous_features, categorical_features, string_features = read(config['train'], quick=config['quick'], old=old)
        old_test_df, _, _, _ = read(config['test'], quick=config['quick'], old=old)

        old_test_df['project_is_approved'] = prepare_nn(
            old_train_df,
            old_test_df,
            old=old,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            string_features=string_features)
        del old_train_df
        gc.collect()

        old = False
        new_train_df, continuous_features, categorical_features, string_features = read(config['train'], quick=config['quick'], old=old)
        new_test_df, _, _, _ = read(config['test'], quick=config['quick'], old=old)

        new_test_df['project_is_approved'] = prepare_nn(
            new_train_df,
            new_test_df,
            old=old,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            string_features=string_features)

        test_df = pd.concat([old_test_df, new_test_df], ignore_index=True)
        del new_train_df, old_test_df, new_test_df
    elif config['model_type']['lgbm']:
        train_df, continuous_features, categorical_features, string_features = read(config['train'], quick=config['quick'])
        test_df, _, _, _ = read(config['test'], quick=config['quick'])

        test_df['project_is_approved'] = prepare_lgbm(
            train_df,
            test_df,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            string_features=string_features)
        del train_df
    # elif config['model_type']['xgb']:
    #     train_df, continuous_features, categorical_features, string_features = read(config['train'], quick=config['quick'])
    #     test_df, _, _, _ = read(config['test'], quick=config['quick'])

    #     test_df['project_is_approved'] = prepare_lgbm(
    #         train_df,
    #         test_df,
    #         continuous_features=continuous_features,
    #         categorical_features=categorical_features,
    #         string_features=string_features)
    #     del train_df    
    else:
        train_df, continuous_features, categorical_features, string_features = read(config['train'], quick=config['quick'])
        test_df, _, _, _ = read(config['test'], quick=config['quick'])

        test_df['project_is_approved'] = prepare_nn(
            train_df,
            test_df,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            string_features=string_features)
        del train_df
    #end if
    gc.collect()

    logger.info('Writing results to: {}'.format(config['output_csv']))
    out_df = test_df[['id', 'project_is_approved']]
    out_df.to_csv(config['output_csv'], index=False)
#end def


if __name__ == "__main__": main()
