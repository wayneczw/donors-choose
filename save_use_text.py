
import pandas as pd
import joblib
import tensorflow as tf
import tensorflow_hub as hub
import gc
import numpy as np
import logging


logger = logging.getLogger(__name__)


USE_MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/2"
USE_EMBED = hub.Module(USE_MODULE_URL, trainable=False)


def get_USE(col_series, session):
    logger.info('Converting to USE....')

    try:
        use_embedded = session.run(USE_EMBED(col_series))
    except TypeError:
        use_embedded = session.run(USE_EMBED(col_series.values))
    #end try
    #end with

    return use_embedded
#end def


def main():
    log_level = 'DEBUG'
    log_format = '%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    logger.info('Loading in data')
    train_df = pd.read_csv('data/train_processed.csv')
    # train_df = train_df[:4000]
    test_df = pd.read_csv('data/test_processed.csv')
    # test_df = test_df[:4000]

    string_features = [
        'project_title',
        'project_essay_1',
        'project_essay_2',
        'project_essay_3',
        'project_essay_4',
        'project_resource_summary',
        'description']
    train_df[string_features] = train_df[string_features].fillna('')
    test_df[string_features] = test_df[string_features].fillna('')

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        use_title = get_USE(train_df['project_title'], session)
        use_essay1 = get_USE(train_df['project_essay_1'], session)
        use_essay2 = get_USE(train_df['project_essay_2'], session)
        use_essay3 = get_USE(train_df['project_essay_3'], session)
        use_essay4 = get_USE(train_df['project_essay_4'], session)
        use_resource_summary = get_USE(train_df['project_resource_summary'], session)
        use_description = get_USE(train_df['description'], session)

        train_text = np.hstack([use_title, use_essay1, use_essay2, use_essay3, use_essay4, use_resource_summary, use_description])

        use_title = get_USE(test_df['project_title'], session)
        use_essay1 = get_USE(test_df['project_essay_1'], session)
        use_essay2 = get_USE(test_df['project_essay_2'], session)
        use_essay3 = get_USE(test_df['project_essay_3'], session)
        use_essay4 = get_USE(test_df['project_essay_4'], session)
        use_resource_summary = get_USE(test_df['project_resource_summary'], session)
        use_description = get_USE(test_df['description'], session)
    #end with

    test_text = np.hstack([use_title, use_essay1, use_essay2, use_essay3, use_essay4, use_resource_summary, use_description])
    del use_title, use_essay1, use_essay2, use_essay3, use_essay4, use_resource_summary, use_description
    gc.collect()

    joblib.dump(train_text, 'data/train_use.joblib', compress=True)
    joblib.dump(test_text, 'data/test_use.joblib', compress=True)
#end def


if __name__ == '__main__': main()
