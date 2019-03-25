import pandas as pd
import calendar
import logging
import pylab as pl
import regex as re
import yaml

import multiprocessing as mp

from multiprocessing import cpu_count

from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.tag import pos_tag

from collections import Counter

from textblob import TextBlob


logger = logging.getLogger(__name__)
ncores = cpu_count()


raw_string_features = [
    'project_title', 'project_essay_1',
    'project_essay_2', 'project_essay_3',
    'project_essay_4', 'project_resource_summary',
    'description']

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

config_path = 'benchmark.yaml'

# Load in config file
with open(config_path, 'r') as f:
    config = yaml.load(f)
#end with


def get_monthofyear(x):
    month_num = x.month
    year_num = x.year
    return calendar.month_abbr[month_num] + '_' + str(year_num)
#end def


def get_dayofweek(x):
    day_num = x.dayofweek
    return calendar.day_name[day_num]
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


def get_sent_count(text):
    try:
        return len(sent_tokenize(text))
    except:
        return 0
#end def


def get_word_count(text):
    try:
        return len(word_tokenize(text))
    except:
        return 0
#end def


def get_pos_count(text):
    d = Counter([t[1] for t in pos_tag(text.split())])
    return [d[t] for t in Tags]
#end def


def get_keyword_count(text):
    d = Counter([word for word in text.split() if word in Keywords])
    return [d[k] for k in Keywords]
#end def


def get_common_word_count(text):
    tmp = text.split('[--]')
    text1 = tmp[0]
    text2 = tmp[1]

    return len(set(re.split('\W', text1.lower())).intersection(re.split('\W', text2.lower())))
#end def


def clean(text):
    
    return re.sub('[!@#$:]', '', ' '.join(re.findall('\w{3,}', str(text).lower())))
#end def


def make_multi_thread(func, series):
    with mp.Pool(ncores) as pool:
        X = pool.imap(func, series, chunksize=1)
        X = [x for x in X]
    #end with
    return pd.Series(X)
#end def


def read(resource_df_path, df_path, old=True, quick=False):
    string_features = []
    categorical_features = [
        'teacher_prefix', 'school_state',
        'project_grade_category',
        'project_subject_categories', 'project_subject_subcategories']
    continuous_features = ['teacher_number_of_previously_posted_projects']

    logger.info("Reading in data from {}....".format(resource_df_path))

    resource_df = pd.read_csv(resource_df_path)
    resource_df['id'] = resource_df['id'].astype('category')
    resource_df[['price', 'quantity']] = resource_df[['price', 'quantity']].apply(pd.to_numeric)
    resource_df['description'] = resource_df['description'].astype(str)

    resource_df['total_price'] = resource_df['price'] * resource_df['quantity']

    # get description
    resource_df_des = resource_df.groupby('id')['description'].apply('\n'.join).reset_index()

    # get diff price attributes
    resource_df_price = resource_df.groupby('id')['total_price'].sum().to_frame()  # benchmark feature
    continuous_features.append('total_price')
    resource_df_price['max_price'] = resource_df.groupby('id')['price'].max()
    continuous_features.append('max_price')
    resource_df_price['min_price'] = resource_df.groupby('id')['price'].min()
    continuous_features.append('min_price')
    resource_df_price['mean_price'] = resource_df.groupby('id')['price'].mean()
    continuous_features.append('mean_price')

    resource_df_qty = resource_df.groupby('id')['quantity'].sum().to_frame()
    resource_df = pd.merge(resource_df_des, resource_df_price, on='id', how='left')
    resource_df = pd.merge(resource_df, resource_df_qty, on='id', how='left')
    continuous_features.append('quantity')

    logger.info("Reading in data from {}....".format(df_path))

    df = pd.read_csv(df_path)
    df = pd.merge(df, resource_df, on='id', how='left')

    # get submission time - benchmark feature
    df['monthofyear'] = make_multi_thread(get_monthofyear,  pd.to_datetime(df['project_submitted_datetime']))
    df['dayofweek'] = make_multi_thread(get_dayofweek,  pd.to_datetime(df['project_submitted_datetime']))
    categorical_features.extend(['dayofweek', 'monthofyear'])

    df = df.drop(['teacher_id', 'project_submitted_datetime'], axis=1)

    if quick:
        df = df[:4000]

    # simple data format cleaning
    df[categorical_features] = df[categorical_features].apply(lambda x: x.astype('category'))
    df[continuous_features] = df[continuous_features].apply(pd.to_numeric)
    try:
        df['project_is_approved'] = df['project_is_approved'].astype('category')
    except KeyError:
        pass

    df[raw_string_features] = df[raw_string_features].fillna('').apply(lambda x: x.astype(str))

    # get sent count of essays
    logger.info('Getting sentence count')
    df['project_essay_1_sent_len'] = make_multi_thread(get_sent_count, df['project_essay_1'])
    df['project_essay_2_sent_len'] = make_multi_thread(get_sent_count, df['project_essay_2'])
    df['project_essay_3_sent_len'] = make_multi_thread(get_sent_count, df['project_essay_3'])
    df['project_essay_4_sent_len'] = make_multi_thread(get_sent_count, df['project_essay_4'])
    continuous_features.extend([
        'project_essay_1_sent_len', 'project_essay_2_sent_len',
        'project_essay_3_sent_len', 'project_essay_4_sent_len'])

    # get word count of essays
    logger.info('Getting word count')
    df['project_essay_1_word_len'] = make_multi_thread(get_word_count, df['project_essay_1'])
    df['project_essay_2_word_len'] = make_multi_thread(get_word_count, df['project_essay_2'])
    df['project_essay_3_word_len'] = make_multi_thread(get_word_count, df['project_essay_3'])
    df['project_essay_4_word_len'] = make_multi_thread(get_word_count, df['project_essay_4'])
    continuous_features.extend([
        'project_essay_1_word_len', 'project_essay_2_word_len',
        'project_essay_3_word_len', 'project_essay_4_word_len'])

    for f in raw_string_features:
        df[f] = make_multi_thread(clean, df[f])

    df['all_essays'] = df['project_essay_1'].str.cat(df[['project_essay_2', 'project_essay_3', 'project_essay_4']], sep='. ', na_rep=' ')
    string_features.append('all_essays')

    # get polarity
    logger.info('Getting polarity scores')
    df['polarity'] = make_multi_thread(get_polarity, df['all_essays'])
    continuous_features.append('polarity')

    # get subjectivity
    logger.info('Getting subjectivity scores')
    df['subjectivity'] = make_multi_thread(get_subjectivity, df['all_essays'])
    continuous_features.append('subjectivity')

    # if tfidf/wordvector, merge all text into one
    df['all_text'] = df['project_title'].str.cat(df[['all_essays', 'project_resource_summary', 'description']], sep='. ', na_rep=' ')
    string_features.append('all_text')

    # clean up continuous features
    df[continuous_features] = df[continuous_features].apply(pd.to_numeric)
   
    # get pos count for each text attribute
    logger.info('Getting POS count')
    for f in raw_string_features:
        temp = pl.array(list(make_multi_thread(get_pos_count, df[f])))

        for i, t in enumerate(Tags):
            df[f + '_' + t + '_count'] = temp[:, i]
            continuous_features.append(f + '_' + t + '_count')
        #end for
    #end for

    # get keyword count for each text attribute
    logger.info('Getting Keywords count')
    for f in raw_string_features:
        temp = pl.array(list(make_multi_thread(get_keyword_count, df[f])))

        for i, t in enumerate(Keywords):
            df[f + '_' + t + '_count'] = temp[:, i]
            continuous_features.append(f + '_' + t + '_count')
        #end for
    #end for

    # get common word count for each text attribute pair
    logger.info('Getting common words count')
    for i, f1 in enumerate(raw_string_features[:-1]):
        for f2 in raw_string_features[i+1:]:
            df[f1 + f2] = df[f1].str.cat(df[f2], sep='[--]', na_rep=' ')
            df['%s_%s_common' % (f1, f2)] = make_multi_thread(get_common_word_count, df[f1 + f2])
            continuous_features.append('%s_%s_common' % (f1, f2))
        #end for
    #end for

    logger.info("Done reading in {} data....".format(df.shape[0]))

    df['teacher_prefix'] = df['teacher_prefix'].fillna('Teacher')

    logger.info('Continuous features: \n{}'.format(continuous_features))
    logger.info('Categorical features: \n{}'.format(categorical_features))
    logger.info('String features: \n{}'.format(string_features))
    return df
#end def


def main():
    log_level = 'DEBUG'
    log_format = '%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    train_df = read(config['resources'], config['train'], quick=config['quick'])

    test_df = read(config['resources'], config['test'], quick=config['quick'])

    train_df.to_csv('./data/train_processed.csv', index=False)
    test_df.to_csv('./data/test_processed.csv', index=False)
#end def


if __name__ == '__main__': main()
