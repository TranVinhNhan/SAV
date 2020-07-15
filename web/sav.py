import numpy as np
import pandas as pd
from pyvi import ViTokenizer
import glob
import re
import string
import codecs
import time
import os.path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
# Algorithm
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from pathlib import Path

from flask import Flask, render_template, request, redirect


app = Flask(__name__)

path_neg = '../sentiment_dicts/neg.txt'
path_pos = '../sentiment_dicts/pos.txt'
path_not = '../sentiment_dicts/not.txt'

with codecs.open(path_neg, 'r', encoding='UTF-8') as f:
    neg = f.readlines()
neg_list = [n.replace('\n', '') for n in neg]

with codecs.open(path_pos, 'r', encoding='UTF-8') as f:
    pos = f.readlines()
pos_list = [n.replace('\n', '') for n in pos]
with codecs.open(path_not, 'r', encoding='UTF-8') as f:
    not_ = f.readlines()
not_list = [n.replace('\n', '') for n in not_]


def normalize_text(text):

    # Remove các ký tự kéo dài: vd: đẹppppppp
    text = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(),
                  text, flags=re.IGNORECASE)

    # Chuyển thành chữ thường
    text = text.lower()

    # Chuẩn hóa tiếng Việt, xử lý emoj, chuẩn hóa tiếng Anh, thuật ngữ
    replace_list = {
        'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ', 'òe': 'oè', 'óe': 'oé', 'ỏe': 'oẻ',
        'õe': 'oẽ', 'ọe': 'oẹ', 'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ', 'ụy': 'uỵ', 'uả': 'ủa',
        'ả': 'ả', 'ố': 'ố', 'u´': 'ố', 'ỗ': 'ỗ', 'ồ': 'ồ', 'ổ': 'ổ', 'ấ': 'ấ', 'ẫ': 'ẫ', 'ẩ': 'ẩ',
        'ầ': 'ầ', 'ỏ': 'ỏ', 'ề': 'ề', 'ễ': 'ễ', 'ắ': 'ắ', 'ủ': 'ủ', 'ế': 'ế', 'ở': 'ở', 'ỉ': 'ỉ',
        'ẻ': 'ẻ', 'àk': u' à ', 'aˋ': 'à', 'iˋ': 'ì', 'ă´': 'ắ', 'ử': 'ử', 'e˜': 'ẽ', 'y˜': 'ỹ', 'a´': 'á',
        # Quy các icon về 2 loại emoj: Tích cực hoặc tiêu cực
        "👹": "negative", "👻": "positive", "💃": "positive", '🤙': ' positive ', '👍': ' positive ',
        "💄": "positive", "💎": "positive", "💩": "positive", "😕": "negative", "😱": "negative", "😸": "positive",
        "😾": "negative", "🚫": "negative",  "🤬": "negative", "🧚": "positive", "🧡": "positive", '🐶': ' positive ',
        '👎': ' negative ', '😣': ' negative ', '✨': ' positive ', '❣': ' positive ', '☀': ' positive ',
        '♥': ' positive ', '🤩': ' positive ', 'like': ' positive ', '💌': ' positive ',
        '🤣': ' positive ', '🖤': ' positive ', '🤤': ' positive ', ':(': ' negative ', '😢': ' negative ',
        '❤': ' positive ', '😍': ' positive ', '😘': ' positive ', '😪': ' negative ', '😊': ' positive ',
        '?': ' ? ', '😁': ' positive ', '💖': ' positive ', '😟': ' negative ', '😭': ' negative ',
        '💯': ' positive ', '💗': ' positive ', '♡': ' positive ', '💜': ' positive ', '🤗': ' positive ',
        '^^': ' positive ', '😨': ' negative ', '☺': ' positive ', '💋': ' positive ', '👌': ' positive ',
        '😖': ' negative ', '😀': ' positive ', ':((': ' negative ', '😡': ' negative ', '😠': ' negative ',
        '😒': ' negative ', '🙂': ' positive ', '😏': ' negative ', '😝': ' positive ', '😄': ' positive ',
        '😙': ' positive ', '😤': ' negative ', '😎': ' positive ', '😆': ' positive ', '💚': ' positive ',
        '✌': ' positive ', '💕': ' positive ', '😞': ' negative ', '😓': ' negative ', '️🆗️': ' positive ',
        '😉': ' positive ', '😂': ' positive ', ':v': '  positive ', '=))': '  positive ', '😋': ' positive ',
        '💓': ' positive ', '😐': ' negative ', ':3': ' positive ', '😫': ' negative ', '😥': ' negative ',
        '😃': ' positive ', '😬': ' 😬 ', '😌': ' 😌 ', '💛': ' positive ', '🤝': ' positive ', '🎈': ' positive ',
        '😗': ' positive ', '🤔': ' negative ', '😑': ' negative ', '🔥': ' negative ', '🙏': ' negative ',
        '🆗': ' positive ', '😻': ' positive ', '💙': ' positive ', '💟': ' positive ',
        '😚': ' positive ', '❌': ' negative ', '👏': ' positive ', ';)': ' positive ', '<3': ' positive ',
        '🌝': ' positive ',  '🌷': ' positive ', '🌸': ' positive ', '🌺': ' positive ',
        '🌼': ' positive ', '🍓': ' positive ', '🐅': ' positive ', '🐾': ' positive ', '👉': ' positive ',
        '💐': ' positive ', '💞': ' positive ', '💥': ' positive ', '💪': ' positive ',
        '💰': ' positive ',  '😇': ' positive ', '😛': ' positive ', '😜': ' positive ',
        '🙃': ' positive ', '🤑': ' positive ', '🤪': ' positive ', '☹': ' negative ',  '💀': ' negative ',
        '😔': ' negative ', '😧': ' negative ', '😩': ' negative ', '😰': ' negative ', '😳': ' negative ',
        '😵': ' negative ', '😶': ' negative ', '🙁': ' negative ', '😅': 'negative',
        # Chuẩn hóa 1 số sentiment words/English words
        ':))': '  positive ', ':)': ' positive ', 'ô kêi': ' ok ', 'okie': ' ok ', ' o kê ': ' ok ', '💣': 'nhiệt', '🍺': 'bia',
        'okey': ' ok ', 'ôkê': ' ok ', 'oki': ' ok ', ' oke ':  ' ok ', ' okay': ' ok ', 'okê': ' ok ',
        ' tks ': u' cám ơn ', 'thks': u' cám ơn ', 'thanks': u' cám ơn ', 'ths': u' cám ơn ', 'thank': u' cám ơn ',
        '⭐': 'star ', '*': 'star ', '🌟': 'star ', '🎉': u' positive ',
        'kg ': u' không ', 'not': u' không ', u' kg ': u' không ', '"k ': u' không ', ' kh ': u' không ', 'kô': u' không ', 'hok': u' không ', ' kp ': u' không phải ', u' kô ': u' không ', '"ko ': u' không ', u' ko ': u' không ', u' k ': u' không ', 'khong': u' không ', u' hok ': u' không ',
        'he he': ' positive ', 'hehe': ' positive ', 'hihi': ' positive ', 'haha': ' positive ', 'hjhj': ' positive ',
        ' lol ': ' negative ', ' cc ': ' negative ', 'cute': u' dễ thương ', 'huhu': ' negative ', ' vs ': u' với ', 'wa': ' quá ', 'wá': u' quá', 'j': u' gì ', '“': ' ',
        ' sz ': u' cỡ ', 'size': u' cỡ ', u' đx ': u' được ', 'dk': u' được ', 'dc': u' được ', 'đk': u' được ',
        'đc': u' được ', 'authentic': u' chuẩn chính hãng ', u' aut ': u' chuẩn chính hãng ', u' auth ': u' chuẩn chính hãng ', 'thick': u' positive ', 'store': u' cửa hàng ',
        'shop': u' cửa hàng ', 'sp': u' sản phẩm ', 'gud': u' tốt ', 'god': u' tốt ', 'wel done': ' tốt ', 'good': u' tốt ', 'gút': u' tốt ',
        'sấu': u' xấu ', 'gut': u' tốt ', u' tot ': u' tốt ', u' nice ': u' tốt ', 'perfect': 'rất tốt', 'bt': u' bình thường ',
        'time': u' thời gian ', 'qá': u' quá ', u' ship ': u' giao hàng ', u' m ': u' mình ', u' mik ': u' mình ',
        'ể': 'ể', 'product': 'sản phẩm', 'quality': 'chất lượng', 'chat': ' chất ', 'excelent': 'hoàn hảo', 'bad': 'tệ', 'fresh': ' tươi ', 'sad': ' tệ ',
        'date': u' hạn sử dụng ', 'hsd': u' hạn sử dụng ', 'quickly': u' nhanh ', 'quick': u' nhanh ', 'fast': u' nhanh ', 'delivery': u' giao hàng ', u' síp ': u' giao hàng ',
        'beautiful': u' đẹp tuyệt vời ', u' tl ': u' trả lời ', u' r ': u' rồi ', u' shopE ': u' cửa hàng ', u' order ': u' đặt hàng ',
        'chất lg': u' chất lượng ', u' sd ': u' sử dụng ', u' dt ': u' điện thoại ', u' nt ': u' nhắn tin ', u' tl ': u' trả lời ', u' sài ': u' xài ', u'bjo': u' bao giờ ',
        'thik': u' thích ', u' sop ': u' cửa hàng ', ' fb ': ' facebook ', ' face ': ' facebook ', ' very ': u' rất ', u'quả ng ': u' quảng  ',
        'dep': u' đẹp ', u' xau ': u' xấu ', 'delicious': u' ngon ', u'hàg': u' hàng ', u'qủa': u' quả ',
        'iu': u' yêu ', 'fake': u' giả mạo ', 'trl': 'trả lời', '><': u' positive ',
        ' por ': u' tệ ', ' poor ': u' tệ ', 'ib': u' nhắn tin ', 'rep': u' trả lời ', u'fback': ' feedback ', 'fedback': ' feedback ',
        # dưới 3* quy về 1*, trên 3* quy về 5*
        '6 sao': ' 5star ', '6 star': ' 5star ', '5star': ' 5star ', '5 sao': ' 5star ', '5sao': ' 5star ',
        'starstarstarstarstar': ' 5star ', '1 sao': ' 1star ', '1sao': ' 1star ', '2 sao': ' 1star ', '2sao': ' 1star ',
        '2 starstar': ' 1star ', '1star': ' 1star ', '0 sao': ' 1star ', '0star': ' 1star ', }

    for k, v in replace_list.items():
        text = text.replace(k, v)

    # chuyen punctuation thành space
    translator = str.maketrans(
        string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)

    text = ViTokenizer.tokenize(text)
    texts = text.split()
    len_text = len(texts)

    texts = [t.replace('_', ' ') for t in texts]
    for i in range(len_text):
        cp_text = texts[i]
        # Xử lý vấn đề phủ định (VD: áo này chẳng đẹp--> áo này notpos)
        if cp_text in not_list:
            numb_word = 2 if len_text - i - 1 >= 4 else len_text - i - 1

            for j in range(numb_word):
                if texts[i + j + 1] in pos_list:
                    texts[i] = 'notpos'
                    texts[i + j + 1] = ''

                if texts[i + j + 1] in neg_list:
                    texts[i] = 'notneg'
                    texts[i + j + 1] = ''
        # Thêm feature cho những sentiment words (áo này đẹp--> áo này đẹp positive)
        else:
            if cp_text in pos_list:
                texts.append('positive')
            elif cp_text in neg_list:
                texts.append('negative')

    text = u' '.join(texts)

    # remove nốt những ký tự thừa thãi
    text = text.replace(u'"', u' ')
    text = text.replace(u'️', u'')
    text = text.replace('🏻', '')
    return text

# Load dataset từ file nếu đã export
if os.path.exists(r'C:\Users\VinhNhan\Desktop\friday\sav_train.csv') and os.path.exists(r'C:\Users\VinhNhan\Desktop\friday\sav_test.csv'):
    print('not null')
    df_train = pd.read_csv(r'C:\Users\VinhNhan\Desktop\friday\sav_train.csv')
    df_test = pd.read_csv(r'C:\Users\VinhNhan\Desktop\friday\sav_test.csv')
else:
    print('null')
    # DataFrame của train dataset
    df_train = pd.DataFrame()

    # DataFrame của test dataset
    df_test = pd.DataFrame()

    # Negative và positive comments dataset
    neg_comments = []
    pos_comments = []

    # Negative và positive comments test dataset
    neg_test_comments = []
    pos_test_comments = []

    # Lấy negative comments trong Train dataset
    neg_paths = glob.glob("../train/neg/*.txt")
    for path in neg_paths:
        with open(path, encoding="utf-8") as file:
            text = file.read()
            text_lower = text.lower()
            text_clean = normalize_text(text_lower)
            text_token = ViTokenizer.tokenize(text_clean)
            neg_comments.append(text_token)
        file.close()

    # Lấy positive comments trong Test dataset
    pos_paths = glob.glob("../train/pos/*.txt")
    for path in pos_paths:
        with open(path, encoding="utf-8") as file:
            text= file.read()
            text_lower = text.lower()
            text_clean = normalize_text(text_lower)
            text_token = ViTokenizer.tokenize(text_clean)
            pos_comments.append(text_token)
        file.close()

    for comment in neg_comments :
        df_train = df_train.append({'text' : comment, 'sentiment' : 'negative'}, ignore_index=True)
    for comment in pos_comments :
        df_train = df_train.append({'text' : comment, 'sentiment' : 'positive'}, ignore_index=True)


    # Lấy negative comments trong Test dataset
    neg_test_paths = glob.glob("../test/neg/*.txt")
    for path in neg_test_paths :
        with open(path,encoding="utf-8") as file:
            text= file.read()
            text_lower = text.lower()
            text_clean = normalize_text(text_lower)
            text_token = ViTokenizer.tokenize(text_clean)
            neg_test_comments.append(text_token)
        file.close()

    # Lấy positive comments trong Train dataset
    pos_test_paths = glob.glob("../test/pos/*.txt")
    for path in pos_test_paths :
        with open(path,encoding="utf-8") as file:
            text= file.read()
            text_lower = text.lower()
            text_clean = normalize_text(text_lower)
            text_token = ViTokenizer.tokenize(text_clean)
        pos_test_comments.append(text_token)
        file.close()


    for comment in neg_test_comments :
        df_test = df_test.append({'text' : comment, 'sentiment' : 'negative'}, ignore_index=True)
    for comment in pos_test_comments :
        df_test = df_test.append({'text' : comment, 'sentiment' : 'positive'}, ignore_index=True)

    # Export ra file
    df_train.to_csv(r'C:\Users\VinhNhan\Desktop\friday\sav_train.csv', index=False, header=True)
    df_test.to_csv(r'C:\Users\VinhNhan\Desktop\friday\sav_test.csv', index=False, header=True)


#_________________________________________________________________________________________________________#
train_features = pd.Series(df_train['text'].values)
test_features = pd.Series(df_test['text'].values)
train_labels = pd.Series(df_train['sentiment'].values)
test_labels = pd.Series(df_test['sentiment'].values)

stop_word = []
with open("../stop_word.txt",encoding="utf-8") as f :
    text = f.read()
    for word in text.split() :
        stop_word.append(word)
    f.close()

#max_features=2500 nghĩa là dùng 2500 từ xuất hiện nhiều nhất
#max_df=0.8 nghĩa là bỏ những từ xuất hiện nhiều hơn 80% trong tập dữ liệu
#min_df=7 nghĩa là bỏ những từ xuất hiện nhỏ hơn 7 lần trong tập dữ liệu
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True, stop_words=stop_word)
train_vectors = vectorizer.fit_transform(train_features)
test_vectors = vectorizer.transform(test_features)

#___________________________________________SVM___________________________________________________________
# Load trained model từ file nếu đã export
filename_svm = 'svm_classifier.joblib.pkl'
svm_model = Path(r'C:\Users\VinhNhan\Desktop\friday\web\svm_classifier.joblib.pkl')
if svm_model.is_file():
    print('Model loaded from directory')
    classifier_linear = joblib.load(filename_svm)
    # results
    print('Result of SVM Algorithm: ')
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_predict = t2-t1
    # results
    print("Prediction time: %fs" % (time_linear_predict))
    svm_report = classification_report(test_labels, prediction_linear, output_dict=True)
    print('positive: ', svm_report['positive'])
    print('negative: ', svm_report['negative'])
    print('accuracy: ' + str(accuracy_score(test_labels, prediction_linear))) #0.8877
    print('score: 0.9270333333333334') # print('score: ' + str(classifier_linear.score(train_vectors, train_labels))) #0.9270333333333334
    print('****************************************************************************************')
else:
    print('null')
    classifier_linear = svm.SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1-t0
    time_linear_predict = t2-t1
    # results
    print('Result of SVM Algorithm: ')
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    svm_report = classification_report(test_labels, prediction_linear, output_dict=True)
    print('positive: ', svm_report['positive'])
    print('negative: ', svm_report['negative'])
    print('accuracy: ' + str(accuracy_score(test_labels, prediction_linear)))
    print('score: ' + str(classifier_linear.score(train_vectors, train_labels))) #0.9270333333333334
    print('****************************************************************************************')
    # Export Model
    _ = joblib.dump(classifier_linear, filename_svm, compress=9)

#________________________________________________LogisticRegression___________________________________________________________
filename_lr = 'lr_classifier.joblib.pkl'
lr_model = Path(r'C:\Users\VinhNhan\Desktop\friday\web\lr_classifier.joblib.pkl')
if lr_model.is_file():
    print('Model loaded from directory')
    lr_clf = joblib.load(filename_lr)
    t1 = time.time()
    prediction_lr = lr_clf.predict(test_vectors)
    t2 = time.time()
    time_lr_predict = t2-t1
    # results
    print('Result of LogisticRegression Algorithm: ')
    print("Prediction time: %fs" % (time_lr_predict))
    lr_report = classification_report(test_labels, prediction_lr, output_dict=True)
    print('positive: ', lr_report['positive'])
    print('negative: ', lr_report['negative'])
    print('accuracy: ' + str(accuracy_score(test_labels, prediction_lr))) #0.888
    print('score: ' + str(lr_clf.score(train_vectors, train_labels))) #0.9143
    print('****************************************************************************************')
else:
    print('null')
    lr_clf = LogisticRegression()
    t0 = time.time()
    lr_clf.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_lr = lr_clf.predict(test_vectors)
    t2 = time.time()
    time_lr_train = t1-t0
    time_lr_predict = t2-t1
    # results
    print('Result of LogisticRegression Algorithm: ')
    print("Training time: %fs; Prediction time: %fs" % (time_lr_train, time_lr_predict))
    lr_report = classification_report(test_labels, prediction_lr, output_dict=True)
    print('positive: ', lr_report['positive'])
    print('negative: ', lr_report['negative'])
    print('accuracy: ' + str(accuracy_score(test_labels, prediction_lr)))
    print('score: ' + str(lr_clf.score(train_vectors, train_labels))) #0.9143
    print('****************************************************************************************')
    # Export Model
    _ = joblib.dump(lr_clf, filename_lr, compress=9)

#____________________________________________________________________SGDClassifier________________________________________________________________________
filename_sgdc = 'sgdc_classifier.joblib.pkl'
sgdc_model = Path(r'C:\Users\VinhNhan\Desktop\friday\web\sgdc_classifier.joblib.pkl')
if sgdc_model.is_file():
    print('Model loaded from directory')
    sgdc_clf = joblib.load(filename_sgdc)
    t1 = time.time()
    prediction_sgdc = sgdc_clf.predict(test_vectors)
    t2 = time.time()
    time_sgdc_predict = t2-t1
    # results
    print('Result of SGDClassifier Algorithm: ')
    print("Prediction time: %fs" % (time_sgdc_predict))
    sgdc_report = classification_report(test_labels, prediction_sgdc, output_dict=True)
    print('positive: ', sgdc_report['positive'])
    print('negative: ', sgdc_report['negative'])
    print('accuracy: ' + str(accuracy_score(test_labels, prediction_sgdc)))
    print('score: ' + str(sgdc_clf.score(train_vectors, train_labels))) #0.9138
    print('****************************************************************************************')
else:
    print('null')
    sgdc_clf = SGDClassifier()
    t0 = time.time()
    sgdc_clf.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_sgdc = sgdc_clf.predict(test_vectors)
    t2 = time.time()
    time_sgdc_train = t1-t0
    time_sgdc_predict = t2-t1
    # results
    print('Result of SGDClassifier Algorithm: ')
    print("Training time: %fs; Prediction time: %fs" % (time_sgdc_train, time_sgdc_predict))
    sgdc_report = classification_report(test_labels, prediction_sgdc, output_dict=True)
    print('positive: ', sgdc_report['positive'])
    print('negative: ', sgdc_report['negative'])
    print('accuracy: ' + str(accuracy_score(test_labels, prediction_sgdc)))
    print('score: ' + str(sgdc_clf.score(train_vectors, train_labels))) #0.9138
    print('****************************************************************************************')
    # Export Model
    _ = joblib.dump(sgdc_clf, filename_sgdc, compress=9)

#____________________________________________________________________DecisionTreeClassifier________________________________________________________________________
filename_dtc = 'decisiontree_classifier.joblib.pkl'
dtc_model = Path(r'C:\Users\VinhNhan\Desktop\friday\web\decisiontree_classifier.joblib.pkl')
if dtc_model.is_file():
    print('Model loaded from directory')
    decisionTree_clf = joblib.load(filename_dtc)
    t1 = time.time()
    prediction_dtc = decisionTree_clf.predict(test_vectors)
    t2 = time.time()
    time_dtc_predict = t2-t1
    # results
    print('Result of DecisionTreeClassifier Algorithm: ')
    print("Prediction time: %fs" % (time_dtc_predict))
    dtc_report = classification_report(test_labels, prediction_dtc, output_dict=True)
    print('positive: ', dtc_report['positive'])
    print('negative: ', dtc_report['negative'])
    print('accuracy: ' + str(accuracy_score(test_labels, prediction_dtc)))
    print('score: ' + str(decisionTree_clf.score(train_vectors, train_labels))) #0.9138
    print('****************************************************************************************')
else:
    print('null')
    decisionTree_clf = DecisionTreeClassifier()
    t0 = time.time()
    decisionTree_clf.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_dtc = decisionTree_clf.predict(test_vectors)
    t2 = time.time()
    time_dtc_train = t1-t0
    time_dtc_predict = t2-t1
    # results
    print('Result of DecisionTreeClassifier Algorithm: ')
    print("Training time: %fs; Prediction time: %fs" % (time_dtc_train, time_dtc_predict))
    dtc_report = classification_report(test_labels, prediction_dtc, output_dict=True)
    print('positive: ', dtc_report['positive'])
    print('negative: ', dtc_report['negative'])
    print('accuracy: ' + str(accuracy_score(test_labels, prediction_dtc)))
    print('score: ' + str(decisionTree_clf.score(train_vectors, train_labels))) #0.9138
    print('****************************************************************************************')
    # Export Model
    _ = joblib.dump(decisionTree_clf, filename_dtc, compress=9)

#____________________________________________________________________MultinomialNB________________________________________________________________________
filename_mnb = 'mnb_classifier.joblib.pkl'
mnb_model = Path(r'C:\Users\VinhNhan\Desktop\friday\web\mnb_classifier.joblib.pkl')
if mnb_model.is_file():
    print('Model loaded from directory')
    mnb_clf = joblib.load(filename_mnb)
    t1 = time.time()
    prediction_mnb = mnb_clf.predict(test_vectors)
    t2 = time.time()
    time_mnb_predict = t2-t1
    # results
    print('Result of MultinomialNB Algorithm: ')
    print("Prediction time: %fs" % (time_mnb_predict))
    mnb_report = classification_report(test_labels, prediction_mnb, output_dict=True)
    print('positive: ', mnb_report['positive'])
    print('negative: ', mnb_report['negative'])
    print('accuracy: ' + str(accuracy_score(test_labels, prediction_mnb)))
    print('score: ' + str(mnb_clf.score(train_vectors, train_labels)))
    print('****************************************************************************************')
else:
    print('null')
    mnb_clf = MultinomialNB()
    t0 = time.time()
    mnb_clf.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_mnb = mnb_clf.predict(test_vectors)
    t2 = time.time()
    time_mnb_train = t1-t0
    time_mnb_predict = t2-t1
    # results
    print('Result of MultinomialNB Algorithm: ')
    print("Training time: %fs; Prediction time: %fs" % (time_mnb_train, time_mnb_predict))
    mnb_report = classification_report(test_labels, prediction_mnb, output_dict=True)
    print('positive: ', mnb_report['positive'])
    print('negative: ', mnb_report['negative'])
    print('accuracy: ' + str(accuracy_score(test_labels, prediction_mnb)))
    print('score: ' + str(mnb_clf.score(train_vectors, train_labels))) #0.9138
    print('****************************************************************************************')
    # Export Model
    _ = joblib.dump(mnb_clf, filename_mnb, compress=9)
    
# Website
def algo_switch(algo, sentence_vector):
    return {
        'svm': classifier_linear.predict(sentence_vector),
        'lr': lr_clf.predict(sentence_vector),
        'sgd': sgdc_clf.predict(sentence_vector),
        'dt': decisionTree_clf.predict(sentence_vector),
        'mnb': mnb_clf.predict(sentence_vector)
    }.get(algo, lr_clf.predict(sentence_vector))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        svm_acc = str(accuracy_score(test_labels, prediction_linear))
        lr_acc = str(accuracy_score(test_labels, prediction_lr))
        sgd_acc = str(accuracy_score(test_labels, prediction_sgdc))
        dt_acc = str(accuracy_score(test_labels, prediction_dtc))
        multiNB_acc = str(accuracy_score(test_labels, prediction_mnb))

        return render_template('sentiment.html', svm_acc=svm_acc, lr_acc=lr_acc, sgd_acc=sgd_acc, dt_acc=dt_acc, multiNB_acc=multiNB_acc)
    if request.method == 'POST':
        svm_acc = str(accuracy_score(test_labels, prediction_linear))
        lr_acc = str(accuracy_score(test_labels, prediction_lr))
        sgd_acc = str(accuracy_score(test_labels, prediction_sgdc))
        dt_acc = str(accuracy_score(test_labels, prediction_dtc))
        multiNB_acc = str(accuracy_score(test_labels, prediction_mnb))
        
        selected = request.form['algo']
        sentences = []
        sentence = request.form['sentiment']
        sentences.append(sentence)
        sentence_vector = vectorizer.transform([sentences[0]])
        output = algo_switch(selected, sentence_vector)
        result = None
        if str(output) == '[\'negative\']':
            result = 'Tiêu cực'
        if str(output) == '[\'positive\']':
            result = 'Tích cực'
        return render_template('sentiment.html', output=result, selected=selected,  svm_acc=svm_acc, lr_acc=lr_acc, sgd_acc=sgd_acc, dt_acc=dt_acc, multiNB_acc=multiNB_acc)
    return render_template('sentiment.html')