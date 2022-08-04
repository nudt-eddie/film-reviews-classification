# 定义删除除字母,数字，汉字以外的所有符号的函数
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
import jieba as jb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors



#去标点符号
def remove_punctuation(line):  #去标点符号
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line


# 加载停用词
stopwords = [line.strip() for line in open("./chineseStopWords.txt", 'r', encoding='utf-8').readlines()]
#stopwords = stopwordslist("./chineseStopWords.txt")
#print(stopwords)

data = pd.read_csv('./test_data.csv')
data=data[['userId','movieId','rating','timestamp','comment','like']]
#删除除字母,数字，汉字以外的所有符号
data['clean_comment'] = data['comment'].apply(remove_punctuation)

#分词，并过滤停用词
data['cut_comment'] = data['clean_comment'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
data.head()

#print(data.head())

#计算cut_comment的TF-IDF的特征值
tfidf = TfidfVectorizer(norm='l2', ngram_range=(1, 2))#ngram_range：词组切分的长度范围
features = tfidf.fit_transform(data.cut_comment)
labels = data.rating
print(features.shape)
#print('-----------------------------')
#print(features)

#文本特征向量化
X_train, X_test, y_train, y_test = train_test_split(data['cut_comment'], data['rating'], test_size=0.33, random_state=2)
#2000个数据做train，1000个数据作test
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test = count_vect.transform(X_test)

#朴素贝叶斯
clf = MultinomialNB().fit(X_train_tfidf, y_train)
pre = clf.predict(X_test)
print('The Accuracy of Naive Bayes Classifier is:', clf.score(X_test,y_test))
print(classification_report(y_test, pre))

#线性支持向量机
LSVC = LinearSVC().fit(X_train_tfidf, y_train)
LSVC_pre = LSVC.predict(X_test)
print('The Accuracy of LVM is:', LSVC.score(X_test,y_test))
print(classification_report(y_test, LSVC_pre))


#决策树
DTC = DecisionTreeClassifier().fit(X_train_tfidf, y_train)
DTC_pre = DTC.predict(X_test)
print('The Accuracy of Decision Tree Classifier is:', DTC.score(X_test,y_test))
print(classification_report(y_test, DTC_pre))

#随机森林
RFC = RandomForestClassifier().fit(X_train_tfidf, y_train)
RFC_pre = RFC.predict(X_test)
print('The Accuracy of Random Forest Classifier is:', RFC.score(X_test,y_test))
print(classification_report(y_test, RFC_pre))


#逻辑回归
LR = LogisticRegression().fit(X_train_tfidf, y_train)
LR_pre = LR.predict(X_test)
print('The Accuracy of LogisticRegression Classifier is:', LR.score(X_test,y_test))
print(classification_report(y_test, LR_pre))


#K-最近邻居
KNN = neighbors.KNeighborsClassifier().fit(X_train_tfidf, y_train)
KNN_pre = KNN.predict(X_test)
print ("The Accuracy of KNeighbors Classifier is:",KNN.score(X_test,y_test))
print(classification_report(y_test, KNN_pre))

#神经网络
neural_network = MLPClassifier().fit(X_train_tfidf, y_train)
neural_network_pre = neural_network.predict(X_test)
print ("The Accuracy of neural_network Classifier is:",neural_network.score(X_test,y_test))
print(classification_report(y_test, neural_network_pre))

#集成学习

bag_clf = BaggingClassifier().fit(X_train_tfidf, y_train)
bag_pre = bag_clf.predict(X_test)
print ("The Accuracy of bagging Classifier is:",bag_clf.score(X_test,y_test))
print(classification_report(y_test, bag_pre))

models = [
    BaggingClassifier(),
    neighbors.KNeighborsClassifier(),
    LogisticRegression(max_iter=10000),
    MLPClassifier(),
    RandomForestClassifier(),
    DecisionTreeClassifier(),
    LinearSVC(),
    MultinomialNB()
]

CV = 2000
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.savefig("temp.png",dpi = 500) 
plt.show()


