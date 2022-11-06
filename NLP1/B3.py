import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

with open("D:\\UET_Lessons\pythonProject\datasets\W2V_150.txt", encoding='utf8') as file:
    vec_data = file.readlines()

for i in range(len(vec_data)):
    vec_data[i] = vec_data[i][:-1]

vec_dict = dict()

for i in range(2, len(vec_data)):
    split_data = vec_data[i].split()
    vec_dict[split_data[0]] = np.array(split_data[1:]).astype('float64')

with open("D:\\UET_Lessons\pythonProject\datasets\Visim-400.txt", encoding='utf8') as file:
    pair_data = file.readlines()

for i in range(len(pair_data)):
    pair_data[i] = pair_data[i][:-1]

pair_dict = dict()

for i in range(1, len(pair_data)):
    split_data = pair_data[i].split()
    pair_dict[i - 1] = np.array(split_data[0:2]).astype("str")

with open('D:\\UET_Lessons\pythonProject\datasets\Antonym_vietnamese.txt', encoding='utf8') as file:
    antonym_data = file.readlines()

for i in range(len(antonym_data)):
    antonym_data[i] = antonym_data[i][:-1]

antonym_dict = dict()

for i in range(len(antonym_data)):
    split_data = antonym_data[i].split()
    antonym_dict[i] = np.array(split_data[0:2]).astype("str")

with open('D:\\UET_Lessons\pythonProject\datasets\Synonym_vietnamese.txt', encoding='utf8') as file:
    synonym_data = file.readlines()

for i in range(len(synonym_data)):
    synonym_data[i] = synonym_data[i][:-1]

synonym_dict = dict()

for i in range(len(synonym_data)):
    split_data = synonym_data[i].split()
    synonym_dict[i] = np.array(split_data[0:2]).astype("str")

X_train = []
y_train = []

for key in antonym_dict.keys():
    word1 = antonym_dict[key][0]
    word2 = antonym_dict[key][1]

    if word1 in vec_dict and word2 in vec_dict:
        X_train.append(vec_dict[word1] + vec_dict[word2])
        y_train.append(1)

for key in list(synonym_dict):
    if (synonym_dict[key].size == 2):
        word1 = synonym_dict[key][0]
        word2 = synonym_dict[key][1]

        if word1 in vec_dict and word2 in vec_dict:
            X_train.append(vec_dict[word1] + vec_dict[word2])
            y_train.append(0)


clf: CatBoostClassifier = CatBoostClassifier(task_type ='GPU')
clf.fit(X_train, y_train, verbose=False)


def test_data_generator(path):
    X_test = []
    y_test = []
    with open(path) as file:
        test_data = file.readlines();

    for i in range(len(test_data)):
        test_data[i] = test_data[i][:-1]

    for i in range(1, len(test_data)):
        split_data = test_data[i].split()
        word1 = split_data[0]
        word2 = split_data[1]

        if word1 in vec_dict and word2 in vec_dict:
            X_test.append(vec_dict[word1] + vec_dict[word2])
            if (split_data[2] == "SYN"):
                y_test.append(0)
            else:
                y_test.append(1)
    return X_test, y_test


X_test, y_test = test_data_generator("D:\\UET_Lessons\pythonProject\datasets\\400_noun_pairs.txt")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['SYN', 'ANT']))

X_test, y_test = test_data_generator("D:\\UET_Lessons\pythonProject\datasets\\400_verb_pairs.txt", encoding='utf8')
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['SYN', 'ANT']))

X_test, y_test = test_data_generator("D:\\UET_Lessons\pythonProject\datasets\\600_adj_pairs.txt", encoding='utf8')
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['SYN', 'ANT']))
