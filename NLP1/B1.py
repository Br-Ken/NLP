import io
import numpy as np
import os

with open('D:\\UET_Lessons\pythonProject\datasets\W2V_150.txt', encoding='utf8' ) as file:
    vec_data = file.readlines()

for i in range(len(vec_data)):
    vec_data[i] = vec_data[i][:-1]

vec_dict = dict()

for i in range(2, len(vec_data)):
    vi_data = vec_data[i].split()
    vec_dict[vi_data[0]] = np.array(vi_data[1:]).astype('float64')


with open('D:\\UET_Lessons\pythonProject\datasets\Visim-400.txt', encoding='utf8') as file :
    vi_data = file.readlines()

for i in range(len(vi_data)):
    vi_data[i] = vi_data[i][:-1]
pair_dict =dict()

for i in range(1, len(vi_data)):
    split_vi_data = vi_data[i].split()
    pair_dict[i-1] = np.array(split_vi_data[0:2]).astype("str")


def cosine_similarity(vec1, vec2):
    n = np.dot(vec1, vec2)

    vec1_n = np.sqrt(np.sum(vec1 ** 2))
    vec2_n = np.sqrt(np.sum(vec2 ** 2))

    d = vec1_n * vec2_n

    cosine_similarity = n / d

    return cosine_similarity

print('Word1 Word2 cosine_similarity')
for i in range(len(vi_data) - 1):
    word1 = pair_dict[i][0]
    word2 = pair_dict[i][1]
    if (word1 in vec_dict and word2 in vec_dict):
        print(word1 + ' ' + word2, end=" ")
        print(cosine_similarity(vec_dict[word1], vec_dict[word2]))
