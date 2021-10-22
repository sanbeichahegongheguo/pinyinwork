import numpy as np
import os
import sys
import fileinput
import json
import re

pinyin2hanzi_dict = {}  # 拼音到字符的dict {pinyin: [chs]}
word_onhead_probability_logarithm = {}  # 某字符出现在句首的概率对数 {str: float}
singleword_count = {}  # 字符出现计数 {str: int}
doublepair_count = {}  # 字符的二元出现次数，结构为两层嵌套字典 {str: {str: int}}

singleword_total = 396468407


# 加载数据
def load():
    global pinyin2hanzi_dict
    global word_onhead_probability_logarithm
    global singleword_count
    global doublepair_count

    with open('../data/pch.txt') as f:
        pinyin2hanzi_dict = eval(f.read())

    with open('../data/fir_p.txt') as f:
        word_onhead_probability_logarithm = eval(f.read())

    with open('../data/sin_count.txt') as f:
        singleword_count = eval(f.read())

    with open('../data/dou_count.json') as f:
        doublepair_count = json.load(fp=f)


# 预处理数据，清洗新浪语料理非汉字字符
def preload_sentences():
    path = '../data/sina_news_gbk/'

    for file in os.listdir(path):
        if file[0] == '.':
            continue

        with open('../data/all_primer_out_sentences.txt',
                  'a') as all_primer_out:

            with open(path + file) as f:
                for line in f.readlines():
                    str_html = eval(line)['html']
                    pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|' \
                              r'\(|\)|-|=|\_|\+|，|。|、|；|‘|’|' \
                              r'【|】|·|！| |…|（|）|：|？|!|“|”|【|】|『|』|{|}|《|》'
                    all_segs = [
                        x for x in re.split(pattern, str_html)
                        if len(x) > 1 and ord(x[0]) > 128
                    ]
                    for s in all_segs:
                        all_primer_out.write(s + '\n')


def preload():
    # 计算单个汉字出现次数，获取字典 singleword_count
    def addone(dict, key):
        if key in dict:
            dict[key] += 1
        else:
            dict[key] = 1

    # # 计算两个汉字对出现次数，获取字典 doublepair_count
    def add2(dict, ch1, ch2):
        if ch1 in dict:
            d = dict[ch1]
            if ch2 in d:
                d[ch2] += 1
            else:
                d[ch2] = 1
        else:
            dict[ch1] = {ch2: 1}

    fir_count = {}
    fir_tot = 0

    # 开始计算 新浪语料 里所有的汉字出现频次
    for line in fileinput.input(['../data/sentences.txt']):
        addone(fir_count, line[0])
        fir_tot += 1

        # 这步是原作者用来显示代码执行进度的，可以注释掉。
        # if fir_tot % 100000 == 0:
        #     print(fir_tot)

        # 计算每行里单个汉字出现的频次
        for ch in line:
            if ch != '\n':
                addone(singleword_count, ch)

        # 计算改行文本里的两个汉字对出现概率
        line_length = len(line)
        for i in range(line_length - 2):
            add2(doublepair_count, line[i], line[i + 1])

    # 把新浪语料 里所有的汉字出现频次改为概率对数
    for ch in fir_count:
        word_onhead_probability_logarithm[ch] = np.log(1.0 * fir_count[ch] /
                                                       fir_tot)
    with open('../data/fir_p.txt', 'w') as f:
        f.write(str(word_onhead_probability_logarithm))

    with open('../data/sin_count.txt', 'w') as f:
        f.write(str(singleword_count))

    with open('../data/dou_count.json', 'w') as f:
        json.dump(doublepair_count, f)


class node():
    def __init__(self, char, probability, prevchar):
        self.char = char
        self.probability = probability
        self.prevchar = prevchar


# 隐马尔科夫链，计算第一个字和第二字组成的二元汉字对出现的概率。
# lam 是一个人为设定的参数，是为了照顾到二元组出现次数为0的情况导致除数为0，对概率进行的平滑处理。
# 返回的是 概率对数值。
def get_2Pair_Probability(ch1, ch2, lam):
    dd = {}  # 用来放置两个汉字双字典的第二个字典
    # get方法会返回指定键的值，如果值不在 字典 中，则返回默认值。这里设置为返回0。
    doublecount = doublepair_count.get(ch1, dd).get(ch2, 0)
    single_1_count = singleword_count.get(ch1, 0)

    if single_1_count > 0:
        single_2_count = singleword_count.get(ch2, 0)
        res = np.log(lam * doublecount / single_1_count +
                     (1 - lam) * single_2_count / singleword_total)
    else:
        res = -50

    return res


def Viterbi(pylist, lam=0.99):
    for py in pylist:
        if py not in pinyin2hanzi_dict:
            return ['Wrong piyin']
    nodes = []

    # first layer
    nodes.append([
        node(x, word_onhead_probability_logarithm.get(x, -25.0), None)
        for x in pinyin2hanzi_dict[pylist[0]]
    ])

    # middle layers
    for i in range(len(pylist)):
        if i == 0:
            continue

        nodes.append([node(x, 0, None) for x in pinyin2hanzi_dict[pylist[i]]])

        for nd in nodes[i]:
            nd.pr = nodes[i - 1][0].pr + get_2Pair_Probability(
                nodes[i - 1][0].ch, nd.ch, lam)
            nd.prev = nodes[i - 1][0]
            for prend in nodes[i - 1]:
                if prend.pr + get_2Pair_Probability(prend.ch, nd.ch,
                                                    lam) > nd.pr:
                    nd.pr = prend.pr + get_2Pair_Probability(
                        prend.ch, nd.ch, lam)
                    nd.prev = prend

    # back propagation
    nd = max(nodes[-1], key=lambda x: x.pr)
    chs = []
    while nd is not None:
        chs.append(nd.ch)
        nd = nd.prev
    return list(reversed(chs))


def pinyin2hanzi(str, lam):
    return ''.join(Viterbi(str.lower().split(), lam))


def main(input, output='../data/output.txt', lam=0.9):
    chcount = 0
    chcorrect = 0
    sencount = 0
    sencorrect = 0

    with open(input) as f:
        lines = [line for line in f]

    pys = ''
    chs = ''
    mychs = ''

    with open(output, 'w') as f:
        for i in range(len(lines)):
            if i % 2 == 0:
                pys = lines[i]
            else:
                chs = lines[i]
                mychs = pinyin2hanzi(pys, lam)
                f.write(pys + mychs + '\n')

                if chs[:len(mychs)] == mychs:
                    sencorrect += 1
                sencount += 1

                for j in range(len(mychs)):
                    if chs[j] == mychs[j]:
                        chcorrect += 1
                    chcount += 1

        print('Sentences:{}, Correct sentences:{}, Correct rate:{}%'.format(
            sencount, sencorrect, round(100.0 * sencorrect / sencount, 2)))
        print('Characters:{},Correct characters:{}, Correct rate:{}%'.format(
            chcount, chcorrect, round(100.0 * chcorrect / chcount, 2)))


# 课程测试用
def test_class(input, output='../data/output.txt'):
    with open(input) as f:
        lines = [line for line in f]

    with open(output, 'w') as f:
        for i in range(len(lines)):
            pys = lines[i]
            mychs = pinyin2hanzi(pys)
            f.write(mychs + '\n')


if __name__ == '__main__':
    # load()
    preload()

    print('Pinyin(2-gram) is loading data...٩(๑>◡<๑)۶')
    load()
    print('Begin test...ヾ(=･ω･=)o')
    if len(sys.argv) == 3:
        test_class(sys.argv[1], sys.argv[2])
    else:
        print('Wrong form.')
