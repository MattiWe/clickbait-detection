import json
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from collections import Counter
import collections

annotations_new = None
annotations_old = None
id_pub_list = None

def read_input_files():
    global annotations_new
    global annotations_old
    global id_pub_list
    with open("clickbait17-validation-170630/truth.jsonl", "r") as inf:
        annotations_new = [json.loads(x) for x in inf.readlines()]

    with open("id_pub_dict.jsonl", "r") as inf:
        id_pub_list = [json.loads(x) for x in inf.readlines()]

    with open("clickbait17-train-170331/truth.jsonl", "r") as inf:
        annotations_old = [json.loads(x) for x in inf.readlines()]

def print_majority_distribution():
    # majority matrix absolute, new dataset
    majority_matrix = {'0.6':[0,0,0,0],'0.0':[0,0,0,0],'1.0':[0,0,0,0],'0.3':[0,0,0,0]}
    for a in annotations_new:
        judge = a['truthJudgments']
        mode = a['truthMode']
        counter = Counter(judge)
        most_common = counter.most_common()
        if (most_common[0][0] == mode):
            majority_matrix[str(most_common[0][0])[:3]][(most_common[0][1])-2] += 1
        else:
            majority_matrix[str(most_common[1][0])[:3]][(most_common[1][1])-2] += 1

    print("     |   2\t  3\t  4\t  5")
    print("--------------------------------------------")
    print("0.0  | " + str(majority_matrix['0.0'][0]) + "\t" + str(majority_matrix['0.0'][1])+ "\t" + str(majority_matrix['0.0'][2])+ "\t" + str(majority_matrix['0.0'][3]))
    print("0.3  | " + str(majority_matrix['0.3'][0]) + "\t" + str(majority_matrix['0.3'][1])+ "\t" + str(majority_matrix['0.3'][2])+ "\t" + str(majority_matrix['0.3'][3]))
    print("0.6  | " + str(majority_matrix['0.6'][0]) + "\t" + str(majority_matrix['0.6'][1])+ "\t" + str(majority_matrix['0.6'][2])+ "\t" + str(majority_matrix['0.6'][3]))
    print("1.0  | " + str(majority_matrix['1.0'][0]) + "\t" + str(majority_matrix['1.0'][1])+ "\t" + str(majority_matrix['1.0'][2])+ "\t" + str(majority_matrix['1.0'][3]))

    # majority matrix absolute, old dataset
    majority_matrix = {'0.6':[0,0,0,0],'0.0':[0,0,0,0],'1.0':[0,0,0,0],'0.3':[0,0,0,0]}

    for a in annotations_old:
        judge = a['truthJudgments']
        mode = a['truthMode']
        counter = Counter(judge)
        most_common = counter.most_common()
        if (most_common[0][0] == mode):
            majority_matrix[str(most_common[0][0])[:3]][max(most_common[0][1], 5)-2] += 1
        else:
            majority_matrix[str(most_common[1][0])[:3]][max(most_common[1][1], 5)-2] += 1

    print("     |   2\t  3\t  4\t  5+")
    print("--------------------------------------------")
    print("0.0  | " + str(majority_matrix['0.0'][0]) + "\t" + str(majority_matrix['0.0'][1])+ "\t" + str(majority_matrix['0.0'][2])+ "\t" + str(majority_matrix['0.0'][3]))
    print("0.3  | " + str(majority_matrix['0.3'][0]) + "\t" + str(majority_matrix['0.3'][1])+ "\t" + str(majority_matrix['0.3'][2])+ "\t" + str(majority_matrix['0.3'][3]))
    print("0.6  | " + str(majority_matrix['0.6'][0]) + "\t" + str(majority_matrix['0.6'][1])+ "\t" + str(majority_matrix['0.6'][2])+ "\t" + str(majority_matrix['0.6'][3]))
    print("1.0  | " + str(majority_matrix['1.0'][0]) + "\t" + str(majority_matrix['1.0'][1])+ "\t" + str(majority_matrix['1.0'][2])+ "\t" + str(majority_matrix['1.0'][3]))



def plot_pub_mode_differences():
    global annotations_new
    global annotations_old
    global id_pub_list
    if id_pub_list is None or annotations_new is None:
        read_input_files()

    id_pub_dict = {}
    for i in id_pub_list:
        try:
            id_pub_dict[i['pub']].append(i['id'])
        except:
            id_pub_dict[i['pub']] = []
            id_pub_dict[i['pub']].append(i['id'])

    fig = plt.figure(num=1, figsize=(8, 8), dpi=100)
    index = 1
    id_mode_dict = {}
    mode_counter_total = {'0.0': 0, '0.3333333333': 0, '0.6666666666': 0, '1.0': 0}

    for a in annotations_new:
        id_mode_dict[a['id']] = a['truthMode']
        mode_counter_total[str(a['truthMode'])] += 1


    id_pub_dict = collections.OrderedDict(sorted(id_pub_dict.items()))

    for key, value in id_pub_dict.items():
        mode_counter = {'0.0': 0, '0.3333333333': 0, '0.6666666666': 0, '1.0': 0}
        for v in value:
            mode_counter[str(id_mode_dict[str(v)])] += 1
        tmp = fig.add_subplot(9, 3, index)
        tmp.bar([0.0, 0.33, 0.66, 1.0], [(mode_counter['0.0'] / len(value) - mode_counter_total['0.0'] / len(annotations_new)) * 100,
                                         (mode_counter['0.3333333333'] / len(value) - mode_counter_total['0.3333333333'] / len(annotations_new)) * 100,
                                         (mode_counter['0.6666666666'] / len(value) - mode_counter_total['0.6666666666'] / len(annotations_new)) * 100,
                                         (mode_counter['1.0'] / len(value) - mode_counter_total['1.0'] / len(annotations_new)) * 100],
                                         width=0.25)
        tmp.set_title(str(key))
        tmp.set_ylim(-10, 10, 10)
        tmp.minorticks_off()
        tmp.set_yticklabels("")
        tmp.set_xticklabels("")
        index += 1

    plt.subplots_adjust(wspace=0.1, hspace=1)
    plt.show()


def plot_mode_major_dist():
    # stacked bar chart of mode distributions
    global annotations_new
    if annotations_new is None:
        read_input_files()

    y_major = {}
    total_nr_tweets = len(annotations_new)
    for a in annotations_new:
        mode = a['truthMode']
        majority = Counter(a['truthJudgments'])[a['truthMode']]
        try:
            y_major[str(a['truthMode'])][str(majority)] += 1
            y_major[str(a['truthMode'])]['total'] += 1
        except KeyError:
            y_major[str(a['truthMode'])] = {'2': 0, '3': 0, '4': 0,'5': 0, 'total': 0}
            y_major[str(a['truthMode'])][str(majority)] += 1
            y_major[str(a['truthMode'])]['total'] += 1

    total_tweets = len(annotations_new)
    print(total_tweets)
    sum = 0
    for modes in y_major:
        sum += y_major[modes]['total']
        print(modes + ", " + str(y_major[modes]['total']))

    print(sum)

    x = [0.0, 0.33, 0.66, 1.0]

    y_5 = [(y_major['0.0']['5'] / total_nr_tweets) * 100,
           (y_major['0.3333333333']['5'] / total_nr_tweets) * 100,
           (y_major['0.6666666666']['5'] / total_nr_tweets) * 100,
           (y_major['1.0']['5'] / total_nr_tweets) * 100]

    y_4 = [(y_major['0.0']['4'] / total_nr_tweets) * 100,
           (y_major['0.3333333333']['4'] / total_nr_tweets) * 100,
           (y_major['0.6666666666']['4'] / total_nr_tweets) * 100,
           (y_major['1.0']['4'] / total_nr_tweets) * 100]

    y_3 = [(y_major['0.0']['3'] / total_nr_tweets) * 100,
           (y_major['0.3333333333']['3'] / total_nr_tweets) * 100,
           (y_major['0.6666666666']['3'] / total_nr_tweets) * 100,
           (y_major['1.0']['3'] / total_nr_tweets) * 100]

    y_2 = [(y_major['0.0']['2'] / total_nr_tweets) * 100,
           (y_major['0.3333333333']['2'] / total_nr_tweets) * 100,
           (y_major['0.6666666666']['2'] / total_nr_tweets) * 100,
           (y_major['1.0']['2'] / total_nr_tweets) * 100]

    width = 0.2       # the width of the bars: can also be len(x) sequence

    plt.figure(num=1, figsize=(8, 8), dpi=100)

    p1 = plt.bar(x, y_5, width)
    p2 = plt.bar(x, y_4, width, bottom=y_5)

    tmp_y4 = [x + y for (x, y) in zip(y_5, y_4)]
    p3 = plt.bar(x, y_3, width, bottom=tmp_y4)

    tmp_y3 = [x + y for (x, y) in zip(tmp_y4, y_3)]
    p4 = plt.bar(x, y_2, width, bottom=tmp_y3)

    plt.title('mode distribution by majority')

    plt.ylabel('percentage of total tweets')
    plt.xlabel('mode')
    plt.xticks(x, ('0.0', '0.33', '0.66', '1.0'))
    # plt.yticks(np.arange(0, 8000, 1000))
    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('5', '4', '3', '2'))
    plt.show()


def plot_mean_major_dist_old():
    global annotations_old
    if annotations_old is None:
        read_input_files()
    # plot mean old dataset
    y_major = [{},{},{},{}]
    y_major_all = {}
    y_major_3 = {}
    y_major_4 = {}
    y_major_5 = {}

    for a in annotations_old:
        mean = str(round(a['truthMean'], 5))[:5]
        most_common = Counter(a['truthJudgments']).most_common(1)
        for di in y_major:
            try:
                di[mean] = di[mean]
            except KeyError:
                di[mean] = 0

        y_major[0][mean] += 1
        if most_common[0][1] > 2:
            y_major[1][mean] += 1
        if most_common[0][1] > 3:
            y_major[2][mean] += 1
        if most_common[0][1] > 4:
            y_major[3][mean] += 1


    ordered_all = collections.OrderedDict(sorted(y_major[0].items()))
    ordered_3 = collections.OrderedDict(sorted(y_major[1].items()))
    ordered_4 = collections.OrderedDict(sorted(y_major[2].items()))
    ordered_5 = collections.OrderedDict(sorted(y_major[3].items()))


    # Area chart
    plt.fill_between([float(x) for x in ordered_5.keys()], 0, list(ordered_5.values()), label="5")
    plt.fill_between([float(x) for x in ordered_4.keys()], list(ordered_5.values()), list(ordered_4.values()), label="4+")
    plt.fill_between([float(x) for x in ordered_3.keys()], list(ordered_4.values()), list(ordered_3.values()), label="3+")
    plt.fill_between([float(x) for x in ordered_all.keys()], list(ordered_3.values()), list(ordered_all.values()), label="all")

    ''' Line graph
    plt.plot([float(x) for x in ordered_5.keys()], list(ordered_5.values()), label="5")
    plt.plot([float(x) for x in ordered_4.keys()], list(ordered_4.values()), label="4+")
    plt.plot([float(x) for x in ordered_3.keys()], list(ordered_3.values()), label="3+")
    plt.plot([float(x) for x in ordered_all.keys()], list(ordered_all.values()), label="all")
    '''

    plt.legend()
    plt.show()

def plot_mean_major_dist_new():
    # plot mean new dataset
    global annotations_new
    if annotations_new is None:
        read_input_files()
    y_major = [{},{},{},{}]
    y_major_all = {}
    y_major_3 = {}
    y_major_4 = {}
    y_major_5 = {}

    for a in annotations_new:
        mean = str(round(a['truthMean'], 5))[:6]
        most_common = Counter(a['truthJudgments']).most_common(1)
        for di in y_major:
            try:
                di[mean] = di[mean]
            except KeyError:
                di[mean] = 0

        y_major[0][mean] += 1
        if most_common[0][1] > 2:
            y_major[1][mean] += 1
        if most_common[0][1] > 3:
            y_major[2][mean] += 1
        if most_common[0][1] > 4:
            y_major[3][mean] += 1


    ordered_all = collections.OrderedDict(sorted(y_major[0].items()))
    ordered_3 = collections.OrderedDict(sorted(y_major[1].items()))
    ordered_4 = collections.OrderedDict(sorted(y_major[2].items()))
    ordered_5 = collections.OrderedDict(sorted(y_major[3].items()))


    # Area chart
    plt.fill_between([float(x) for x in ordered_5.keys()], 0, list(ordered_5.values()), label="5")
    plt.fill_between([float(x) for x in ordered_4.keys()], list(ordered_5.values()), list(ordered_4.values()), label="4+")
    plt.fill_between([float(x) for x in ordered_3.keys()], list(ordered_4.values()), list(ordered_3.values()), label="3+")
    plt.fill_between([float(x) for x in ordered_all.keys()], list(ordered_3.values()), list(ordered_all.values()), label="all")

    ''' Line graph
    plt.plot([float(x) for x in ordered_5.keys()], list(ordered_5.values()), label="5")
    plt.plot([float(x) for x in ordered_4.keys()], list(ordered_4.values()), label="4+")
    plt.plot([float(x) for x in ordered_3.keys()], list(ordered_3.values()), label="3+")
    plt.plot([float(x) for x in ordered_all.keys()], list(ordered_all.values()), label="all")
    '''
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_mode_major_dist()
