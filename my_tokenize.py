import csv
# import jieba
#
#
# def tokenize(string):
#     re = jieba.cut(string)
#     return ' '.join(re)
#
#
# input_tokenizer = jieba.Tokenizer(dictionary="data/input_vocab")
# output_tokenizer = jieba.Tokenizer(dictionary="data/output_vocab")
#
#
# def do_tokenize(tokennizer, string):
#     re = tokennizer.tokenize(string, HMM=False)
#     tokens = []
#     for one in re:
#         tokens.append(one[0])
#     j = 0
#     for i in range(len(tokens)):
#         if tokens[j] == ' ':
#             tokens.pop(j)
#         else:
#             j += 1
#     return ' '.join(tokens)
#
#
# print(do_tokenize(input_tokenizer, "kǎo哈哈"))
#
#
# def create_map_from_vocab(f):
#     vocab_map = dict()
#     for (i, line) in enumerate(f):
#         vocab_map[line.split(' ')[0]] = str(i)
#     return vocab_map

# with open("data/SAOKE_DATA_discrete.csv") as s:
#     with open("data/SAOKE_DATA_tokenize.csv", 'w') as t:
#         reader = csv.reader(s, delimiter="\t")
#         writer = csv.writer(t, delimiter="\t")
#         writer.writerow(next(reader))
#         for line in reader:
#             writer.writerow([do_tokenize(input_tokenizer, line[0]), do_tokenize(output_tokenizer, line[1])])

# with open("data/SAOKE_DATA_tokenize.csv") as s:
#     with open("data/SAOKE_DATA_tokenize_id.csv", 'w') as t:
#         reader = csv.reader(s, delimiter="\t")
#         writer = csv.writer(t, delimiter="\t")
#         writer.writerow(next(reader))
#         input_vocab_map = create_map_from_vocab(open("data/input_vocab"))
#         output_vocab_map = create_map_from_vocab(open("data/output_vocab"))
#         for line in reader:
#             writer.writerow([
#                 ' '.join(map(lambda x: input_vocab_map[x], line[0].split(' '))),
#                 ' '.join(map(lambda x: output_vocab_map[x], line[1].split(' '))),
#             ])
