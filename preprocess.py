import json
import csv

"""
预处理过程包括：
1）全角转半角
2）删除logic数组为空的行
3）将time和palace当作subject的限定词
4）lower()
"""


def to_unicode(string):
    ret = ''
    for v in string:
        ret = ret + hex(ord(v)).upper().replace('0X', '\\u')

    return ret


def knowledge_string(dict):
    attribute = []
    if dict["place"] is not "_":
        attribute.append(dict["place"])
    if dict["time"] is not "_":
        attribute.append(dict["time"])
    if attribute:
        attribute = "[" + ("|".join(attribute)) + "]"
    else:
        attribute = ""
    return [attribute + dict["subject"] + "," + dict["predicate"] + "," + obj for obj in dict["object"]]


header = ["knowledge", "natural"]


def is_number(uchar):
    """判断一个unicode是否是半角数字"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


def is_Qnumber(uchar):
    """判断一个unicode是否是全角数字"""
    if uchar >= u'\uff10' and uchar <= u'\uff19':
        return True
    else:
        return False


def is_alphabet(uchar):
    """判断一个unicode是否是半角英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False


def is_Qalphabet(uchar):
    """判断一个unicode是否是全角英文字母"""
    if (uchar >= u'\uff21' and uchar <= u'\uff3a') or (uchar >= u'\uff41' and uchar <= u'\uff5a'):
        return True
    else:
        return False


# 我觉得可以打乱knowledge的顺序生成多个（训练，测试）集
def Q2B(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


def stringQ2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])


s = open("data/SAOKE_DATA.json")
d = open("data/SAOKE_DATA_discrete_.csv", 'w')

writer = csv.writer(d, delimiter="\t")
writer.writerow(header)
for line in s.readlines():
    json_obj = json.loads(line)
    if not json_obj["logic"]:
        continue
    knowledge = []
    knowledge = [knowledge_string(one) for one in json_obj['logic']]
    writer.writerow([';'.join([';'.join(one) for one in knowledge]), json_obj['natural']])
s.close()
d.close()
# s = open("data/SAOKE_DATA_discrete_.csv")
# d = open("data/SAOKE_DATA_discrete_.csv", 'w')
# for line in s.readlines():
#     d.write(stringQ2B(line))
