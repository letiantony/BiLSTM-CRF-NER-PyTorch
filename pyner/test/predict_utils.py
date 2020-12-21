#encoding:utf-8
import json
from collections import defaultdict

def format_result(result,words):
    '''
    形式调整
    :param result:
    :param words:
    :return:
    '''
    entities = defaultdict(list)
    entities['text'] = "".join(words)
    if len(result) == 0:
        entities['info'] = result
    else:
        for record in result:
            begin = record['begin']
            end  = record['end']
            entities['info'].append({
                'start':begin+1,
                'end':end+1,
                'word': "".join(words[begin:end + 1]),
                'type':record['type']
            })
    return entities

def get_entity(path, tag_map, bioes=True, sep='-'):
    '''
    entity：B_ORG B_PER B_T
    :param bioes:
    :param path:
    :param tag_map:
    :return:
    '''
    results = []
    record = {}
    last_tag = ''
    for index, tag_id in enumerate(path):
        if tag_id == 0:  # 0是我们的pad label
            continue
        tag = tag_map[tag_id]
        if bioes:
            if tag.startswith("B" + sep) or tag.startswith("S" + sep):
                if record:
                    results.append(record)
                if tag.startswith("B" + sep):
                    record = {'begin': index, 'end': index, 'type': tag.split(sep)[1]}
                else:
                    record = {'begin': index, 'end': index, 'type': tag.split(sep)[1]}
                    results.append(record)
                    record = {}
            elif tag.startswith('I' + sep):
                tag_type = tag.split(sep)[1]
                if record and tag_type == record['type'] and (last_tag == 'B' or last_tag == 'I'):
                    record['end'] = index
                else:
                    if record:
                        results.append(record)
                    record = {'begin': index, 'end': index, 'type': tag_type}
            elif tag.startswith('E' + sep):
                tag_type = tag.split(sep)[1]
                if record and tag_type == record['type']:
                    record['end'] = index
                    results.append(record)
                    record = {}
                else:
                    if record:
                        results.append(record)
                    record = {'begin': index, 'end': index, 'type': tag_type}
                    results.append(record)
                    record = {}
            else:
                if record:
                    results.append(record)
                record = {}
        else:
            if tag.startswith("B" + sep):
                if record:
                    results.append(record)
                record = {'begin': index, 'end': index, 'type': tag.split(sep)[1]}
            elif tag.startswith('I' + sep):
                tag_type = tag.split(sep)[1]
                if record and tag_type == record['type'] and (last_tag == 'I' or last_tag == 'B'):
                    record['end'] = index
                else:
                    if record:
                        results.append(record)
                    record = {'begin': index, 'end': index, 'type': tag_type}
            else:
                if record:
                    results.append(record)
                record = {}
        last_tag = tag.split(sep)[0]
    if record:
        results.append(record)
    return results

def test_write(data,filename,raw_text_path):
    '''
    将test结果保存到文本中，这里是以json格式写入
    :param data:
    :param filename:
    :param raw_text_path:
    :return:
    '''
    with open(raw_text_path,'r') as fr,open(filename,'w') as fw:
        for text,result in zip(fr.readlines(),data):
            words = text.split()
            record = format_result(result,words)
            encode_json = json.dumps(record)
            print(json.loads(encode_json))
            print(encode_json, file=fw)
