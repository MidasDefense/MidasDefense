#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import json
import re
import hashlib
import logging
import pickle
from gensim.models import Word2Vec
from sklearn.feature_extraction import FeatureHasher
from dirty_cat import SimilarityEncoder

target = "./chract"

class FeatureType(object):
    name = ''
    dim = 0

    def __repr__(self):
        return '{}({})'.format(self.name, self.dim)

    def raw_features(self, input_dict):
        raise (NotImplemented)

    def process_features(self, raw_obj):
        raise (NotImplemented)

    def feature_vector(self, input_dict):
        return self.process_raw_features(self.raw_features(input_dict))


class APIName(FeatureType):
    name = 'api_name'
    dim = 32

    def __init__(self):
        super(FeatureType, self).__init__()
        self.encoder = Word2Vec.load(target+'/skip-gram_previous.model')
        self.feature = np.zeros((32,), dtype=np.float32)

    def raw_features(self, input_dict):
        try:
            self.feature = self.encoder.wv[input_dict]
        except:
            logging.error("api %s is not in the corpus" % input_dict)
        return self.feature

    def process_raw_features(self, raw_obj):
        return raw_obj


class ArgIntInfo(FeatureType):
    ''' int hash info '''

    name = 'int'
    dim = 16

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, input_dict):
        hasher = FeatureHasher(self.dim).transform([input_dict]).toarray()[0]
        return hasher

    def process_raw_features(self, raw_obj):
        return raw_obj


class ArgPDRUIInfo(FeatureType):
    name = 'pdrui'
    dim = 16 + 16 + 16 + 16 + 16

    def __init__(self):
        super(FeatureType, self).__init__()
        self._paths = re.compile('^c:\\\\', re.IGNORECASE)
        self._dlls = re.compile('.+\.dll$', re.IGNORECASE)
        self._urls = re.compile('^https?://(.+?)[/|\s|:]', re.IGNORECASE)
        self._registry = re.compile('^HKEY_')
        self._ips = re.compile('^((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})(\.((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})){3}$')
        with open(target + "/paths_sim.pkl", 'rb') as file:
            self.paths_encoder = pickle.loads(file.read())
        with open(target + "/dlls_sim.pkl", 'rb') as file:
            self.dlls_encoder = pickle.loads(file.read())
        with open(target + "/registry_sim.pkl", 'rb') as file:
            self.registry_encoder = pickle.loads(file.read())
        with open(target + "/urls_sim.pkl", 'rb') as file:
            self.urls_encoder = pickle.loads(file.read())
        with open(target + "/ips_sim.pkl", 'rb') as file:
            self.ips_encoder = pickle.loads(file.read())

    def raw_features(self, input_dict):
        paths_feature = np.zeros((16,), dtype=np.float32)
        dlls_feature = np.zeros((16,), dtype=np.float32)
        registry_feature = np.zeros((16,), dtype=np.float32)
        urls_feature = np.zeros((16,), dtype=np.float32)
        ips_feature = np.zeros((16,), dtype=np.float32)
        for str_name, str_value in input_dict.items():
            if self._dlls.match(str_value):
                dll = re.split('\\\\', str_value)[-1]
                dlls_feature += self.dlls_encoder.transform([[dll]]).reshape(-1)
            if self._paths.match(str_value):
                paths_feature += self.paths_encoder.transform([[str_value]]).reshape(-1)
            elif self._registry.match(str_value):
                registry_feature += self.registry_encoder.transform([[str_value]]).reshape(-1)
            elif self._urls.match(str_value):
                urls_feature += self.urls_encoder.transform([[str_value]]).reshape(-1)
            elif self._ips.match(str_value):
                ips_feature += self.ips_encoder.transform([[str_value]]).reshape(-1)

        return np.hstack([paths_feature, dlls_feature, registry_feature, urls_feature, ips_feature]).astype(np.float32)

    def process_raw_features(self, raw_obj):
        return raw_obj



class ArgStrSInfo(FeatureType):
    ''' Other printable strings info '''

    name = 'strs'
    dim = 4

    def __init__(self):
        super(FeatureType, self).__init__()
        self._allstrings = re.compile(b'[\x20-\x7f]{5,}')
        self._mz = re.compile(b'MZ')
        super(FeatureType, self).__init__()

    def raw_features(self, input_dict):
        bytez = '\x11'.join(input_dict.values()).encode('UTF-8', 'ignore')
        allstrings = self._allstrings.findall(bytez)
        if allstrings:
            # statistics about strings:
            string_lengths = [len(s) for s in allstrings]
            avlength = sum(string_lengths) / len(string_lengths)
            # map printable characters 0x20 - 0x7f to an int array consisting of 0-95, inclusive
            as_shifted_string = [b - ord(b'\x20') for b in b''.join(allstrings)]
            c = np.bincount(as_shifted_string, minlength=96)  # histogram count
            # distribution of characters in printable strings
            csum = c.sum()
            p = c.astype(np.float32) / csum
            wh = np.where(c)[0]
            H = np.sum(-p[wh] * np.log2(p[wh]))  # entropy
        else:
            avlength = 0
            c = np.zeros((96,), dtype=np.float32)
            H = 0
            csum = 0
        return {
            'numstrings': len(allstrings),
            'avlength': avlength,
            'entropy': float(H),
            'MZ': len(self._mz.findall(bytez))
        }

    def process_raw_features(self, raw_obj):
        return np.hstack([raw_obj['numstrings'], raw_obj['avlength'], raw_obj['entropy'], raw_obj['MZ']]).astype(
            np.float32)


# Parsing the function call graph(FCG) of IoT malware.
class CallGraphReportEncoding(object):

    def __init__(self, file_md5, input_path, output_path, max_len):
        self.file_md5 = file_md5
        self.input_path = input_path
        self.output_path = output_path
        self.max_len = max_len
        self.features = dict((fe.name, fe) for fe in [APIName(), ArgIntInfo(), ArgPDRUIInfo(), ArgStrSInfo()])
        self.data = []

    def extract_features_for_classification(self):
        f = open(self.input_path)
        t = json.load(f)
        procs = t['behavior']['processes']
        for proc in procs:
            calls = proc['calls']
            previous_hashed = ""
            for call in calls:
                if len(self.data) >= self.max_len:
                    return True
                if 'api' not in call:
                    continue
                if call['api'][:2] == '__':
                    continue
                if 'arguments' not in call:
                    call['arguments'] = {}
                if 'category' not in call:
                    call['category'] = ""
                if 'status' not in call:
                    call['status'] = 0
                api = call['api']
                arguments = call['arguments'] 
                call_sign = api + "-" + str(arguments)
                current_hashed = hashlib.md5(call_sign.encode()).hexdigest()
                if previous_hashed == current_hashed:
                    continue
                else:
                    previous_hashed = current_hashed
                api_name_feature = self.features['api_name'].feature_vector(api)
                arg_int_dict, arg_str_dict = {}, {}
                for key, value in arguments.items():
                    if isinstance(value, (list, dict, tuple)):
                        continue
                    if isinstance(value, (int, float)):
                        arg_int_dict[key] = np.log(np.abs(value) + 1)
                    else:
                        if value is None:
                            continue
                        elif value[:2] == '0x':
                            continue
                        else:
                            arg_str_dict[key] = value

                try:
                    arg_int_feature = self.features['int'].feature_vector(arg_int_dict) 
                    arg_pdrui_feature = self.features['pdrui'].feature_vector(arg_str_dict) 
                    arg_strs_feature = self.features['strs'].feature_vector(arg_str_dict)  
                    api_feature = np.hstack(
                        [api_name_feature, arg_int_feature, arg_pdrui_feature, arg_strs_feature]).astype(np.float32)
                    # print(api_feature)
                    self.data.append(api_feature)
                except Exception as e:
                    logging.error("api error: %s" % e)
                    pass

        return True

    def save(self):
        np.savez(self.output_path + self.file_md5, data=self.data)
        return True


# Get IoT malware features
if __name__ == '__main__':
    files = os.listdir("./callgraphs/")
    for callgraph_file in files:
        input_path = './callgraphs/'+callgraph_file  # path of the call graph
        output_path = './data/sequence/'  # path of the encoded feature vectors
        max_len = 1000
        call_graph_report_encoding = CallGraphReportEncoding(callgraph_file, input_path, output_path, max_len)
        if call_graph_report_encoding.extract_features_for_classification():
            call_graph_report_encoding.save()