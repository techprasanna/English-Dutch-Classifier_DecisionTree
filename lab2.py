"""
This program is an English-Dutch classifier.

"""

# Author: Prasanna Bhope

import sys
import codecs
import re
import math
import pickle
from dTree import Node

class Functions():
    maintable =[]
    predict_dataset = []
    entropy = 0
    f1_article = ["a","an","the"]
    f2_pronoun = ["I","you","he","she","it","they","we","their","our","ours","theirs"]
    f3_average_length_of_word = 4.7
    f4_wh_words = ["when","who","what","where","whom"]
    f5_simple_preposition = ["in","at","on","to","for","of","after","before","under","with","behind","ahead"]
    f6_double_preposition = ["concerning","notwithstanding","pending","during","given","failing"]
    f7_past_tense = ["was","were","had"]
    f8_future_tense = ["will","shall"]
    f9_negation = ["no","not","neither","nor","never"]
    f10_dutch_words = ["ik","je","de","niet","het","wat"]
    list_features = [1,2,3,4,5,6,7,8,9,10]
    possible_values = ["Present","Not Present"]
    result_list = []

    def feature_1(self,linelist):
        for i in linelist:
            if i not in self.f1_article:
                continue
            else:
                return "Present"
        return "Not Present"

    def feature_2(self,linelist):
        for i in linelist:
            if i not in self.f2_pronoun:
                continue
            else:
                return "Present"
        return "Not Present"

    def feature_3(self,linelist):
        avg_len = 0

        for i in linelist:
            avg_len = avg_len + len(i)

        average = avg_len/len(linelist)

        if average > 4 and average < 5:
            return "Present"
        return "Not Present"

    def feature_4(self,linelist):
        for i in linelist:
            if i not in self.f4_wh_words:
                continue
            else:
                return "Present"
        return "Not Present"

    def feature_5(self,linelist):
        for i in linelist:
            if i not in self.f5_simple_preposition:
                continue
            else:
                return "Present"
        return "Not Present"

    def feature_6(self,linelist):
        for i in linelist:
            if i not in self.f6_double_preposition:
                continue
            else:
                return "Present"
        return "Not Present"

    def feature_7(self,linelist):
        for i in linelist:
            if i not in self.f7_past_tense:
                continue
            else:
                return "Present"
        return "Not Present"

    def feature_8(self,linelist):
        for i in linelist:
            if i not in self.f8_future_tense:
                continue
            else:
                return "Present"
        return "Not Present"

    def feature_9(self,linelist):
        for i in linelist:
            if i not in self.f9_negation:
                continue
            else:
                return "Present"
        return "Not Present"

    def feature_10(self,linelist):
        for i in linelist:
            if i not in self.f10_dutch_words:
                continue
            else:
                return "Not Present"
        return "Present"

    def screen_feature(self,linelist):
        features = []
        if linelist[0] == "en" or linelist[0] == "nl":
            label = linelist[0]
            features.append(label)
            features.append(self.feature_1(linelist))
            features.append(self.feature_2(linelist))
            features.append(self.feature_3(linelist))
            features.append(self.feature_4(linelist))
            features.append(self.feature_5(linelist))
            features.append(self.feature_6(linelist))
            features.append(self.feature_7(linelist))
            features.append(self.feature_8(linelist))
            features.append(self.feature_9(linelist))
            features.append(self.feature_10(linelist))
        else:
            features.append(self.feature_1(linelist))
            features.append(self.feature_2(linelist))
            features.append(self.feature_3(linelist))
            features.append(self.feature_4(linelist))
            features.append(self.feature_5(linelist))
            features.append(self.feature_6(linelist))
            features.append(self.feature_7(linelist))
            features.append(self.feature_8(linelist))
            features.append(self.feature_9(linelist))
            features.append(self.feature_10(linelist))
        return features

    def max_ig(self,maintable):
        ig_val = []
        # print("Max ig called")
        for i in range(1,len(maintable[0])):
            ig_val.append(self.calculate_ig(i,maintable))
        # print(ig_val)
        return ig_val.index(max(ig_val))

    def remove_attr(self,feature_num,maintable):

        for line in maintable:
            line.pop(feature_num+1)
        return maintable

    def buildtable(self,linelist):
        self.maintable.append(self.screen_feature(linelist))

    def buildtable_predict(self,linelist):
        self.predict_dataset.append(self.screen_feature(linelist))

    def traindata_ada(self,filename):
        inputfile = []
        with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                check = re.findall(r"[\w']+", line)
                self.buildtable(check)

        n = len(self.maintable)
        initial_weights = 1 / n
        #print(self.maintable)
        weights = (self.add_Weights(self.maintable,initial_weights,n))
        self.AdaBoost_decisionTree(self.maintable,1,weights)

    def ada_boostAlgo(self,maintable,K):
        w = self.weightList(maintable)
        h = []
        z = []

        for k in range(1,K+1):
            h.append(self.AdaBoost_decisionTree(maintable,1,w))


    def AdaBoost_decisionTree(self, maintable,level,weights):

        error = 0
        root = Node(self.max_ig(maintable))
        #print(root.value)
        # root = Node(self.max_ig(maintable))
        # print(level)
        # if level <0:
        #     return root
        # # list_features.pop(root.value)
        index = []
        present_indices = self.seg_values_ada(maintable, root.value, "Present")
        present_values = self.seg_values(maintable, root.value, "Present")
        for i in present_values:
            if root.value == 10:
                if i[0] == "en":
                    error = error + 1
            else:
                if i[0] == "nl":
                    error = error + 1
        not_present_values = self.seg_values(maintable, root.value, "Not Present")
        not_present_indices = self.seg_values_ada(maintable, root.value, "Not Present")
        error_indices = present_indices + not_present_indices
        root.present = Node(self.calc_plularity(present_values))
        root.not_present = Node(self.calc_plularity(not_present_values))

        print(error_indices)
        for i in not_present_values:
            if root.value == 10:
                if i[0] == "en":
                    error = error + 1
            else:
                if i[0] == "nl":
                    error = error + 1
        total_error = error*weights[0]

        performance_stump = 1/2 * math.log((1-total_error)/total_error)
        # print(performance_stump)
        for i in range(len(weights)):
            if i in error_indices:
                weights[i] = weights[i] * math.exp(performance_stump)
            else:
                weights[i] = weights[i] * math.exp(-performance_stump)
        add = sum(weights)
        for i in range(len(weights)):
            weights[i] = weights[i]/add
        # print(weights)

        #
        # # maintable = self.remove_attr(root.value, maintable)
        # root.present = self.AdaBoost_decisionTree(present_values, level-1,weight_list)
        # root.not_present = self.AdaBoost_decisionTree(not_present_values, level-1,weight_list)
        # # return root


    def add_Weights(self,maintable,initial_weights,n):
        weights = []
        for i in range(n):
            weights.append(initial_weights)
        return weights

    def weightList(self,maintable):
        weights = []
        for line in maintable:
            weights.append(line[-1])
        return weights

    def traindata_dt(self,filename):
        inputfile = []
        with codecs.open(filename,'r',encoding='utf-8',errors='ignore') as f:
            for line in f:
                check = re.findall(r"[\w']+",line)
                self.buildtable(check)
        # print(self.calculate_entropy(self.maintable))
        #print(self.maintable)
        # print(self.calculate_ig(6,self.maintable))
        # print(self.max_ig(self.maintable))
        # print(self.remove_attr(self.max_ig(self.maintable),self.maintable))
        n = Node(self.decisionTree(self.maintable,self.list_features,self.maintable,self.possible_values))

        # self.print_leaf(n.value)
        pickle.dump(n.value,open(sys.argv[3],"wb"))
        print("Tree is created and object stored to filename given in arguments")

    # def print_leaf(self,root):
    #     if(root == None):
    #         return
    #     elif(root.present == None and root.not_present == None):
    #         print (root.value)
    #
    #     if root.present:
    #         self.print_leaf(root.present)
    #     if root.not_present:
    #         self.print_leaf(root.not_present)

    def calculate_ig(self,feature_num,maintable):
        # print("Feature number:",feature_num)
        no_present = []
        no_notpresent = []

        if(feature_num == 0):
            print ("Wrong feature selected")
            return

        for line in maintable:
            if line[feature_num] =='Present':
                no_present.append(line)
            elif line[feature_num] == 'Not Present':
                no_notpresent.append(line)

        # print("Present", len(no_present))
        # print("Not Present",len(no_notpresent))

        number_p = len(no_present)
        number_np = len(no_notpresent)
        # print(number_p,number_np)
        e_maintable = self.calculate_entropy(maintable)
        e_p = self.calculate_entropy(no_present)
        e_np = self.calculate_entropy(no_notpresent)
        weighted_avg = ((number_p/len(maintable)) * e_p) + ((number_np/len(maintable)) * e_np)
        ig =  (e_maintable - weighted_avg)
        return ig

    def calculate_entropy(self,maintable):
        if len(maintable) ==0:
            return 0
        no_en = 0
        no_nl = 0
        for line in maintable:
            if line[0] == "nl":
                no_nl = no_nl + 1
            elif line[0] == "en":
                no_en = no_en + 1
        prob_en = no_en/len(maintable)
        prob_nl = no_nl/len(maintable)
        val = 0
        val1 = 0
        if(prob_en == 0):
            val = 0
        else:
            val = math.log2(prob_en)

        if(prob_nl == 0):
            val1 = 0
        else:
            val1 = math.log2(prob_nl)
        #print(prob_en,prob_nl)

        entropy = (-prob_en * val) - (prob_nl * val1)
        return entropy

    def calc_plularity(self,maintable):
        no_en = 0
        no_nl = 0
        for line in maintable:
            if line[0] == "nl":
                no_nl = no_nl + 1
            elif line[0] == "en":
                no_en = no_en + 1
        if(no_en < no_nl):
            return "nl"
        else:
            return "en"

    def check_sameValues(self,maintable):
        check_val = []
        for line in maintable:
            check_val.append(line[0])
        result = all(element == check_val[0] for element in check_val)
        if result:
            return True
        else:
            return False

    def seg_values_ada(self,maintable,feature_num,value):
        indices = []
        for line in maintable:
            if line[feature_num+1] == value:
                if value == "Present" and line[0] == "nl":
                    indices.append(maintable.index(line))
                elif value == "Not Present" and line[0] == "en":
                    indices.append(maintable.index(line))
            else:
                continue

        return indices

    def seg_values(self,maintable,feature_num,value):
        values_present = []
        values_notPresent = []

        for line in maintable:
            if line[feature_num+1] == value:
                values_present.append(line)
            else:
                continue

        return values_present





    def decisionTree(self,maintable,list_features,parent_dataset,possible_values):
        if len(maintable) == 0:
            root = Node(self.calc_plularity(parent_dataset))
            # print("When length of maintable is zero:",root.value)
            return root

        elif self.check_sameValues(maintable):
            root = Node(maintable[0][0])
            # print ("When having same values",root.value)
            return root

        elif len(list_features)==0:
            root = Node(self.calc_plularity(maintable))
            # print("When all features are over",root.value)
            return root

        else:
            root = Node(self.max_ig(maintable))
            #print(root.value)
            list_features.pop(root.value)
            present_values = self.seg_values(maintable, root.value, "Present")
            not_present_values = self.seg_values(maintable, root.value, "Not Present")
            maintable = self.remove_attr(root.value,maintable)
            root.present = self.decisionTree(present_values,list_features,maintable,self.possible_values)
            root.not_present = self.decisionTree(not_present_values,list_features,maintable,self.possible_values)
            return root



    def process_data(self,filename):
        inputfile = []
        with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                check = re.findall(r"[\w']+", line)
                self.buildtable_predict(check)
            return (self.predict_dataset)




    def predictdata(self,root,line):
        result_list = []
        if root.value == "en" or root.value == "nl":
            self.result_list.append(root.value)
        else:
            value = line[root.value]
            if value == "Present":
                self.predictdata(root.present,line)
            elif value == "Not Present":
                self.predictdata(root.not_present,line)





if __name__ == '__main__':
    f = Functions()
    if len(sys.argv)==1:
        print("Please enter full arguments")
        exit(1)
    if sys.argv[1] == "train":
        if sys.argv[4] == "dt":
            f.traindata_dt(sys.argv[2])
        elif sys.argv[4] == "ada":
            f.traindata_ada(sys.argv[2])
        else:
            print("Bad argument at specifying learning algo")
            exit(1)
    elif sys.argv[1] == "predict":
        root_node = pickle.load( open(sys.argv[2], "rb" ) )
        check = f.process_data(sys.argv[3])
        for line in check:
            (f.predictdata(root_node,line))
        for i in f.result_list:
            print(i)
    else:
        print("Bad argument at specifying learning algo")
        exit(1)
