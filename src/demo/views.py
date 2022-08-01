

from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
import os
import json
import requests
import time
import pandas as pd
import numpy as np

import traceback

DATA_PATH = 'static/data/'
TXT_PATH = 'static/reftxtpapers/overall/'

Survey_dict = {
    '2742488' : 'Energy Efficiency in Cloud Computing',
    '2830555' : 'Cache Management for Real-Time Systems',
    '2907070' : 'Predictive Modeling on Imbalanced Data',
    '3073559' : 'Malware Detection with Data Mining',
    '3274658' : 'Analysis of Handwritten Signature'
}



Survey_Topic_dict = {
    '2742488' : ['energy'],
    '2830555' : ['cache'],
    '2907070' : ['imbalanced'],
    '3073559' : ['malware', 'detection'],
    '3274658' : ['handwritten', 'signature']
}


Survey_n_clusters = {
    '2742488' : 3,
    '2830555' : 3,
    '2907070' : 3,
    '3073559' : 3,
    '3274658' : 2
}

Global_survey_id = ""
Global_ref_list = []
Global_category_description = []
Global_category_label = []
Global_df_selected = ""


from demo.taskDes import absGen, introGen,introGen_supervised, methodologyGen, conclusionGen
from demo.category_and_tsne import clustering, get_cluster_description, clustering_with_criteria


class reference_collection(object):
    def __init__(
            self,
            input_df
    ):
        self.input_df = input_df

    def full_match_with_entries_in_pd(self, query_paper_titles):
        entries_in_pd = self.input_df.copy()
        entries_in_pd['ref_title'] = entries_in_pd['ref_title'].apply(str.lower)
        query_paper_titles = [i.lower() for i in query_paper_titles]

        # matched_entries = entries_in_pd[entries_in_pd['ref_title'].isin(query_paper_titles)]
        matched_entries = self.input_df[entries_in_pd['ref_title'].isin(query_paper_titles)]
        #print(matched_entries.shape)
        return matched_entries,matched_entries.shape[0]

    # select the sentences that can match with the topic words
    def match_ref_paper(self, query_paper_titles,match_mode='full', match_ratio=70):
        # query_paper_title = query_paper_title.lower()
        # two modes for str matching
        if match_mode == 'full':
            matched_entries, matched_num = self.full_match_with_entries_in_pd(query_paper_titles)
        return matched_entries, matched_num

def index(request):
    return render(request, 'demo/index.html')


@csrf_exempt
def upload_refs(request):
    file_dict = request.FILES
    file_name = list(file_dict.keys())[0]
    file_obj = file_dict[file_name]
    with open(DATA_PATH + file_name, 'wb+') as f:
        for chunk in file_obj.chunks():
            f.write(chunk)

    input_csv = pd.read_csv(DATA_PATH + file_name, sep = '\t')

    references = []
    ref_links = []
    ref_ids = []

    if input_csv['reference paper category id (optional)'].nunique() == 0:  # no topics
        topic_words = []
        references = list(input_csv['reference paper title'])
        ref_links = list(input_csv.index)  # unavailable
        ref_ids = list(input_csv.index)
    for df in input_csv.groupby('reference paper category id (optional)'):
        topic_words = ['Class Imbalance', 'Evaluation Measure', 'Performance Models']
        references.append(list(df[1]['reference paper title']))
        ref_links.append(list(df[1]['reference paper doi link (optional)']))
        ref_ids.append(list(df[1].index))

    global Global_survey_id
    Global_survey_id = '2907070'



    # Survey_dict[Global_survey_id] = topic
    # Survey_Topic_dict[Global_survey_id] = [topic.lower()]
    # references, ref_links, ref_ids = get_refs(Global_survey_id)
    #
    # ref_links = [TXT_PATH + i for i in files]


    ref_list = {
            'topic_words': topic_words,
            'references': references,
            'ref_links': ref_links,
            'ref_ids': ref_ids
    }


    ref_list = json.dumps(ref_list)
    return HttpResponse(ref_list)




    # with open('some/file/name.txt', 'wb+') as destination:
    #     for chunk in f.chunks():
    #         destination.write(chunk)
    # return HttpResponse(file_name)
    # files = request.POST.getlist('files[]')
    # global Global_survey_id
    # Global_survey_id = '001'
    # Survey_dict[Global_survey_id] = topic
    #
    # print(files[0])
    # Survey_Topic_dict[Global_survey_id] = [topic.lower()]
    #
    # Survey_n_clusters[Global_survey_id] = 2
    #
    # input_folder_dir = DATA_PATH + 'merge.tsv'
    # input_tsv = pd.read_csv(input_folder_dir, sep='\t', header=0)
    # query_paper_titles = [i.split('.')[0] for i in files]
    #
    #
    #
    # ref_set = reference_collection(input_tsv)
    # matched_entries_pd, matched_entries_num = ref_set.match_ref_paper(query_paper_titles, match_mode='full')
    #
    # # print(matched_entries_pd)
    # matched_entries_pd.to_csv(DATA_PATH + '001.tsv', sep='\t')
    #
    # references, ref_links, ref_ids = get_refs(Global_survey_id)
    #
    # ref_links = [TXT_PATH+i for i in files]
    #
    # for i in ref_links:
    #     print(i)
    #





@csrf_exempt
def get_topic(request):
    topic = request.POST.get('topics', False)
    references, ref_links, ref_ids = get_refs(topic)
    global Global_survey_id
    Global_survey_id = topic
    ref_list = {
        'references' : references,
        'ref_links'  : ref_links,
        'ref_ids'    : ref_ids
    }
    ref_list = json.dumps(ref_list)
    return HttpResponse(ref_list)

@csrf_exempt
def automatic_taxonomy(request):
    ref_dict = dict(request.POST)
    print(ref_dict)
    ref_list = ref_dict['refs']
    query = ref_dict['taxonomy_standard'][0]
    global Global_ref_list
    Global_ref_list = ref_list

    colors, category_label, category_description =  Clustering_refs(n_clusters=Survey_n_clusters[Global_survey_id])
    # colors, category_label, category_description = Clustering_refs_with_criteria(n_clusters=Survey_n_clusters[Global_survey_id], query=query)

    global Global_category_description
    Global_category_description = category_description
    global Global_category_label
    Global_category_label = category_label



    df_tmp = Global_df_selected.reset_index()
    df_tmp['index'] = df_tmp.index
    ref_titles = list(df_tmp.groupby(df_tmp['label'])['ref_title'].apply(list))
    ref_indexs = list(df_tmp.groupby(df_tmp['label'])['index'].apply(list))
    cate_list = {
        'colors': colors,
        'category_label': category_label,
        'survey_id': Global_survey_id,
        'ref_titles': ref_titles,
        'ref_indexs': ref_indexs
    }
    print(cate_list)
    cate_list = json.dumps(cate_list)
    return HttpResponse(cate_list)


@csrf_exempt
def select_sections(request):

    sections = request.POST
    # print(sections)

    survey = {}




    for k,v in sections.items():
        if k == "title":
            survey['title'] = "A Survey of " + Survey_dict[Global_survey_id]
        if k == "abstract":
            survey['abstract'] = ["The class imbalance problem is encountered in a large number of practical applications of machine learning and data mining, for example, information retrieval and filtering, and the detection of credit card fraud. The imbalanced learning problem is concerned with the performance of learning algorithms in the presence of underrepresented data and severe class distribution skews. Classification of data with imbalanced class distribution has posed a significant drawback of the performance attainable by most standard classifier learning algorithms, which assume a relatively balanced class distribution and equal misclassification costs. In this survey, we conduct a comprehensive overview of predictive modeling on imbalanced data."," We classify existing methods into three categories: Class Imbalance, Evaluation Measure,  and Performance Models."]
        if k == "introduction":
            survey['introduction'] = '''Class imbalance is a common problem in machine learning (ml ) and data mining (dm ) that occurs when a large number of classes are present in training data (e.g., in medical applications or in the internet of things (iot ) applications ). In this situation, which is found in real world data describing an infrequent but important event, the learning system may have difficulties to learn the concept related to the minority class. Class imbalance has attracted a huge amount of attention from researchers and practitioners in the last decade. However, there are still many fundamental open-ended questions such as are all learning paradigms equally affected by class imbalance? support vector machines (svms ) is a popular machine learning technique that works effectively with balanced datasets. However, when it comes to imbalanced datasets, svms produce suboptimal classification models. The class imbalance problem has attracted growing attention from both academia and industry due to the explosive growth of applications that use and produce imbalanced data. It is important to accurately identify the class of interest but occurs relatively rarely such as cases of fraud, instances of disease, and regions of interest in largescale simulations, it then becomes more costly to misclassify the interesting class.
            In this paper, we provide a comprehensive review of the state-of-the-art class imbalance learning methods and their evaluation metrics that can be used to evaluate the performance of a learning system in the presence of imbalanced data. In addition, we also provide a detailed analysis of the evaluation metrics used in the evaluation of class imbalanced learning methods in the literature so far. The rest of the paper is organized as follows. In section 2, we introduce the class imbalance problem and its evaluation metrics in section 3. Section 4 presents an overview of the existing evaluation metrics for class imbalance in uci data sets and section 5 presents a comprehensive analysis of their performance in the absence of imbalance in section 6. Section 7 concludes the paper with a discussion of the future research directions in the field of uci learning and evaluation metrics. The remainder of this paper is structured as follows: in section 8 we review the existing class imbalance evaluation metrics and their performance evaluation in the uci domain. Section 9 presents a detailed review of their evaluations in section 10. Section 11 presents an analysis of how the current evaluation metrics can be improved in the future and section 12 presents a summary of their future directions in section 13. In section 13 we present a detailed comparison of the performance evaluation metrics of different uci evaluation metrics with respect to their effectiveness in the current uci environment. Section 14 presents a comparison of different evaluation metrics based on their effectiveness and their efficiency in dealing with the imbalanced problem in the present uci scenario.
            In the following section, we will introduce existing works in each category in detail.
            '''
        if k == "methodology":
            survey['methodology'] = [
                'We reviewed existing works and classify them into three types namely: Class Imbalance, Evaluation Measure,  and Performance Models. The first category is about the class imbalance. the empirical study also suggests that some methods that have been believed to be effective in addressing the class imbalance problem may, in fact, only be effective on learning with imbalanced two-class data sets.  The sencond category is about the evaluation measure. moreover, if this evaluation measure is used for parameter optimization, a parameter choice may result that makes the system behave very much like a trivial system.  The third category is about the performance models. in this paper we present the notion of rec surfaces, describe how to use them to compare the performance of models, and illustrate their use with an important practical class of applications: the prediction of rare extreme values. \n',
                [{'subtitle': 'Class Imbalance',
                  'content': 'For Class Imbalance, there are several existing works. Batista, et al. [1] proposes a simple experimental design to assess the performance of class imbalance treatment methods.  E.A.P.A. et al. [2] performs a broad experimental evaluation involving ten methods, three of them proposed by the authors, to deal with the class imbalance problem in thirteen uci data sets.  Batuwita, et al. [3] presents a method to improve fsvms for cil (called fsvm-cil), which can be used to handle the class imbalance problem in the presence of outliers and noise.  V. et al. [4] implements a wrapper approach that computes the amount of under-sampling and synthetic generation of the minority class examples (smote) to improve minority class accuracy.  Chen, et al. [5] presents ranked minority oversampling in boosting (ramoboost), which is a ramo technique based on the idea of adaptive synthetic data generation in an ensemble learning system.  Chen, et al. [6] proposes a new feature selection method, feature assessment by sliding thresholds (fast), which is based on the area under a roc curve generated by moving the decision boundary of a single feature classifier with thresholds placed using an even-bin distribution.  Davis, et al. [7] shows that a deep connection exists between roc space and pr space, such that a curve dominates in roc space if and only if it dominates in pr space.  In classifying documents, the system combines the predictions of the learners by applying evolutionary techniques as well [8]. Ertekin, et al. [9] is concerns with the class imbalance problem which has been known to hinder the learning performance of classification algorithms.  Ertekin, et al. [10] demonstrates that active learning is capable of solving the problem.  Garcı́???aÿ, et al. [11] analyzes a generalization of a new metric to evaluate the classification performance in imbalanced domains, combining some estimate of the overall accuracy with a plain index about how dominant the class with the highest individual accuracy is.  Ghasemi, et al. [12] proposes an active learning algorithm that can work when only samples of one class as well as a set of unlabeled data are available.  He, et al. [13] provides a comprehensive review of the development of research in learning from imbalanced data.  Li, et al. [14] proposes an oversampling method based on support degree in order to guide people to select minority class samples and generate new minority class samples.  Li, et al. [15] analyzes the intrinsic factors behind this failure and proposes a suitable re-sampling method.  Liu, et al. [16] proposes two algorithms to overcome this deficiency.  J. et al. [17] considers the application of these ensembles to imbalanced data : classification problems where the class proportions are significantly different.  Seiffert, et al. [18] presents a new hybrid sampling/boosting algorithm, called rusboost, for learning from skewed training data.  Song, et al. [19] proposes an improved adaboost algorithm called baboost (balanced adaboost), which gives higher weights to the misclassified examples from the minority class.  Sun, et al. [20] develops a cost-sensitive boosting algorithm to improve the classification performance of imbalanced data involving multiple classes.  Van et al. [21] presents a comprehensive suite of experimentation on the subject of learning from imbalanced data.  Wasikowski, et al. [22] presents a first systematic comparison of the three types of methods developed for imbalanced data classification problems and of seven feature selection metrics evaluated on small sample data sets from different applications.  an active under-sampling approach is proposed for handling the imbalanced problem in Yang, et al. [23]. Zhou, et al. [24] studies empirically the effect of sampling and threshold-moving in training cost-sensitive neural networks. \n'},
                 {'subtitle': 'Evaluation Measure',
                  'content': 'For Evaluation Measure, there are several existing works. Baccianella, et al. [25] proposes a simple way to turn standard measures for or into ones robust to imbalance.  Lin, et al. [26] applies a fuzzy membership to each input point and reformulate the svms such that different input points can make different constributions to the learning of decision surface. \n'},
                 {'subtitle': 'Performance Models',
                  'content': 'For Performance Models, there are several existing works. Drummond, et al. [27] proposes an alternative to roc representation, in which the expected cost of a classi er is represented explicitly.  Tao, et al. [28] develops a mechanism to overcome these problems.  Torgo et al. [29] presents a generalization of regression error characteristic (rec) curves.  C. et al. [30] demonstrates that class probability estimates attained via supervised learning in imbalanced scenarios systematically underestimate the probabilities for minority class instances, despite ostensibly good overall calibration.  Yoon, et al. [31] proposes preprocessing majority instances by partitioning them into clusters.  Zheng, et al. [32] investigates the usefulness of explicit control of that combination within a proposed feature selection framework. \n'}]]

        if k == "conclusion":
            conclusion = '''In this survey, we conduct a comprehensive overview of predictive modeling on imbalanced data. We provide a taxonomy which groups the researchs of predictive modeling on imbalanced data into three categories: Class Imbalance, Evaluation Measure,  and Performance Models.
            '''
            survey['conclusion'] = conclusion

        # for k, v in sections.items():
        #     if k == "title":
        #         survey['title'] = "A Survey of " + Survey_dict[Global_survey_id]
        #     if k == "abstract":
        #         abs, last_sent = absGen(Global_survey_id, Global_df_selected, Global_category_label)
        #         survey['abstract'] = [abs, last_sent]
        #     if k == "introduction":
        #         # intro = introGen_supervised(Global_survey_id, Global_df_selected, Global_category_label, Global_category_description, sections)
        #         intro = introGen(Global_survey_id, Global_df_selected, Global_category_label,
        #                          Global_category_description, sections)
        #         survey['introduction'] = intro
        #     if k == "methodology":
        #         proceeding, detailed_des = methodologyGen(Global_survey_id, Global_df_selected, Global_category_label,
        #                                                   Global_category_description)
        #         survey['methodology'] = [proceeding, detailed_des]
        #         print('======')
        #         print(survey['methodology'])
        #         print('======')
        #
        #     if k == "conclusion":
        #         conclusion = conclusionGen(Global_survey_id, Global_category_label)
        #         survey['conclusion'] = conclusion


        ## reference
        ## here is the algorithm part
        # df = pd.read_csv(data_path, sep='\t')

        survey['references'] = []
        # try:
        #     for ref in Global_df_selected['ref_entry']:
        #         entry = str(ref)
        #         survey['References'].append(entry)
        # except:
        #     colors, category_label, category_description = Clustering_refs(n_clusters=Survey_n_clusters[Global_survey_id])
        #     for ref in Global_df_selected['ref_entry']:
        #         entry = str(ref)
        #         survey['References'].append(entry)


    survey_dict = json.dumps(survey)

    return HttpResponse(survey_dict)


@csrf_exempt
def get_survey(request):
    survey_dict = get_survey_text()
    survey_dict = json.dumps(survey_dict)
    return HttpResponse(survey_dict)


def get_refs(topic):
    '''
    Get the references from given topic
    Return with a list
    '''
    default_references = ['ref1','ref2','ref3','ref4','ref5','ref6','ref7','ref8','ref9','ref10']
    default_ref_links = ['', '', '', '', '', '', '', '', '', '']
    default_ref_ids = ['', '', '', '', '', '', '', '', '', '']
    references = []
    ref_links = []
    ref_ids = []

    try:
        ## here is the algorithm part
        ref_path   = os.path.join(DATA_PATH, topic + '.tsv')
        df         = pd.read_csv(ref_path, sep='\t')
        for i,r in df.iterrows():
            # print(r['intro'], r['ref_title'], i)
            if not pd.isnull(r['intro']):
                references.append(r['ref_title'])
                ref_links.append(r['ref_link'])
                ref_ids.append(i)
    except:
        print(traceback.print_exc())
        references = default_references
        ref_links = default_ref_links
        ref_ids = default_ref_ids
    print(len(ref_ids))
    return references, ref_links, ref_ids


def get_survey_text(refs=Global_ref_list):
    '''
    Get the survey text from a given ref list
    Return with a dict as below default value:
    '''
    # print(refs)
    survey = {
        'Title': "A Survey of " + Survey_dict[Global_survey_id],
        'Abstract': "test "*150,
        'Introduction': "test "*500,
        'Methodology': [
            "This is the proceeding",
            [{"subtitle": "This is the first subtitle", "content": "test "*500},
             {"subtitle": "This is the second subtitle", "content": "test "*500},
             {"subtitle": "This is the third subtitle", "content": "test "*500}]
        ],
        'Conclusion': "test "*150,
        'References': []
    }

    try:
        ## abs generation
        abs, last_sent = absGen(Global_survey_id, Global_df_selected, Global_category_label)
        survey['Abstract'] = [abs, last_sent]

        ## Intro generation
        #intro = introGen_supervised(Global_survey_id, Global_df_selected, Global_category_label, Global_category_description)
        intro = introGen(Global_survey_id, Global_df_selected, Global_category_label, Global_category_description)
        survey['Introduction'] = intro

        ## Methodology generation
        proceeding, detailed_des = methodologyGen(Global_survey_id, Global_df_selected, Global_category_label, Global_category_description)
        survey['Methodology'] = [proceeding, detailed_des]

        ## Conclusion generation
        conclusion = conclusionGen(Global_survey_id, Global_category_label)
        survey['Conclusion'] = conclusion

        ## reference
        ## here is the algorithm part
        # df = pd.read_csv(data_path, sep='\t')
        try:
            for ref in Global_df_selected['ref_entry']:
                entry = str(ref)
                survey['References'].append(entry)
        except:
            colors, category_label, category_description = Clustering_refs(n_clusters=Survey_n_clusters[Global_survey_id])
            for ref in Global_df_selected['ref_entry']:
                entry = str(ref)
                survey['References'].append(entry)

    except:
        print(traceback.print_exc())
    return survey


def Clustering_refs(n_clusters):
    df = pd.read_csv(DATA_PATH + Global_survey_id + '.tsv', sep='\t', index_col=0)
    df_selected = df.iloc[Global_ref_list]

    print(df_selected.shape)
    ## update cluster labels and keywords
    df_selected, colors = clustering(df_selected, n_clusters, Global_survey_id)
    # print(colors)
    print(df_selected.shape)
    ## get description and topic word for each cluster
    description_list = get_cluster_description(df_selected, Global_survey_id)
    # print(description_list)

    global Global_df_selected
    Global_df_selected = df_selected
    category_description = [0]*len(colors)
    category_label = [0]*len(colors)
    for i in range(len(colors)):
        for j in description_list:
            if j['category'] == i:
                category_description[i] = j['category_desp']
                category_label[i] = j['topic_word'].replace('-', ' ').title()
    return colors, category_label, category_description
    # return 1,0,1

def Clustering_refs_with_criteria(n_clusters, query):
    df = pd.read_csv(DATA_PATH + Global_survey_id + '.tsv', sep='\t', index_col=0)
    df_selected = df.iloc[Global_ref_list]

    print(df_selected.shape)
    ## update cluster labels and keywords
    df_selected, colors = clustering_with_criteria(df_selected, n_clusters, Global_survey_id, query)
    # print(colors)
    print(df_selected.shape)
    ## get description and topic word for each cluster
    description_list = get_cluster_description(df_selected, Global_survey_id)
    # print(description_list)

    global Global_df_selected
    Global_df_selected = df_selected
    category_description = [0]*len(colors)
    category_label = [0]*len(colors)
    for i in range(len(colors)):
        for j in description_list:
            if j['category'] == i:
                category_description[i] = j['category_desp']
                category_label[i] = j['topic_word'].replace('-', ' ').title()
    return colors, category_label, category_description



