import json
import pandas as pd
import numpy as np
import re
from nltk.translate.bleu_score import sentence_bleu
import time
import argparse
import os


def indicator_cal(json_standered,json_test):

    json_standered = pd.DataFrame(json_standered)
    json_standered = json_standered['mid_json']
    json_test = pd.DataFrame(json_test)
    json_test = json_test['mid_json']


    start1 = time.time() 
    '''批量读取中间生成的json文件'''
    test_inline_equations=[]
    test_interline_equations=[]
    test_dropped_text_bboxes=[]
    test_dropped_text_tag=[]
    test_dropped_image_bboxes=[]
    test_dropped_table_bboxes=[] 
    test_preproc_num=[]#阅读顺序
    test_para_num=[]

    for i in json_test:
        mid_json=pd.DataFrame(i)
        mid_json=mid_json.iloc[:,:-1]
        for j1 in mid_json.loc['inline_equations',:]:
            for k1 in j1:
                test_inline_equations.append(k1['latex_text'])
        for j2 in mid_json.loc['interline_equations',:]:
            for k2 in j2:
                test_interline_equations.append(k2['latex_text'])
        for j3 in mid_json.loc['droped_text_block',:]:
            page_in=[]
            for k3 in j3:
                page_in.append(k3['bbox'])
                if 'tag' in k3:
                    test_dropped_text_tag.append(k3['tag'])
           
            test_dropped_text_bboxes.append(page_in)
        for j4 in mid_json.loc['droped_image_block',:]:
                test_dropped_image_bboxes.append(j4)
        for j5 in mid_json.loc['droped_table_block',:]:
                test_dropped_table_bboxes.append(j5)
        for j6 in mid_json.loc['preproc_blocks',:]:
            page_in=[]
            for k6 in j6:
                page_in.append(k6['number'])
            test_preproc_num.append(page_in)     
        for j7 in mid_json.loc['para_blocks',:]:
            test_para_num.append(len(j7))          



    standered_inline_equations=[]
    standered_interline_equations=[]
    standered_dropped_text_bboxes=[]
    standered_dropped_text_tag=[]
    standered_dropped_image_bboxes=[]
    standered_dropped_table_bboxes=[] 
    standered_preproc_num=[]#阅读顺序
    standered_para_num=[]

    for i in json_standered:
        mid_json=pd.DataFrame(i)
        mid_json=mid_json.iloc[:,:-1]
        for j1 in mid_json.loc['inline_equations',:]:
            for k1 in j1:
                standered_inline_equations.append(k1['latex_text'])
        for j2 in mid_json.loc['interline_equations',:]:
            for k2 in j2:
                standered_interline_equations.append(k2['latex_text'])
        for j3 in mid_json.loc['droped_text_block',:]:
            page_in=[]
            for k3 in j3:
                page_in.append(k3['bbox'])
                if 'tag' in k3:
                    standered_dropped_text_tag.append(k3['tag'])
                
            standered_dropped_text_bboxes.append(page_in)
        for j4 in mid_json.loc['droped_image_block',:]:
                standered_dropped_image_bboxes.append(j4)
        for j5 in mid_json.loc['droped_table_block',:]:
                standered_dropped_table_bboxes.append(j5)
        for j6 in mid_json.loc['preproc_blocks',:]:
            page_in=[]
            for k6 in j6:
                page_in.append(k6['number'])
            standered_preproc_num.append(page_in)     
        for j7 in mid_json.loc['para_blocks',:]:
            standered_para_num.append(len(j7))  
    
    end1 = time.time()
    print('Running time: %s Seconds'%(end1-start1))

    """
    在计算指标之前最好先确认基本统计信息是否一致
    """

    '''行内公式编辑距离和bleu'''
    dis1=[]
    bleu1=[]
    for a,b in zip(test_inline_equations,standered_inline_equations):
        if len(a)==0 and len(b)==0:
            continue
        else:
            if a==b:
                dis1.append(0)
                bleu1.append(1)
            else:
                dis1.append(Levenshtein_Distance(a,b))
                bleu1.append(sentence_bleu([a],b))
    inline_equations_edit=np.sum(dis1)
    inline_equations_bleu=np.mean(bleu1)

    '''行间公式编辑距离和bleu'''
    dis2=[]
    bleu2=[]
    for a,b in zip(test_interline_equations,standered_interline_equations):
        if len(a)==0 and len(b)==0:
            continue
        else:
            if a==b:
                dis2.append(0)
                bleu2.append(1)
            else:
                dis2.append(Levenshtein_Distance(a,b))
                bleu2.append(sentence_bleu([a],b))
    interline_equations_edit=np.sum(dis2)
    interline_equations_bleu=np.mean(bleu2)




    '''可以先检查page和bbox数量是否一致'''

    '''删除text block的准确率'''
    text_match_bbox=[]
    for a,b in zip(test_dropped_text_bboxes,standered_dropped_text_bboxes):
        if len(a)==0 and len(b)==0:
            text_match_bbox.append(1)
        else:
            for i in a:
                judge=0
                for j in b:
                    if bbox_offset(i,j):
                        judge=1
                        break
                text_match_bbox.append(judge)
    acc_text_block=np.mean(text_match_bbox)


    '''删除image block的准确率'''
    '''有数据格式不一致的问题'''
    image_match_bbox=[]
    for a,b in zip(test_dropped_image_bboxes,standered_dropped_image_bboxes):
        if len(a)==0 and len(b)==0:
            image_match_bbox.append(1)
        else:
            for i in a:
                if len(i)!=4:
                    continue
                else:
                    judge=0
                    for j in b:
                        if bbox_offset(i,j):
                            judge=1
                            break
                    image_match_bbox.append(judge)
    acc_image_block=np.mean(image_match_bbox)

    '''删除table block的准确率'''
    table_match_bbox=[]
    for a,b in zip(test_dropped_table_bboxes,standered_dropped_table_bboxes):
        if len(a)==0 and len(b)==0:
            table_match_bbox.append(1)
        else:
            for i in a:
                judge=0
                for j in b:
                    if bbox_offset(i,j):
                        judge=1
                        break
                table_match_bbox.append(judge)
    acc_table_block=np.mean(table_match_bbox)

    
   

    '''删除的text_block的tag的准确率'''
    tag_acc={}
    #standered_dropped_text_tag中的所有元素
    tag_type=set(standered_dropped_text_tag)
    for i in tag_type:
        tag_acc[i]=np.mean(test_dropped_text_tag[standered_dropped_text_tag==i]==standered_dropped_text_tag[standered_dropped_text_tag==i])
    
   

    '''阅读顺序编辑距离的均值'''
    preproc_num_dis=[]
    for a,b in zip(test_preproc_num,standered_preproc_num):
        # a=''.join(str(i) for i in a)
        # b=''.join(str(i) for i in b)
        preproc_num_dis.append(Levenshtein_Distance(a,b))
    preproc_num_edit=np.mean(preproc_num_dis)



    '''分段准确率'''
    test_para_num=np.array(test_para_num)
    standered_para_num=np.array(standered_para_num)
    acc_para=np.mean(test_para_num==standered_para_num)

    
    output=pd.DataFrame()
    output['行内公式编辑距离']=[inline_equations_edit]
    output['行间公式编辑距离']=interline_equations_edit
    output['行内公式bleu']=inline_equations_bleu
    output['行间公式bleu']=interline_equations_bleu
    output['删除的text_block的准确率']=acc_text_block
    output['删除的image_block的准确率']=acc_image_block
    output['删除的table_block的准确率']=acc_table_block
    output['阅读顺序编辑距离的均值']=preproc_num_edit
    output['分段准确率']=acc_para
    
    for i in tag_acc.keys():
        output['dropped_text_block中删除的'+i+'tag的准确率']=[tag_acc[i]]

    return output



"""
计算编辑距离
"""
def Levenshtein_Distance(str1, str2):
    matrix = [[ i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if(str1[i-1] == str2[j-1]):
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)
    return matrix[len(str1)][len(str2)]


'''
计算bbox偏移量是否符合标准的函数
'''
def bbox_offset(b_t,b_s):
    '''b_t是test_doc里的bbox,b_s是standered_doc里的bbox'''
    # print('1',b_t,'2',b_s)
    x1_t,y1_t,x2_t,y2_t=b_t
    x1_s,y1_s,x2_s,y2_s=b_s
    x1=max(x1_t,x1_s)
    x2=min(x2_t,x2_s)
    y1=max(y1_t,y1_s)
    y2=min(y2_t,y2_s)
    # if x1>x2 or y2<y1:
    #     return False #重合面积为0,似乎有点小问题，后续check
    # else:   
    #     area_overlap=(x2-x1)*(y2-y1)
    #     area_t=(x2_t-x1_t)*(y2_t-y1_t)+(x2_s-x1_s)*(y2_s-y1_s)-area_overlap
    #     if area_t-area_overlap==0 or area_overlap/(area_t-area_overlap)>0.95:
    #         return True
    #     else:
    #         return False
    area_overlap=(x2-x1)*(y2-y1)
    area_t=(x2_t-x1_t)*(y2_t-y1_t)+(x2_s-x1_s)*(y2_s-y1_s)-area_overlap
    if area_t-area_overlap==0 or area_overlap/(area_t-area_overlap)>0.95:
        return True
    else:
        return False
        




   
parser = argparse.ArgumentParser()
parser.add_argument('--test', type=str)
parser.add_argument('--standered', type=str)
args = parser.parse_args()
pdf_json_test = args.test
pdf_json_standered = args.standered

#    '''测试版本的json文件'''
#    pdf_json_test='pdf_json_label_0229.json'

#    '''标准版本的json文件'''
#    pdf_json_standered='pdf_json_label_0306.json'



if __name__ == '__main__':
    
   pdf_json_test = [json.loads(line) 
                        for line in open(pdf_json_test, 'r', encoding='utf-8')]
   pdf_json_standered = [json.loads(line) 
                    for line in open(pdf_json_standered, 'r', encoding='utf-8')]
   
   overall_indicator=indicator_cal(pdf_json_standered,pdf_json_test)

   '''计算的指标输出到overall_indicator_output.json中'''
   overall_indicator.to_json('overall_indicator_output.json',orient='records',lines=True,force_ascii=False)
    