import os
import random

def generate_adhd_qa_txt(input_txt):
    docs = input_txt.split('|')
    age = docs[0][5:]
    gender = docs[1][8:]
    age_gender_text = 'A {}-year-old {} participant. '.format(age, gender)

    diagnosis = docs[2][11:]
    text_dict = dict()
    if diagnosis == 'ADHD':       
        text_dict['answer'] = 'Yes, the diagnosis is Attention Deficit Hyperactivity Disorder [SEP]'
        text_dict['base_text'] = age_gender_text + 'The diagnosis is Attention Deficit Hyperactivity Disorder.'
    else:
        text_dict['answer'] = 'No, the diagnosis is Typically Developing Children [SEP]'
        text_dict['base_text'] = age_gender_text + 'The diagnosis is Typically Developing Children.'

    text_dict['question'] = 'Attention Deficit Hyperactivity Disorder has typical symptoms of hyperactivity, impulsivity, inattention, disinhibition, impaired working memory. Will this participant be diagnosed with Attention Deficit Hyperactivity Disorder?'
    text_dict['tq'] = age_gender_text + 'Question: {} Answer: '.format(text_dict['question'])
    
    return text_dict

def generate_hcp_txt(input_txt):
    docs = input_txt.split(', ')
    # print (docs)

    age = docs[0].split(': ')[1]
    gender = docs[1].split(': ')[1]
    race = docs[2].split(': ')[1]
    base_info_text = 'A {}-year-old {} participant.'.format(age, gender)

    flanker = docs[3].split(': ')[1]
    dccs = docs[4].split(': ')[1]
    lswmt = docs[5].split(': ')[1]
    
    orrt = docs[6].split(': ')[1]
    pcps = docs[7].split(': ')[1]
    psm = docs[8].split(': ')[1]
    tpvt = docs[9].split(': ')[1]

    cognition_txt_set = []
    abnorm_cognition_txt_set = []

    flanker_txt = ''
    if flanker != '':
        flanker_score = int(flanker)
        if flanker_score <= 85:
            flanker_txt = 'The Flanker test scores {}, below the normal limit. The Flanker test shows poor performance of attention function, indicating a risk of disinhibition and inattention.'.format(flanker_score)
        else:
            flanker_txt = 'The Flanker test scores {}, above the normal limit. The Flanker test shows normal performance of attention function.'.format(flanker_score)

        flanker_q = 'The participant accepts Flanker test. What is performance of attention function?'
        cognition_txt_set.append([flanker_txt, flanker_q])

        if flanker_score <= 85:
            abnorm_cognition_txt_set.append([flanker_txt, flanker_q])

    dccs_txt = ''
    if dccs != '':
        dccs_score = int(dccs)
        if dccs_score <= 85:
            dccs_txt = 'The Dimensional Change Card Sort test scores {}, below the normal limit. The Dimensional Change Card Sort test shows poor performance of shifting function, indicating a risk of cognitive inflexibility.'.format(dccs_score)           
        else:
            dccs_txt = 'The Dimensional Change Card Sort test scores {}, above the normal limit. The Dimensional Change Card Sort test shows normal performance of shifting function.'.format(dccs_score)
        
        dccs_q = 'The participant accepts Dimensional Change Card Sort test. What is performance of cognitive flexibility function?'
        cognition_txt_set.append([dccs_txt, dccs_q])

        if dccs_score <= 85:
            abnorm_cognition_txt_set.append([dccs_txt, dccs_q])

    lswmt_txt = ''
    if lswmt != '':
        lswmt_score = int(lswmt)
        if lswmt_score <= 85:
            lswmt_txt = 'The List Sorting test scores {}, below the normal limit. The List Sorting test shows poor performance of working memory function, indicating a risk of impaired working memory.'.format(lswmt_score)
        else:
            lswmt_txt = 'The List Sorting test scores {}, above the normal limit. The List Sorting test shows normal performance of working memory function.'.format(lswmt_score)

        lswmt_q = 'The participant accepts List Sorting test. What is performance of working memory function?'
        cognition_txt_set.append([lswmt_txt, lswmt_q])

        if lswmt_score <= 85:
            abnorm_cognition_txt_set.append([lswmt_txt, lswmt_q])

    orrt_txt = ''
    if orrt != '':
        orrt_score = int(orrt)
        if orrt_score <= 85:
            orrt_txt = 'The Oral Reading test scores {}, below the normal limit. The Oral Reading test shows poor performance of reading coding function, indicating a risk of delayed language skill in reading coding.'.format(orrt_score)
        else:
            orrt_txt = 'The Oral Reading test scores {}, above the normal limit. The Oral Reading test shows normal performance of reading coding function.'.format(orrt_score)

        orrt_q = 'The participant accepts Oral Reading Recognition test. What is performance of reading coding skill?'
        cognition_txt_set.append([orrt_txt, orrt_q])

        if orrt_score <= 85:
            abnorm_cognition_txt_set.append([orrt_txt, orrt_q])

    pcps_txt = ''
    if pcps != '':
        pcps_score = int(pcps)
        if pcps_score <= 85:
            pcps_txt = 'The Pattern Comparison test scores {}, below the normal limit. The Pattern Comparison test shows poor performance of processing speed function, indicating a risk of slow processing speed.'.format(pcps_score)
        else:
            pcps_txt = 'The Pattern Comparison test scores {}, above the normal limit. The Pattern Comparison test shows normal performance of processing speed function.'.format(pcps_score)

        pcps_q = 'The participant accepts Pattern Comparison Processing Speed test. What is performance of processing speed function?'
        cognition_txt_set.append([pcps_txt, pcps_q])

        if pcps_score <= 85:
            abnorm_cognition_txt_set.append([pcps_txt, pcps_q])


    psm_txt = ''
    if psm != '':
        psm_score = int(psm)
        if psm_score <= 85:
            psm_txt = 'The Picture Sequence test scores {}, below the normal limit. The Picture Sequence test test shows poor performance of episodic memory function, indicating a risk of episodic memory loss and impairment.'.format(psm_score)
        else:
            psm_txt = 'The Picture Sequence test scores {}, above the normal limit. The Picture Sequence test test shows normal performance of episodic memory function.'.format(psm_score)

        psm_q = 'The participant accepts Picture Sequence Memory test. What is performance of episodic memory function?'
        cognition_txt_set.append([psm_txt, psm_q])

        if psm_score <= 85:
            abnorm_cognition_txt_set.append([psm_txt, psm_q])

    tpvt_txt = ''
    if tpvt != '':
        tpvt_score = int(tpvt)
        if tpvt_score <= 85:
            tpvt_txt = 'The Picture Vocabulary test scores {}, below the normal limit. The Picture Vocabulary test shows poor performance of vocabulary comprehension function, indicating a risk of delayed language skill in general vocabulary knowledge.'.format(tpvt_score)
        else:
            tpvt_txt = 'The Picture Vocabulary test scores {}, above the normal limit. The Picture Vocabulary test shows normal performance of vocabulary comprehension function.'.format(tpvt_score)

        tpvt_q = 'The participant accepts Picture Vocabulary test. What is performance of vocabulary comprehension function?'
        cognition_txt_set.append([tpvt_txt, tpvt_q])

    random.shuffle(cognition_txt_set)

    text_dict = dict()
    text_dict['base_text'] = base_info_text + ' ' + cognition_txt_set[0][0]
    text_dict['question'] = cognition_txt_set[0][1] 
    text_dict['answer'] = cognition_txt_set[0][0]
    text_dict['tq'] = base_info_text + ' Question: {} Answer:'.format(cognition_txt_set[0][1])
    
    return text_dict


if __name__ == '__main__':
    test_txt = 'age: 26, gender: female, race: Unknown or Not Reported, flanker: , dccs: 86, lswmt: , orrt: 113, pcps: 68, psm: 130, tpvt: 111'
    text_dict = generate_hcp_txt(test_txt)

    print (text_dict)
