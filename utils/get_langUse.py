import spacy
import numpy as np
import pandas as pd


POS = {
    "": 'NO_TAG',
    "ADJ": 'ADJ',
    "ADP": 'ADP',
    "ADV": 'ADV',
    "AUX": 'AUX',
    "CONJ": 'CONJ',  # U20
    "CCONJ": 'CCONJ',
    "DET": 'DET',
    "INTJ": 'INTJ',
    "NOUN": 'NOUN',
    "NUM": 'NUM',
    "PART": 'PART',
    "PRON": 'PRON',
    "PROPN": 'PROPN',
    "PUNCT": 'PUNCT',
    "SCONJ": 'SCONJ',
    "SYM": 'SYM',
    "VERB": 'VERB',
    "X": 'X',
    "EOL": 'EOL',
    "SPACE": 'SPACE'
}
MORPH = ['VerbForm=Ger', 'VerbForm=Conv', 'Mood=Prp', 'Animacy=Hum', 'Case=Gen', 'Aspect=Prog', 'NumType=Ord', 'Number=Count', 'Case=Loc', 'Number=Pauc', 'Case=Abe', 'Case=Ill', 'Case=Voc', 'Case=Sup', 'Case=Par', 'NounClass=Bantu20', 'Case=Ine', 'NounClass=Bantu15', 'Voice=Lfoc', 'Degree=Dim', 'Mood=Cnd', 'NounClass=Bantu10', 'Case=Erg', 'Voice=Rcp', 'Number=Inv', 'PronType=Emp', 'Mood=Des', 'Tense=Past', 'Degree=Abs', 'Polite=Form', 'Clusivity=In', 'NounClass=Bantu3', 'Animacy=Nhum', 'Case=Tem', 'NounClass=Wol4', 'Person=1', 'Evident=Nfh', 'Mood=Opt', 'NounClass=Bantu13', 'NounClass=Wol10', 'Voice=Act', 'NounClass=Bantu4', 'Polarity=Pos', 'Polite=Humb', 'NumType=Frac', 'Gender=Com', 'Case=Equ', 'Case=Per', 'Case=Ela', 'Voice=Bfoc', 'Degree=Aug', 'Number=Coll', 'Number=Tri', 'Typo=Yes', 'Aspect=Iter', 'Case=Ins', 'Voice=Cau', 'NumType=Sets', 'PronType=Neg', 'Voice=Antip', 'Case=Cns', 'Degree=Pos', 'Aspect=Imp', 'Definite=Def', 'VerbForm=Fin', 'Case=Cmp', 'Voice=Pass', 'Mood=Pot', 'Case=Spl', 'Person=0', 'Person=4', 'Definite=Spec', 'Case=Add', 'NounClass=Bantu19', 'NumType=Dist', 'Number=Dual', 'PronType=Ind', 'Number=Sing', 'NounClass=Bantu6', 'Number=Grpa', 'VerbForm=Inf', 'Animacy=Anim', 'PronType=Prs', 'Case=Com', 'NounClass=Bantu5', 'PronType=Tot', 'NounClass=Wol8', 'Polite=Elev', 'NounClass=Wol2', 'NounClass=Bantu14', 'Case=Acc', 'Case=Sub', 'NounClass=Bantu16', 'NounClass=Wol7', 'VerbForm=Part', 'NumType=Range', 'Voice=Inv', 'NounClass=Bantu9', 'Polite=Infm', 'NounClass=Wol9', 'VerbForm=Gdv', 'Tense=Pres', 'Abbr=Yes', 'NumType=Mult', 'Definite=Com', 'Case=Ade', 'NounClass=Wol12', 'NounClass=Bantu18', 'Case=Dat', 'NounClass=Bantu23', 'Case=Sbl', 'Case=Ter', 'Gender=Fem', 'Case=Abl', 'Mood=Jus', 'Mood=Imp', 'NounClass=Bantu7', 'PronType=Int', 'NounClass=Wol3', 'NumType=Card', 'NounClass=Bantu17', 'Aspect=Perf', 'Mood=Ind', 'PronType=Rcp', 'Aspect=Hab', 'Degree=Cmp', 'Evident=Fh', 'Case=Nom', 'Tense=Fut', 'Case=Dis', 'Tense=Imp', 'Case=Ess', 'Mood=Int', 'Gender=Neut', 'NounClass=Bantu12', 'VerbForm=Vnoun', 'Gender=Masc', 'Case=All', 'Tense=Pqp', 'Mood=Qot', 'Number=Ptan', 'Voice=Mid', 'NounClass=Bantu2', 'Case=Sbe', 'Mood=Nec', 'Mood=Sub', 'Number=Plur', 'Foreign=Yes', 'Degree=Equ', 'Reflex=Yes', 'Number=Grpl', 'Voice=Dir', 'Aspect=Prosp', 'NounClass=Wol5', 'Person=3', 'Case=Cau', 'VerbForm=Sup', 'Poss=Yes', 'NounClass=Bantu8', 'PronType=Art', 'Case=Tra', 'Case=Abs', 'PronType=Rel', 'Mood=Adm', 'PronType=Exc', 'NounClass=Bantu22', 'PronType=Dem', 'Definite=Ind', 'NounClass=Wol6', 'NounClass=Bantu1', 'Person=2', 'Case=Lat', 'Case=Ben', 'Polarity=Neg', 'Clusivity=Ex', 'Definite=Cons', 'Case=Core: ', 'Case=Del', 'Animacy=Inan', 'Mood=Irr', 'Degree=Sup']
DEP = ['ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'compound', 'conj', 'csubj', 'csubjpass', 'dative', 'dep', 'det', 'dobj', 'expl', 'intj', 'mark', 'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 'nsubjpass', 'nummod', 'oprd', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj', 'predet', 'prep', 'prt', 'punct', 'quantmod', 'relcl', 'xcomp']



nlp = spacy.load("en_core_web_sm")


def extract_features(sentence):

    # initialization
    pos_counts = {pos: 0 for pos in POS}
    dep_counts = {dep: 0 for dep in DEP}
    morph_counts = {morph: 0 for morph in MORPH}

    # sentence processing
    doc = nlp(sentence)
    for token in doc:
        pos_counts[token.pos_] += 1
        dep_counts[token.dep_] += 1
        for morph in str(token.morph).split('|'):
            if morph in morph_counts:
                morph_counts[morph] += 1

    pos_vector = [pos_counts[pos] for pos in POS]
    dep_vector = [dep_counts[dep] for dep in DEP]
    morph_vector = [morph_counts[morph] for morph in MORPH]

    # merge to a single vector
    feature_vector = pos_vector + dep_vector + morph_vector
    return feature_vector

# exp_sentence = '''I think that it is kind of a forest because there are some trees in the background and there are also fields. The girl sitting on the right is painting some trees on her picture. And I think that painting in the nature would be really kind of cool. And the advantage is that you can really get exposed to nature and you can enjoy a really relaxing life and with no pressure. but the disadvantage may be you may be bitten by bugs in the grass so you make sure that you wear long sleeves and also pants. And the people in the pictures are mostly dressed in pants because to prevent the bugs and they are all painting the pictures of the forest in the background. And they are mostly wearing long sleeves. Some of them are wearing jackets. So I guess maybe it's kind of cold. Maybe it's the forest on a mountain.'''

# path could be anypath which contain the ASR transcription
path = '/share/nas165/peng/thesis_project/delivery_feat/delivery_0510_v2.csv'
df = pd.read_csv(path)

data = {}
for index, row in df.iterrows():
    dictt = {}

    # Getting transcription from ASR
    feature_vector = extract_features(row['whisperX_transcription'])
    
    for index, pos_name in enumerate(POS.keys()):
        dictt[pos_name] = feature_vector[index]
    pos_end_index = len(POS)

    for index, dep_name in enumerate(DEP):
        dictt[dep_name] = feature_vector[pos_end_index + index]
    dep_end_index = len(POS) + len(DEP)

    for index, morph_name in enumerate(MORPH):
        dictt[morph_name] = feature_vector[dep_end_index + index]
    data[row['speaker_id']] = dictt
    
print(data)

new_data = pd.DataFrame.from_dict(data, orient='index')
new_data = new_data.rename_axis('speaker_id')
new_data.to_csv('speaker_features_counts_0510.csv')
new_data.to_excel('speaker_features_counts_0510.xlsx')

print(new_data)

