import sys
import json
import math
import nltk
import scipy
nltk.download('stopwords')
from scipy import spatial
import subprocess
import numpy as np
from nltk.corpus import stopwords
import krovetz

queries=sys.argv[1]
relevance=sys.argv[2]
index=sys.argv[3]
#print(queries)
#print(relevance)
#print(index)

s=subprocess.getstatusoutput('galago dump-index-manifest '+str(index)+'/corpus/')
N=(json.loads((s[1])))["keyCount"]   #total#docs in corpus

s=subprocess.getstatusoutput('galago dump-keys '+str(index)+'/postings.krovetz')  #vocab
x=s[1].split('\n')

x_new=[]
en_stops = set(stopwords.words('english'))
for word in x:
    if word not in en_stops:
        x_new.append(word)

x=x_new
#print(len(x))
rows, cols = (len(x)+1,N+1)

tftable =[[0]*cols for _ in range(rows)]
idftable=[[0]*2 for _ in range(len(x)+1)]

for y in range(1,len(x)+1):
    tftable[y][0]=(x[y-1])
    idftable[y][0] = (x[y - 1])

for y in range(0,N):
    tftable[0][y+1]=y


idftable[0][1]=1
tftable=np.array(tftable)
idftable=np.array(idftable)

f = open(queries)
data = json.load(f)
docidandname={}
runninglistofevaldocs=[]
for k in data['queries']:
    query=k['text']
    first=query[9:]
    second=first[:-2]
    docsforthisq=[]
    binaryq = [0] * len(x)
    listofwordsforcurrq=second.split(' ')

    listofwordsforcurrq_new = []       #stopword removal
    for word in listofwordsforcurrq:
        if word not in en_stops:
            listofwordsforcurrq_new.append(word)

    listofwordsforcurrq = listofwordsforcurrq_new

    stem = []                     ##stemming queries
    ks = krovetz.PyKrovetzStemmer()
    for word in listofwordsforcurrq:
        cut = ks.stem(word)
        stem.append(cut)
    listofwordsforcurrq=stem

    listofwordsforcurrq = list(dict.fromkeys(listofwordsforcurrq))
    #print(listofwordsforcurrq)

    #print("numberq:\n")
    numq=k['number']
    #print(numq)
    #print(listofwordsforcurrq)
    for i in np.arange(len(listofwordsforcurrq)):
        if listofwordsforcurrq[i] in x :            #( query term is in vocab)
            currwordind=x.index(listofwordsforcurrq[i])
            #print(currwordind)
            binaryq[currwordind]=1
            #print("index"+str(i))
            #print(listofwordsforcurrq[i])
            s = subprocess.getstatusoutput('galago dump-key-value '+str(index)+'/postings.krovetz '+ listofwordsforcurrq[i])
            #print("s:"+s[1]+"\n")
            splitteddkv = np.array(s[1].split('\n'))
            listfory = []

            #print('TARGETED DOCS\n')
            for a in np.arange(1, len(splitteddkv)):
                entry = (splitteddkv[a]).split(',')
                #print("entry1:\n")
                tuple=(int(entry[1]),len(entry)-2)
                #print(tuple)
                listfory.append(tuple)
                #print("ADDED")
                docsforthisq.append(int(entry[1]))

            for j in np.arange(len(listfory)):
                if listfory[j][0] not in runninglistofevaldocs:
                    s = subprocess.getstatusoutput('galago dump-doc-terms --index='+str(index)+'/postings.krovetz --iidList='+str(listfory[j][0])+',')
                    documentstat=s[1].split('\n')
                    #print(documentstat)
                    docnameslist=documentstat[0].split('\t')
                    str1 = ''.join(docnameslist[0])
                    #print(str1)
                    DOCNO=str1.split("[",1)[1][:-1]
                    #print(DOCNO)
                    docidandname[listfory[j][0]]=DOCNO


                    for b in np.arange(1, len(documentstat)):
                        entry2 = (documentstat[b]).split(',')
                        #print(listfory[j][0])
                        #print(len(entry2))
                        #print(runninglistofevaldocs)
                        if len(entry2)>1:
                                #print("first time")
                                #print("TERM DOC INSTANCES")
                                #print(entry2)
                                #print(tftable[:, 0])
                                #print(entry2[0])
                                r=np.where(tftable[:, 0]==entry2[0])
                                #print(r[0])
                                if r[0].any():
                                    r=r[0][0]
                                    """
                                    print("in")
                                    print(r)
                                    """
                                    tftable[r][listfory[j][0]+1]=listfory[j][1]
                                    idftable[r][1]=math.log2(N/len(listfory))
                    #print("DONE")
                    #print(listfory[j][0])
                    #print(listfory)
                    runninglistofevaldocs.append(listfory[j][0])
                    #print(runninglistofevaldocs)


    #print("BEFORE")
    #print(tftable.tolist())
    #print(tftable.shape)
    docsforthisq = list(dict.fromkeys(docsforthisq))
    runninglistofevaldocs=list(dict.fromkeys(runninglistofevaldocs))
    dummytftable = np.delete(tftable, 0, 0)
    dummytftable = np.delete(dummytftable, 0,1 )
    #print("AFTER")
    #print(dummytftable.tolist())
    #print(dummytftable.shape)

    #print("BEFORE")
    #print(idftable.tolist())
    #print(idftable.shape)
    dummyidftable = np.delete(idftable, 0, 0)
    dummyidftable = np.delete(dummyidftable, 0, 1)
    #print("AFTER")
    #print(dummyidftable.tolist())
    #print(dummyidftable.shape)

    dummytftable = dummytftable.astype(float)
    dummyidftable = dummyidftable.astype(float)
    finaltable=np.multiply(dummytftable, dummyidftable)

    #print("FINALTABLE")
    #print(finaltable.tolist())


    #docsforthisq = [x + 1 for x in docsforthisq]
    finallist = []
    for z in np.arange(len(docsforthisq)):
        v1=binaryq
        v2=finaltable[:,docsforthisq[z]]
        result = 1 - spatial.distance.cosine(v1, v2)
        tuple=(result,docsforthisq[z])
        #print(tuple)
        finallist.append(tuple)

    finallist=sorted(finallist, key=lambda x: x[0],reverse=True)


    for s in np.arange(len(finallist)):
        print(str(numq)+" Q0 "+str(docidandname[finallist[s][1]])+" "+str(s+1)+" "+str((finallist[s][0]))[0:10]+" galago")



#print(idftable.tolist())
#print(tftable.tolist())

f.close()








