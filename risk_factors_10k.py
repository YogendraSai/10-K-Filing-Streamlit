import streamlit as st
from sec_api import ExtractorApi
import urllib.request 
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from selenium import webdriver
from time import sleep
import re
import pandas as pd
import requests 
import urllib.request 
import os
from nltk.tokenize import sent_tokenize 
import string
import nltk
#import nltkmodules
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier

# Function to highlight sentences based on labels
def highlight_sentences(sentence, label):
    colors = {
        'Business Risk': 'green',
        'Legal Risk':'red',
        'Technology Risk':'brown',
        'Security Risk':'blue',
        'Market Risk':'purple',
        'Financial Risk':'orange',
        'General Risk':'grey'
    }
    return f'<span style="background-color:{colors[label]}; padding: 1px; border-radius: 1px;">{sentence}</span>'

#phase scraping: scrape risk factors from 10k filings through API
def risk_fac(text_link):
    req = requests.get(text_link, headers={'User-Agent': 'XYZ/3.0'})
    soup = BeautifulSoup(req.content, 'lxml')
    
    for filing_document in soup.find_all('document'): 
        document_type = filing_document.type.find(text=True, recursive=False).strip()
        if document_type == "10-K": 
            TenKtext = filing_document.find('text').extract().text
            matches = re.compile(r'(item\s(1a[\.\s]|1b[\.\s])|''risk\sfactors[0-9|\s|a-z]|''(unresolved\sstaff\scomments))', re.IGNORECASE)
            matches_array = pd.DataFrame([(match.group(), match.start()) for match in matches.finditer(TenKtext)])
            matches_array.columns = ['SearchTerm', 'Start']
            Rows = matches_array['SearchTerm'].count()
            count = 0 
            while count < (Rows-1): 
                matches_array.at[count,'Selection'] = (matches_array.iloc[count,0] + matches_array.iloc[count+1,0]).lower() # Convert to lower case
                count += 1
            matches_item1a = re.compile(r'(item\s1a(\.|\s)risk\sfac[a-z0-9]*)')
            matches_item1b = re.compile(r'(item\s1b(\.|\s)unresolved\sstaff\s[a-z]*)')
            Start_Loc = []
            End_Loc = []
            count = 0 
            
            while count < (Rows-1): 
                if re.match(matches_item1a, matches_array.at[count,'Selection']):
                    Start_Loc.append(matches_array.iloc[count,1]) 
                if re.match(matches_item1b, matches_array.at[count,'Selection']):
                    End_Loc.append(matches_array.iloc[count,1])
                count += 1
            end = End_Loc[1]
            start = Start_Loc[1]
            Start_Loc.sort(reverse=True)
            for i in range(len(Start_Loc)):
                if Start_Loc[i] < end:
                    start = Start_Loc[i]
                    break
            
            TenKItem1a = TenKtext[start:end]
            
            TenKItem1a = TenKItem1a.strip() 
            TenKItem1a = TenKItem1a.replace('\n', ' ') 
            TenKItem1a = TenKItem1a.replace('\r', '') 
            while '  ' in TenKItem1a:
                TenKItem1a = TenKItem1a.replace('  ', ' ') 


    return TenKItem1a
        
    
#phase scraping: generate dynamic URL from the ticker by scraping the CIK id from ticker 
def get10kUrl(x):
    cik_dict = {}
    decode = ''
    
    for line in urllib.request.urlopen("https://www.sec.gov/include/ticker.txt"):
        decode = line.decode('utf-8')
        decode = decode.split()
        key, value = decode[0],decode[1]
        cik_dict[key] = value
        
    url10k = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000'+cik_dict[x]+'&type=10-k'
    return url10k

#phase scraping: from the search result, click on doc button and access the URL of the 10K file 
def get_doc(url10k):
    req = Request(url10k, headers={'User-Agent': 'XYZ/3.0'})
    webpage = urlopen(req, timeout=10).read()
    soup = BeautifulSoup(webpage, 'html.parser')
    a = soup.find(id='documentsbutton', href=True)
    doclink = a['href']
    req = Request("https://www.sec.gov/"+doclink, headers={'User-Agent': 'XYZ/3.0'})
    webpage = urlopen(req, timeout=10).read()
    soup = BeautifulSoup(webpage, 'html.parser')
    text_link = []
    for rows in soup.find_all('a', href=True, text=re.compile(r'.*\.txt')):
        text_link.append(rows['href'])
    text_link = "https://www.sec.gov/"+text_link[0]
    return text_link

############input ticker from User
def input_ticker(x):
    x = str(x)
    if x.startswith("https:") or x.startswith("http:"):
        risk_factor_data = risk_fac(x)
    else:
        x = x.lower()
        url = get10kUrl(x)
        doc_link = get_doc(url)
        risk_factor_data = risk_fac(doc_link)
        #print(x,doc_link)
    return risk_factor_data

#phase preprocessing: sentence tokenizer
def sentences(risk_factor_data):
    risk_sen = sent_tokenize(risk_factor_data)
    risk_factors = pd.DataFrame(risk_sen, columns =['risk_factor'])
    risk_factors = risk_factors[risk_factors["risk_factor"].str.contains("Item 1A.") == False] 
    risk_factors = risk_factors[risk_factors["risk_factor"].str.contains("item 1a.") == False] 
    return risk_factors

#phase preprocessing: normalize data

def prepro(risk_factors):
    ##to string
    risk_factors['risk_factor'] = risk_factors['risk_factor'].apply(lambda x: str(x))
    ## Lower case
    risk_factors['risk_factor'] = risk_factors['risk_factor'].apply(lambda x: " ".join(x.lower()for x in x.split()))
    ## remove special charecters
    risk_factors['risk_factor'] = risk_factors['risk_factor'].str.replace('[^a-zA-Z0-9]',' ')
    ## digits
    risk_factors['risk_factor'] = risk_factors['risk_factor'].str.replace('\d+','')
    
    #remove stop words
    stop = stopwords.words('english')
    risk_factors['risk_factor'] = risk_factors['risk_factor'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    
    datenumber = ["HUNDRED","THOUSAND","MILLION","BILLION","TRILLION","DATE ","ANNUAL","ANNUALLY","ANNUM","YEAR","YEARLY","QUARTER","QUARTERLY","QTR","MONTH","MONTHLY","WEEK","WEEKLY","DAY","DAILY","JANUARY ","FEBRUARY","MARCH","APRIL","MAY","JUNE","JULY","AUGUST","SEPTEMBER","OCTOBER","NOVEMBER","DECEMBER","JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","SEPT","OCT","NOV","DEC","MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY","ONE  ","TWO","THREE","FOUR","FIVE","SIX","SEVEN","EIGHT","NINE","TEN","ELEVEN","TWELVE","THIRTEEN","FOURTEEN","FIFTEEN","SIXTEEN","SEVENTEEN","EIGHTEEN","NINETEEN","TWENTY","THIRTY","FORTY","FIFTY","SIXTY","SEVENTY","EIGHTY","NINETY","FIRST","SECOND","THIRD","FOURTH","FIFTH","SIXTH","SEVENTH","EIGHTH","NINTH","TENTH","I ","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV","XV","XVI","XVII","XVIII","XIX","XX"]
    out = map(lambda x:x.lower(), datenumber)
    datenumber = list(out)
    
    generic = ["ABOUT","ABOVE","AFTER","AGAIN","ALL","AM","AMONG","AN","AND","ANY","ARE","AS","AT","BE","BECAUSE","BEEN","BEFORE","BEING","BELOW","BETWEEN","BOTH","BUT","BY","CAN","DID","DO","DOES","DOING","DOWN","DURING","EACH","FEW","FOR","FROM","FURTHER","HAD","HAS","HAVE","HAVING","HE","HER","HERE","HERS","HERSELF","HIM","HIMSELF","HIS","HOW","IF","IN","INTO","IS","IT","ITS","ITSELF","JUST","ME","MORE","MOST","MY","MYSELF","NO","NOR","NOT","NOW","OF","OFF","ON","ONCE","ONLY","OR","OTHER","OUR","OURS","OURSELVES","OUT","OVER","OWN","SAME","SHE","SHOULD","SO","SOME","SUCH","THAN","THAT","THE","THEIR","THEIRS","THEM","THEMSELVES","THEN","THERE","THESE","THEY","THIS","THOSE","THROUGH","TO","TOO","UNDER","UNTIL","UP","VERY","WAS","WE","WERE","WHAT","WHEN","WHERE","WHICH","WHILE","WHO","WHOM","WHY","WITH","YOU","YOUR","YOURS","YOURSELF","YOURSELVES"]
    out = map(lambda x:x.lower(), generic)
    generic = list(out)
    
    currcountry = ["AFGHANI","Afghanistan","ARIARY","Madagascar","BAHT","Thailand","BALBOA","Panama","BIRR","Ethiopia","BOLIVAR","Venezuela","BOLIVIANO","Bolivia","CEDI","Ghana","COLON","CostaRica","CÓRDOBA","Nicaragua","DALASI","Gambia","DENAR","Macedonia","DINAR","Algeria","DIRHAM","Morocco","DOBRA","SãoTomandPríncipe","DONG","Vietnam","DRAM","Armenia","ESCUDO","CapeVerde","EURO","Belgium","FLORIN","Aruba","FORINT","Hungary","GOURDE","Haiti","GUARANI","Paraguay","GULDEN","NetherlandsAntilles","HRYVNIA","Ukraine","KINA","PapuaNewGuinea","KIP","Laos","KONVERTIBILNAMARKA","Bosnia-Herzegovina","KORUNA","CzechRepublic","KRONA","Sweden","KRONE","Denmark","KROON","Estonia","KUNA","Croatia","KWACHA","Zambia","KWANZA","Angola","KYAT","Myanmar","LARI","Georgia","LATS","Latvia","LEK","Albania","LEMPIRA","Honduras","LEONE","SierraLeone","LEU","Romania","LEV","Bulgaria","LILANGENI","Swaziland","LIRA","Lebanon","LITAS","Lithuania","LOTI","Lesotho","MANAT","Azerbaijan","METICAL","Mozambique","NAIRA","Nigeria","NAKFA","Eritrea","NEWLIRA","Turkey","NEWSHEQEL","Israel","NGULTRUM","Bhutan","NUEVOSOL","Peru","OUGUIYA","Mauritania","PATACA","Macau","PESO","Mexico","POUND","Egypt","PULA","Botswana","QUETZAL","Guatemala","RAND","SouthAfrica","REAL","Brazil","RENMINBI","China","RIAL","Iran","RIEL","Cambodia","RINGGIT","Malaysia","RIYAL","SaudiArabia","RUBLE","Russia","RUFIYAA","Maldives","RUPEE","India","RUPEE","Pakistan","RUPIAH","Indonesia","SHILLING","Uganda","SOM","Uzbekistan","SOMONI","Tajikistan","SPECIALDRAWINGRIGHTS","InternationalMonetaryFund","TAKA","Bangladesh","TALA","WesternSamoa","TENGE","Kazakhstan","TUGRIK","Mongolia","VATU","Vanuatu","WON","Korea","YEN","Japan","ZLOTY","Poland"]
    out = map(lambda x:x.lower(), currcountry)
    currcountry = list(out)
    
    other_stop_words = datenumber + generic + currcountry
    risk_factors['risk_factor'] = risk_factors['risk_factor'].apply(lambda x: " ".join(x for x in x.split() if x not in other_stop_words))

    
    #print("Preprocessed data: \n")
    #print(df['risk_factor'])
    return risk_factors

def multilabelbinarizer(data):
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(data['label'].str.split(', '))
    return Y

def train_data():
    df = pd.read_csv("risk_data.csv")
    return prepro(df)

def model(raw_data, processed_data):
    data = train_data() 
    
    xgb = XGBClassifier(scale_pos_weight=4)
    Y = multilabelbinarizer(data)
    X_train = tuple(data['risk_factor']) 
    classifier = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=1,stop_words="english")),
    ('clf', OneVsRestClassifier(estimator=xgb))])
    classifier.fit(X_train, Y)
    
    test_data = tuple(processed_data['risk_factor'])
    
    predicted = classifier.predict(test_data)
    
    pred_labels = []
    CATEGORIES = data['label'].unique()
    for i in range(len(test_data)):  
        encoded_label = np.argmax(predicted, axis=-1)[i]
        Labels = CATEGORIES[encoded_label]
        pred_labels.append(Labels)
    
    raw_data['label'] = pred_labels
    return raw_data


# Streamlit app
def main():
    
    #HTML_WRAPPER = """<div style="overflow-x: auto; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
    HTML_WRAPPER = """<div style="overflow-x: auto; border-radius: 0.1rem; padding: 1rem; margin-bottom: 1rem">{}</div>"""
    st.title("Risk Factors")
    ticker = st.text_input('Please Enter Company Ticker', '')
    st.write(ticker)
    
    if ticker != '':
        st.subheader("Labels")
        
        color = st.color_picker('Business Risk', '#00f900')
        #st.write('The current color is', color)
        color1 = st.color_picker('Legal Risk', '#FF0000')
        #st.write('The current color is', color1)
        color2 = st.color_picker('Financial  Risk', '#FFA500')
        #st.write('The current color is', color2)
        color3 = st.color_picker('Market Risk', '#808080')
        #st.write('The current color is', color3)
        color4 = st.color_picker('Technology Risk', '#A52A2A')
        #st.write('The current color is', color4)
        color5 = st.color_picker('General Risk', '#800080')
        #st.write('The current color is', color5)
        color6 = st.color_picker('Security Risk', '#0000FF')
        #st.write('The current color is', color6)
        
        risk_factor_data = input_ticker(ticker)
        raw_data = sentences(risk_factor_data)
        preprocess = prepro(sentences(risk_factor_data))
        df = model(raw_data, preprocess)
        
        st.subheader("Highlighted Sentences")
        # Highlight and display sentences based on labels
        senten = []
        for i,j in df.iterrows():
            colour_high = highlight_sentences(j['risk_factor'], j['label'])
            colour_high = colour_high.replace("\n", " ")
            senten.append(colour_high)
            #st.write(colour_high)
        
        senten= ' '.join(senten)
        st.write(HTML_WRAPPER.format(senten), unsafe_allow_html=True)
        

if __name__ == "__main__":
    main()
