#!/usr/bin/env python
# coding: utf-8

# In[9]:


import re
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import spacy
nlp = spacy.load('es_core_news_lg')
import random
from collections import Counter
sns.set_theme(style="darkgrid")


# In[10]:


# Loading the data.

df = df = pd.read_csv('/Spanish_dataset.csv')


# In[11]:


# Dimension of df. 

df.shape


# In[12]:


# Showing the first 5 rows of the .csv 

df.head(5)


# In[13]:


# Selection of a random sample. Random module is used to make random numbers. 
# seed()	Initialize the random number generator. 
# range()	It returns an immutable sequence from 1 to the established number (in this case between 1 and 20000). 

random.seed(2000)
sample = df.letra[random.sample(range(1,9325),100)]
sample = ' '.join(sample.tolist())



# In[15]:


# The str() function is used to combine all the rows of the sample into one long string. It needs to do this because 
# nlp() takes only string. 

lyrics = str(sample).replace(r"\r\n", " ")
len(lyrics)


# In[16]:


## Pre-processing. 

#1. Lowercasing the text
text = lyrics.lower()


# In[17]:


#2. Remove punctuation. 

punct = set(string.punctuation) 
text = "".join([ch for ch in text if ch not in punct])
#print(text)


# In[18]:


#3. Encoding the text to ASCII format. 

text_encode = text.encode(encoding="ascii", errors="ignore")
# decoding the text
text_decode = text_encode.decode()
# cleaning the text to remove extra whitespace 
clean_text = " ".join([word for word in text_decode.split()])
#print(clean_text)


# In[19]:


#4. It needs to use join and isdigit to remove numeric digits from string. 

res = ''.join([i for i in clean_text if not i.isdigit()])
 


# In[20]:


#5. It needs to use the regular expression re.sub to substitute the double space between words "  " with a single
#space " ".

res2 = re.sub(' +', ' ', res)
#res2


# In[21]:


# With nlp (), spaCy segments the texts into tokens and returns the processed document. 

doc = nlp(res2)
words = [token.text
         for token in doc]


# In[22]:


# The five most common tokens of the document. 

word_freq = Counter(words)
common_words = word_freq.most_common(5)
common_words


# In[23]:


# pandas.DataFrame.from_records creates a DataFrame object from the dictionary. 
# We obtain the relationship between the frequency and the length of the tokens. 

df2 = pd.DataFrame.from_records(list(dict(Counter(words)).items()), columns=['word','frequency'])

df2 = df2.sort_values(by=['frequency'], ascending=False)
df2['rank'] = list(range(1, len(df2) + 1))
listlen=[]
for w in df2['word']:
    listlen.append(len(w))
df2['length'] = listlen
df2


# In[24]:


# We plot Zipf's power law, obtaining that while few words are very frequent, many words are used very rarely. 

sns.relplot(x="rank", y="frequency", data=df2);
plt.show()
plt.close()


# In[25]:


# Drop first row from data frame. 

df2.drop(index=df2.index[0], 
        axis=0, 
        inplace=True)

# re-do rank with new elements
df2['rank'] = list(range(1, len(df2) + 1)) 

sns.relplot(x="rank", y="frequency", data=df2);
plt.show()
plt.close()


# In[26]:


# We plot Zipf's law of abbreviation using a logarithmic scale for the variables x (length) and y (frequency).
# We obtain that while the length increase, the frequency decreases.

x = df2['length']
y= df2['frequency']

# Initialize layout
fig, ax = plt.subplots(figsize = (9, 6))

# Add scatterplot
ax.scatter(x, y, s=60, alpha=0.7, edgecolors="k");

# Set logarithmic scale on the both variables
ax.set_xscale("log")
ax.set_yscale("log")

plt.scatter(x, y)
plt.show()


# In[19]:


# We obtain that the variables' length and frequency are negatively correlated because while the length increase, 
#the frequency decreases. The non diagonal entries are the Pearson's coefficent. 

X=df2['length']
Y=df2['frequency']

np.corrcoef(X, Y)


# In[ ]:





# In[ ]:




