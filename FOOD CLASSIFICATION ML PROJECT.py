#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

get_ipython().system('pip install ipython-autotime')
get_ipython().run_line_magic('load_ext', 'autotime')






# In[2]:


import os
import io
import pickle as pkl
import matplotlib.pyplot as plt     #displaying the image
import numpy as np       # for numericals
from skimage.io import imread      # to read an image
from skimage.transform import resize    # to resize


# In[3]:


target = []
images = []
flat_data = []


# In[4]:


DATADIR = 'C:\\Users\\user\\Desktop\\new'
CATEGORIES = ['french_fries','pizza','samosa']

for category in CATEGORIES:
    class_num = CATEGORIES.index(category) # label encoding the values
    path = os.path.join(DATADIR,category) # create path to use all the images
    for img in os.listdir(path):
        img_array = imread(os.path.join(path,img))
        #print(img_array.shape)
        #plt.imshow(img_array)
        img_resized = resize(img_array,(150,150,3))
        flat_data.append(img_resized.flatten())
        images.append(img_resized)
        target.append(class_num)


# In[5]:


flat_data = np.array(flat_data)
target = np.array(target)
images = np.array(images)


# In[6]:


flat_data[0]


# In[7]:


target


# In[8]:


unique,count = np.unique(target,return_counts = True)
plt.bar(CATEGORIES,count)


# In[9]:


# split data into training and testing


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(flat_data,target,test_size = 0.3,random_state = 100)


# In[17]:


from sklearn.model_selection import GridSearchCV
from sklearn import svm


# In[18]:


param_grid = [
                {'C':[1,10,100,1000],'kernel':['linear']},
                {'C':[1,10,100,1000],'gamma':[0.001,0.0001],'kernel':['rbf']}
]


# In[19]:


svc = svm.SVC(probability = False)


# In[20]:


clf = GridSearchCV(svc,param_grid)


# In[21]:


clf.fit(x_train,y_train)


# In[22]:


y_pred = clf.predict(x_test)


# In[23]:


y_pred


# In[24]:


y_test


# In[25]:


from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_pred,y_test)
confusion_matrix(y_pred,y_test)


# In[26]:


#save model using pickle library
import pickle
pickle.dump(clf,open('img_model.p','wb'))


model = pickle.load(open('img_model.p','rb'))


# In[28]:


# testing a brand new image
flat_data = []
url = input('Enter your URL')
img = imread(url)
img_resized = resize(img,(150,150,3))
flat_data.append(img_resized.flatten())
flat_data = np.array(flat_data)
print(img.shape)
plt.imshow(img_resized)
y_out = model.predict(flat_data)
y_out = CATEGORIES[y_out[0]]
print(f'PRECDICTED OUTPUT: {y_out}')


# In[29]:


from pyngrok import ngrok


# In[35]:


get_ipython().run_cell_magic('writefile', 'food-app.py', 'import streamlit as st\nimport numpy as np  \nfrom skimage.io import imread  \nfrom skimage.transform import resize\nimport pickle\nfrom PIL import image\nst.title(\'Image classifier using Machine Learning\')\nst.text(\'Upload the Image\')\n\nmodel = pickle.load(open(\'img_model.p\',\'rb\'))\n\nuploaded_file = st.file_uploader("choose an image...", type = "jpg")\n\nif uploaded_file is not None:\n    CATEGORIES = [\'french_fries\',\'pizza\',\'samosa\']\n    st.write(\'Result...\')\n    flat_data = []\n    img = np.array(img)\n    img_resized = resize(img,(150,150,3))\n    flat_data.append(img_resized.flatten())\n    flat_data = np.array(flat_data)\n    y_out = model.predict(flat_data)\n    y_out = CATEGORIES[y_out[0]]\n    st.title(f\'PRECDICTED OUTPUT: {y_out}\')\n    for index, item in enumerate(CATEGORIES):\n        st.write(f\'{item} : {q[0][index]*100}%)\n    ')




# In[ ]:


get_ipython().system('nohup streamlit run food-app.py &')


# In[ ]:


url = ngrok.connect(port = '8501')
url


# In[ ]:




