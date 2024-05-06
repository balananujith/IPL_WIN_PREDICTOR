#!/usr/bin/env python
# coding: utf-8

# # IPL WINNING TEAM PREDICTION

# ## Import Necessary Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')


# ## Loading Datasets

# In[2]:


match = pd.read_csv("C:/Users/Effat/Desktop/SystemTronInternship/IPL_Win_Prediction/Datasets/matches.csv")
delivery = pd.read_csv("C:/Users/Effat/Desktop/SystemTronInternship/IPL_Win_Prediction/Datasets/deliveries.csv")


# ### Top 5 rows of match dataset

# In[3]:


match.head()


# In[4]:


match.shape


# ### Top 5 rows of delivery dataset

# In[5]:


delivery.head()


# # EDA- Exploratory Data Analysis

# ### Information of dataset

# In[6]:


match.info()


# ### Statistic 

# In[7]:


match.describe()


# ### Unique values of each Column

# In[8]:


# find unique values of each column
for i in match.columns:
    print("Unique value of:>>> {} ({})\n{}\n".format(i, len(match[i].unique()), match[i].unique()))


# # Data Visualization

# In[9]:


sns.countplot(x = 'Season', data = match)
plt.show()


# In[10]:


sns.countplot(x = 'toss_winner' , data = match)
plt.xticks(rotation='vertical')


# In[11]:


winneroft = match['toss_winner'] == match['winner']
winneroft.groupby(winneroft).size()
sns.countplot(winneroft)


# In[12]:


## Barplot of Runs

#sns.barplot(x="day", y="total_bill", data=tips)
fig, ax = plt.subplots()
#fig.figsize = [16,10]
#ax.set_ylim([0,20])
ax.set_xlabel("Runs")
ax.set_title("Winning by Runs - Team Performance")
#top_players.plot.bar()
sns.boxplot(y = 'winner', x = 'win_by_runs', data=match[match['win_by_runs']>0], orient = 'h'); #palette="Blues");
plt.show()

## Barplot of Wickets Win

#sns.barplot(x="day", y="total_bill", data=tips)
fig, ax = plt.subplots()
#fig.figsize = [16,10]
#ax.set_ylim([0,20])
ax.set_title("Winning by Wickets - Team Performance")
#top_players.plot.bar()
sns.boxplot(y = 'winner', x = 'win_by_wickets', data=match[match['win_by_wickets']>0], orient = 'h'); #palette="Blues");
plt.show()


# # Data Preprocessing 

# In[13]:


#groupby total runs of both the innings of each match and convert into data frames
total_score = delivery.groupby(['match_id','inning'])['total_runs'].sum().reset_index() 


# In[14]:


total_score = total_score[total_score['inning']==1]
total_score['total_runs'] = total_score['total_runs'].apply(lambda x:x+1)


# In[15]:


#1528 rows = each match(756) * 2(each match has 2 innings)
total_score


# In[16]:


#filter data for 1st inning 
total_score = total_score[total_score['inning'] == 1]


# In[17]:


total_score


# In[18]:


#merge total_score in 1st inning with match data frame to get all the details of match
match_df = match.merge(total_score[['match_id','total_runs']],left_on='id',right_on='match_id')


# In[19]:


match_df


# In[20]:


#display all the unique value of team1
match_df['team1'].unique()


# In[21]:


#make list of all all unique teams
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians', 
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings', 
    'Rajasthan Royals',
    'Delhi Capitals'
]


# In[22]:


#replace team's old name to new name 
match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')


# In[23]:


#teams which are playing IPL currently
match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]


# In[24]:


match_df.shape


# In[25]:


match_df


# In[26]:


#counts number of matches affected by rains and check by dl(Duckworth-Lewis)
match_df['dl_applied'].value_counts()


# In[27]:


#filter out 15 matches affected by rains
match_df = match_df[match_df['dl_applied'] == 0]


# In[28]:


match_df


# In[29]:


#access the required records from match_df for evaluation
match_df = match_df[['match_id','city','winner','total_runs']]


# In[30]:


#merge with the delivery table with common attribute match_id for required columns
delivery_df = match_df.merge(delivery,on='match_id')


# In[31]:


delivery_df 


# In[32]:


#access the column required for team2 to chase
delivery_df = delivery_df[delivery_df['inning'] == 2]


# In[33]:


delivery_df


# In[34]:


#calculate total runs after each ball
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()


# In[35]:


delivery_df


# In[36]:


#by subtracting total_runs_x(required runs to win for 2nd team) - current_score(till how many runs made) = runs_left(required runs to chase team1)
delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']


# In[37]:


delivery_df


# In[38]:


# balls left after each ball for each over
delivery_df['balls_left'] = 126 - (delivery_df['over']*6 + delivery_df['ball'])


# In[39]:


delivery_df


# In[40]:


#calculate how many wickets left after each ball
delivery_df['player_dismissed']=delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed']=delivery_df['player_dismissed'].apply(lambda x:x if x=="0" else "1")
delivery_df['player_dismissed'] =delivery_df['player_dismissed'].astype('int')
wickets=delivery_df.groupby('match_id')['player_dismissed'].cumsum().values
delivery_df['wickets']=10-wickets
delivery_df


# In[41]:


#kitni balls khel chuke hain vo nikalenge 120 mai se balls left minus kr k
#crr=runs/overs
delivery_df['crr']=(delivery_df['current_score']*6)/(120-delivery_df['balls_left'])


# In[42]:


#find required run rate by dividing runs left with no of overs left
delivery_df['rrr']=(delivery_df['runs_left']*6)/delivery_df['balls_left']


# In[43]:


delivery_df


# In[44]:


sns.heatmap(delivery_df.isnull(), cbar = False, yticklabels = False, cmap = 'viridis')


# In[45]:


#check if batting team is same as winner then return 1 else return 0
def result(row):
    return 1 if row['batting_team']==row['winner'] else 0


# In[46]:


#find result on y axis
delivery_df['result']=delivery_df.apply(result,axis=1)


# In[47]:


delivery_df


# In[48]:


final_df=delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]


# In[49]:


final_df


# In[50]:


#shuffling in fianl_df to avoid biasness
final_df.sample(final_df.shape[0])


# In[51]:


#randomly access any record
final_df.sample()


# In[52]:


#number of rows in each column having null values
final_df.isnull().sum()


# In[53]:


#drop all row with null values
final_df.dropna(inplace=True)


# In[54]:


# drop column with balls_left are 0 because values of required run rate reaches to infinity
final_df=final_df[final_df['balls_left']!=0]


# # Model Development

# ### Splitting of training and testing data

# In[55]:


X=final_df.iloc[:,:-1]
y=final_df.iloc[:,-1]


# In[56]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)


# In[57]:


X_train


# ### Column Transformer

# In[58]:


#we transform the catgorical column into numeric values with the use of columntransformer and one hot encoder

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
trf=ColumnTransformer([
    ('trf',OneHotEncoder(sparse_output=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough')
                      


# # Logistic Regression

# In[59]:


#import logistic and pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# In[60]:


#apply logistic regression , we apply LR bcz its give more accurate result rather than randomforestclassify
pipe=Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
    ])


# In[61]:


#train the model
pipe.fit(X_train,y_train)


# In[62]:


#predict the y
y_pred = pipe.predict(X_test)


# In[63]:


#find accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[64]:


pipe.predict_proba(X_test)[10]


# # Saving Model

# In[65]:


import joblib
# Saving the entire pipeline
joblib.dump(pipe, 'C:/Users/Effat/Desktop/SystemTronInternship/IPL_Win_Prediction/Model/ipl_lr_model.pkl')


# In[66]:


# creating a summary like: "Batting Team-[team_name] | Bowling Team-[team_name] | Target-[target_score]"
def match_summary(row):
    print("Batting Team-" + row['batting_team'] + " | Bowling Team-" + row['bowling_team'] + " | Target- " + str(row['total_runs_x']))   


# In[67]:


#function to determine the performance of the chasing team after each over for diff match id's
def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id] 
    match = match[(match['ball'] == 6)]
    temp_df = match[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target


# In[68]:


#performence for the match id 74
temp_df,target = match_progression(delivery_df,74,pipe)
temp_df


# In[69]:


import matplotlib.pyplot as plt
plt.figure(figsize=(18,8))
plt.plot(temp_df['end_of_over'],temp_df['wickets_in_over'],color='yellow',linewidth=3)
plt.plot(temp_df['end_of_over'],temp_df['win'],color='#00a65a',linewidth=4)
plt.plot(temp_df['end_of_over'],temp_df['lose'],color='red',linewidth=4)
plt.bar(temp_df['end_of_over'],temp_df['runs_after_over'])
plt.title('Target-' + str(target))


# In[70]:


temp_df,target = match_progression(delivery_df,7,pipe)
temp_df


# In[71]:


import matplotlib.pyplot as plt
plt.figure(figsize=(18,8))
plt.plot(temp_df['end_of_over'],temp_df['wickets_in_over'],color='yellow',linewidth=3)
plt.plot(temp_df['end_of_over'],temp_df['win'],color='#00a65a',linewidth=4)
plt.plot(temp_df['end_of_over'],temp_df['lose'],color='red',linewidth=4)
plt.bar(temp_df['end_of_over'],temp_df['runs_after_over'])
plt.title('Target-' + str(target))


# In[72]:


temp_df,target = match_progression(delivery_df,14,pipe)
temp_df


# In[73]:


import matplotlib.pyplot as plt
plt.figure(figsize=(18,8))
plt.plot(temp_df['end_of_over'],temp_df['wickets_in_over'],color='yellow',linewidth=3)
plt.plot(temp_df['end_of_over'],temp_df['win'],color='#00a65a',linewidth=4)
plt.plot(temp_df['end_of_over'],temp_df['lose'],color='red',linewidth=4)
plt.bar(temp_df['end_of_over'],temp_df['runs_after_over'])
plt.title('Target-' + str(target))


# In[74]:


teams


# In[75]:


delivery_df['city'].unique()


# # Thank You
