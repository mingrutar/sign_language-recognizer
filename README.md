
# Artificial Intelligence Engineer Nanodegree - Probabilistic Models
## Project: Sign Language Recognition System
- [Introduction](#intro)
- [Part 1 Feature Selection](#part1_tutorial)
    - [Tutorial](#part1_tutorial)
    - [Features Submission](#part1_submission)
    - [Features Unittest](#part1_test)
- [Part 2 Train the models](#part2_tutorial)
    - [Tutorial](#part2_tutorial)
    - [Model Selection Score Submission](#part2_submission)
    - [Model Score Unittest](#part2_test)
- [Part 3 Build a Recognizer](#part3_tutorial)
    - [Tutorial](#part3_tutorial)
    - [Recognizer Submission](#part3_submission)
    - [Recognizer Unittest](#part3_test)
- [Part 4 (OPTIONAL) Improve the WER with Language Models](#part4_info)

<b>To use hmmlearn</b>: use Visual Studio 2017 buildtool Cross Tools, activate py35


<a id='intro'></a>
## Introduction
The overall goal of this project is to build a word recognizer for American Sign Language video sequences, demonstrating the power of probabalistic models.  In particular, this project employs  [hidden Markov models (HMM's)](https://en.wikipedia.org/wiki/Hidden_Markov_model) to analyze a series of measurements taken from videos of American Sign Language (ASL) collected for research (see the [RWTH-BOSTON-104 Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-104.php)).  In this video, the right-hand x and y locations are plotted as the speaker signs the sentence.
[![ASLR demo](http://www-i6.informatik.rwth-aachen.de/~dreuw/images/demosample.png)](https://drive.google.com/open?id=0B_5qGuFe-wbhUXRuVnNZVnMtam8)

The raw data, train, and test sets are pre-defined.  You will derive a variety of feature sets (explored in Part 1), as well as implement three different model selection criterion to determine the optimal number of hidden states for each word model (explored in Part 2). Finally, in Part 3 you will implement the recognizer and compare the effects the different combinations of feature sets and model selection criteria.  

At the end of each Part, complete the submission cells with implementations, answer all questions, and pass the unit tests.  Then submit the completed notebook for review!

<a id='part1_tutorial'></a>
## PART 1: Data

### Features Tutorial
##### Load the initial database
A data handler designed for this database is provided in the student codebase as the `AslDb` class in the `asl_data` module.  This handler creates the initial [pandas](http://pandas.pydata.org/pandas-docs/stable/) dataframe from the corpus of data included in the `data` directory as well as dictionaries suitable for extracting data in a format friendly to the [hmmlearn](https://hmmlearn.readthedocs.io/en/latest/) library.  We'll use those to create models in Part 2.

To start, let's set up the initial database and select an example set of features for the training set.  At the end of Part 1, you will create additional feature sets for experimentation.


```python
import numpy as np
import pandas as pd
from asl_data import AslDb


asl = AslDb() # initializes the database
asl.df.head() # displays the first five rows of the asl database, indexed by video and frame
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>left-x</th>
      <th>left-y</th>
      <th>right-x</th>
      <th>right-y</th>
      <th>nose-x</th>
      <th>nose-y</th>
      <th>speaker</th>
    </tr>
    <tr>
      <th>video</th>
      <th>frame</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">98</th>
      <th>0</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
    </tr>
  </tbody>
</table>
</div>




```python
asl.df.ix[98,1]  # look at the data available for an individual frame
```

    C:\ProgramData\Anaconda3\envs\py35\lib\site-packages\ipykernel\__main__.py:1: DeprecationWarning:
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing

    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      if __name__ == '__main__':





    left-x         149
    left-y         181
    right-x        170
    right-y        175
    nose-x         161
    nose-y          62
    speaker    woman-1
    Name: (98, 1), dtype: object



The frame represented by video 98, frame 1 is shown here:
![Video 98](http://www-i6.informatik.rwth-aachen.de/~dreuw/database/rwth-boston-104/overview/images/orig/098-start.jpg)

##### Feature selection for training the model
The objective of feature selection when training a model is to choose the most relevant variables while keeping the model as simple as possible, thus reducing training time.  We can use the raw features already provided or derive our own and add columns to the pandas dataframe `asl.df` for selection. As an example, in the next cell a feature named `'grnd-ry'` is added. This feature is the difference between the right-hand y value and the nose y value, which serves as the "ground" right y value.


```python
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df.head()  # the new feature 'grnd-ry' is now in the frames dictionary
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>left-x</th>
      <th>left-y</th>
      <th>right-x</th>
      <th>right-y</th>
      <th>nose-x</th>
      <th>nose-y</th>
      <th>speaker</th>
      <th>grnd-ry</th>
    </tr>
    <tr>
      <th>video</th>
      <th>frame</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">98</th>
      <th>0</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
    </tr>
    <tr>
      <th>1</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
    </tr>
    <tr>
      <th>2</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
    </tr>
    <tr>
      <th>3</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
    </tr>
    <tr>
      <th>4</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
    </tr>
  </tbody>
</table>
</div>



##### Try it!


```python
from asl_utils import test_features_tryit
# TODO add df columns for 'grnd-rx', 'grnd-ly', 'grnd-lx' representing differences between hand and nose locations
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

# test the code
test_features_tryit(asl)
```

    asl.df sample



<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>left-x</th>
      <th>left-y</th>
      <th>right-x</th>
      <th>right-y</th>
      <th>nose-x</th>
      <th>nose-y</th>
      <th>speaker</th>
      <th>grnd-ry</th>
      <th>grnd-rx</th>
      <th>grnd-ly</th>
      <th>grnd-lx</th>
    </tr>
    <tr>
      <th>video</th>
      <th>frame</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">98</th>
      <th>0</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
    </tr>
  </tbody>
</table>
</div>





<font color=green>Correct!</font><br/>




```python
# collect the features into a list
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
 #show a single set of features for a given (video, frame) tuple
[asl.df.ix[98,1][v] for v in features_ground]
```




    [9, 113, -12, 119]



##### Build the training set
Now that we have a feature list defined, we can pass that list to the `build_training` method to collect the features for all the words in the training set.  Each word in the training set has multiple examples from various videos.  Below we can see the unique words that have been loaded into the training set:


```python
training = asl.build_training(features_ground)
print("Training words: {}".format(training.words))
```

    Training words: ['JANA', 'TOMORROW', 'BORROW', 'NOT', 'SHOULD', 'APPLE', 'FRIEND', 'HERE', 'PEOPLE', 'FUTURE', 'KNOW', 'TOY1', 'STOLEN', 'BLAME', 'CAN', 'FISH', 'OLD', 'HOMEWORK', 'GO', 'VEGETABLE', 'MOTHER', 'VIDEOTAPE', 'FIND', 'BROCCOLI', 'GO1', 'GIRL', 'BUY', 'SAY-1P', 'BROTHER', 'SELF', 'PARTY', 'FUTURE1', 'ARRIVE', 'PUTASIDE', 'WANT', 'BOY', 'GIVE', 'TELL', 'PAST', 'CORN1', 'IX-1P', 'HIT', 'SUE', 'READ', 'ALL', 'THROW', 'THINK', 'JOHN', 'WONT', 'FINISH', 'PREFER', 'SELL', 'SEARCH-FOR', 'MOVIE', 'IX', 'BOOK', 'YESTERDAY', 'SEE', 'EAT', 'BLUE', 'LAST-WEEK', 'MANY', 'TOY', 'HOUSE', 'CHINA', 'WOMAN', 'LIKE', 'NEW-YORK', 'LIVE', 'NEXT-WEEK', 'GIVE2', 'WHAT', 'NEW', 'LEG', 'DECIDE', 'HAVE', 'SAY', 'GO2', 'LOVE', 'CHICKEN', 'COAT', 'CORN', 'BUT', 'SOMETHING-ONE', 'POTATO', 'MARY', 'CANDY', 'TEACHER', 'WHO', 'GIVE1', 'BUY1', 'WILL', 'VISIT', 'LEAVE', 'NAME', 'CHICAGO', 'GET', 'GIVE3', 'BREAK-DOWN', 'BILL', 'CHOCOLATE', 'WRITE', 'GROUP', 'STUDENT', 'SHOOT', 'BOX', 'POSS', 'CAR', 'FRED', 'MAN', 'FRANK', 'ANN']


The training data in `training` is an object of class `WordsData` defined in the `asl_data` module.  in addition to the `words` list, data can be accessed with the `get_all_sequences`, `get_all_Xlengths`, `get_word_sequences`, and `get_word_Xlengths` methods. We need the `get_word_Xlengths` method to train multiple sequences with the `hmmlearn` library.  In the following example, notice that there are two lists; the first is a concatenation of all the sequences(the X portion) and the second is a list of the sequence lengths(the Lengths portion).


```python
training.get_word_Xlengths('CHOCOLATE')
```




    (array([[-11,  48,   7, 120],
            [-11,  48,   8, 109],
            [ -8,  49,  11,  98],
            [ -7,  50,   7,  87],
            [ -4,  54,   7,  77],
            [ -4,  54,   6,  69],
            [ -4,  54,   6,  69],
            [-13,  52,   6,  69],
            [-13,  52,   6,  69],
            [ -8,  51,   6,  69],
            [ -8,  51,   6,  69],
            [ -8,  51,   6,  69],
            [ -8,  51,   6,  69],
            [ -8,  51,   6,  69],
            [-10,  59,   7,  71],
            [-15,  64,   9,  77],
            [-17,  75,  13,  81],
            [ -4,  48,  -4, 113],
            [ -2,  53,  -4, 113],
            [ -4,  55,   2,  98],
            [ -4,  58,   2,  98],
            [ -1,  59,   2,  89],
            [ -1,  59,  -1,  84],
            [ -1,  59,  -1,  84],
            [ -7,  63,  -1,  84],
            [ -7,  63,  -1,  84],
            [ -7,  63,   3,  83],
            [ -7,  63,   3,  83],
            [ -7,  63,   3,  83],
            [ -7,  63,   3,  83],
            [ -7,  63,   3,  83],
            [ -7,  63,   3,  83],
            [ -7,  63,   3,  83],
            [ -4,  70,   3,  83],
            [ -4,  70,   3,  83],
            [ -2,  73,   5,  90],
            [ -3,  79,  -4,  96],
            [-15,  98,  13, 135],
            [ -6,  93,  12, 128],
            [ -2,  89,  14, 118],
            [  5,  90,  10, 108],
            [  4,  86,   7, 105],
            [  4,  86,   7, 105],
            [  4,  86,  13, 100],
            [ -3,  82,  14,  96],
            [ -3,  82,  14,  96],
            [  6,  89,  16, 100],
            [  6,  89,  16, 100],
            [  7,  85,  17, 111]], dtype=int64), [17, 20, 12])



###### More feature sets
So far we have a simple feature set that is enough to get started modeling.  However, we might get better results if we manipulate the raw values a bit more, so we will go ahead and set up some other options now for experimentation later.  For example, we could normalize each speaker's range of motion with grouped statistics using [Pandas stats](http://pandas.pydata.org/pandas-docs/stable/api.html#api-dataframe-stats) functions and [pandas groupby](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html).  Below is an example for finding the means of all speaker subgroups.


```python
df_means = asl.df.groupby('speaker').mean()
df_means
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left-x</th>
      <th>left-y</th>
      <th>right-x</th>
      <th>right-y</th>
      <th>nose-x</th>
      <th>nose-y</th>
      <th>grnd-ry</th>
      <th>grnd-rx</th>
      <th>grnd-ly</th>
      <th>grnd-lx</th>
    </tr>
    <tr>
      <th>speaker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>man-1</th>
      <td>206.248203</td>
      <td>218.679449</td>
      <td>155.464350</td>
      <td>150.371031</td>
      <td>175.031756</td>
      <td>61.642600</td>
      <td>88.728430</td>
      <td>-19.567406</td>
      <td>157.036848</td>
      <td>31.216447</td>
    </tr>
    <tr>
      <th>woman-1</th>
      <td>164.661438</td>
      <td>161.271242</td>
      <td>151.017865</td>
      <td>117.332462</td>
      <td>162.655120</td>
      <td>57.245098</td>
      <td>60.087364</td>
      <td>-11.637255</td>
      <td>104.026144</td>
      <td>2.006318</td>
    </tr>
    <tr>
      <th>woman-2</th>
      <td>183.214509</td>
      <td>176.527232</td>
      <td>156.866295</td>
      <td>119.835714</td>
      <td>170.318973</td>
      <td>58.022098</td>
      <td>61.813616</td>
      <td>-13.452679</td>
      <td>118.505134</td>
      <td>12.895536</td>
    </tr>
  </tbody>
</table>
</div>



To select a mean that matches by speaker, use the pandas [map](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.map.html) method:


```python
asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
asl.df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>left-x</th>
      <th>left-y</th>
      <th>right-x</th>
      <th>right-y</th>
      <th>nose-x</th>
      <th>nose-y</th>
      <th>speaker</th>
      <th>grnd-ry</th>
      <th>grnd-rx</th>
      <th>grnd-ly</th>
      <th>grnd-lx</th>
      <th>left-x-mean</th>
    </tr>
    <tr>
      <th>video</th>
      <th>frame</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">98</th>
      <th>0</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
      <td>164.661438</td>
    </tr>
    <tr>
      <th>1</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
      <td>164.661438</td>
    </tr>
    <tr>
      <th>2</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
      <td>164.661438</td>
    </tr>
    <tr>
      <th>3</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
      <td>164.661438</td>
    </tr>
    <tr>
      <th>4</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
      <td>164.661438</td>
    </tr>
  </tbody>
</table>
</div>



##### Try it!


```python
from asl_utils import test_std_tryit
# TODO Create a dataframe named `df_std` with standard deviations grouped by speaker
df_std = asl.df.groupby('speaker').std()
# test the code
test_std_tryit(df_std)
```

    df_std



<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left-x</th>
      <th>left-y</th>
      <th>right-x</th>
      <th>right-y</th>
      <th>nose-x</th>
      <th>nose-y</th>
      <th>grnd-ry</th>
      <th>grnd-rx</th>
      <th>grnd-ly</th>
      <th>grnd-lx</th>
      <th>left-x-mean</th>
    </tr>
    <tr>
      <th>speaker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>man-1</th>
      <td>15.154425</td>
      <td>36.328485</td>
      <td>18.901917</td>
      <td>54.902340</td>
      <td>6.654573</td>
      <td>5.520045</td>
      <td>53.487999</td>
      <td>20.269032</td>
      <td>36.572749</td>
      <td>15.080360</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>woman-1</th>
      <td>17.573442</td>
      <td>26.594521</td>
      <td>16.459943</td>
      <td>34.667787</td>
      <td>3.549392</td>
      <td>3.538330</td>
      <td>33.972660</td>
      <td>16.764706</td>
      <td>27.117393</td>
      <td>17.328941</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>woman-2</th>
      <td>15.388711</td>
      <td>28.825025</td>
      <td>14.890288</td>
      <td>39.649111</td>
      <td>4.099760</td>
      <td>3.416167</td>
      <td>39.128572</td>
      <td>16.191324</td>
      <td>29.320655</td>
      <td>15.050938</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>





<font color=green>Correct!</font><br/>



<a id='part1_submission'></a>
### Features Implementation Submission
Implement four feature sets and answer the question that follows.
- normalized Cartesian coordinates
    - use *mean* and *standard deviation* statistics and the [standard score](https://en.wikipedia.org/wiki/Standard_score) equation to account for speakers with different heights and arm length

- polar coordinates
    - calculate polar coordinates with [Cartesian to polar equations](https://en.wikipedia.org/wiki/Polar_coordinate_system#Converting_between_polar_and_Cartesian_coordinates)
    - use the [np.arctan2](https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.arctan2.html) function and *swap the x and y axes* to move the $0$ to $2\pi$ discontinuity to 12 o'clock instead of 3 o'clock;  in other words, the normal break in radians value from $0$ to $2\pi$ occurs directly to the left of the speaker's nose, which may be in the signing area and interfere with results.  By swapping the x and y axes, that discontinuity move to directly above the speaker's head, an area not generally used in signing.

- delta difference
    - as described in Thad's lecture, use the difference in values between one frame and the next frames as features
    - pandas [diff method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.diff.html) and [fillna method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html) will be helpful for this one

- custom features
    - These are your own design; combine techniques used above or come up with something else entirely. We look forward to seeing what you come up with!
    Some ideas to get you started:
        - normalize using a [feature scaling equation](https://en.wikipedia.org/wiki/Feature_scaling)
        - normalize the polar coordinates
        - adding additional deltas



```python
# TODO add features for normalized by speaker values of left, right, x, y
# Name these 'norm-rx', 'norm-ry', 'norm-lx', and 'norm-ly'
# using Z-score scaling (X-Xmean)/Xstd

features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
#asl.df['norm-rx'] = asl.df['right-x']/asl.df['speaker'].map(df_std['right-x'])-(df_means['right-x']/df_std['right-x'])
asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['speaker'].map(df_means['right-x'])) \
    /asl.df['speaker'].map(df_std['right-x'])
asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['speaker'].map(df_means['right-y'])) \
    /asl.df['speaker'].map(df_std['right-y'])
asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['speaker'].map(df_means['left-x'])) \
    /asl.df['speaker'].map(df_std['left-x'])
asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['speaker'].map(df_means['left-y'])) \
    /asl.df['speaker'].map(df_std['left-y'])
[asl.df.ix[98,1][v] for v in features_norm]
```




    [1.1532321114002382,
     1.6634329223668574,
     -0.89119923044101379,
     0.74183544610811614]




```python
# TODO add features for polar coordinate values where the nose is the origin
# Name these 'polar-rr', 'polar-rtheta', 'polar-lr', and 'polar-ltheta'
# Note that 'polar-rr' and 'polar-rtheta' refer to the radius and angle

features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
asl.df['polar-rr'] = np.sqrt((asl.df['right-x']- asl.df['nose-x'])**2 + (asl.df['right-y']-asl.df['nose-y'])**2)
asl.df['polar-rtheta'] = np.arctan2(asl.df['right-x']- asl.df['nose-x'],asl.df['right-y'] - asl.df['nose-y'])
asl.df['polar-lr'] = np.sqrt((asl.df['left-x']-asl.df['nose-x'])**2 + (asl.df['left-y']-asl.df['nose-y'])**2)
asl.df['polar-ltheta'] = np.arctan2(asl.df['left-x']- asl.df['nose-x'], asl.df['left-y'] - asl.df['nose-y'])
[asl.df.ix[98,1][v] for v in features_polar]
```




    [113.35784048754634,
     0.079478244608206572,
     119.60351165413162,
     -0.10050059905462982]




```python
# TODO add features for left, right, x, y differences by one time step, i.e. the "delta" values discussed in the lecture
# Name these 'delta-rx', 'delta-ry', 'delta-lx', and 'delta-ly'

features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']
asl.df['delta-rx'] = asl.df['right-x'].diff().fillna(0)
asl.df['delta-ry'] = asl.df['right-y'].diff().fillna(0)
asl.df['delta-lx'] = asl.df['left-x'].diff().fillna(0)
asl.df['delta-ly'] = asl.df['left-y'].diff().fillna(0)
[asl.df.ix[98,35][v] for v in features_delta]
```




    [0.0, 0.0, 6.0, -9.0]




```python
[asl.df.ix[98,18][v] for v in features_delta]
```




    [-14.0, -9.0, 0.0, 0.0]



Custom feature <b>features_podel</b> is feature_polar plus delta over feature_polar.
<br>
feature_polar captures the characteristics on a frame and the delat captures the change characteristics between 2 frames.


```python
# TODO add features of your own design, which may be a combination of the above or something else
# Name these whatever you would like
features_podel = ['podel-rr', 'podel-rtheta', 'podel-lr', 'podel-ltheta']
asl.df['podel-rr'] = asl.df['polar-rr'] + asl.df['polar-rr'].diff().fillna(0)
asl.df['podel-rtheta'] = asl.df['polar-rtheta'] + asl.df['polar-rtheta'].diff().fillna(0)
asl.df['podel-lr'] = asl.df['polar-lr'] + asl.df['polar-lr'].diff().fillna(0)
asl.df['podel-ltheta'] = asl.df['polar-ltheta'] + asl.df['polar-ltheta'].diff().fillna(0)
[asl.df.ix[98,36][v] for v in features_podel]
# TODO define a list named 'features_custom' for building the training set
```




    [29.698484809834994,
     -0.78539816339744828,
     114.83332419142495,
     -0.12983714941684962]




```python
df_means = asl.df.groupby('speaker').mean()
df_means
df_std = asl.df.groupby('speaker').std()
df_std
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left-x</th>
      <th>left-y</th>
      <th>right-x</th>
      <th>right-y</th>
      <th>nose-x</th>
      <th>nose-y</th>
      <th>grnd-ry</th>
      <th>grnd-rx</th>
      <th>grnd-ly</th>
      <th>grnd-lx</th>
      <th>...</th>
      <th>s-polar-ltheta</th>
      <th>s-podel-ltheta</th>
      <th>scaled-rx</th>
      <th>delta-srx</th>
      <th>scaled-ry</th>
      <th>delta-sry</th>
      <th>scaled-lx</th>
      <th>delta-slx</th>
      <th>scaled-ly</th>
      <th>delta-sly</th>
    </tr>
    <tr>
      <th>speaker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>man-1</th>
      <td>15.154425</td>
      <td>36.328485</td>
      <td>18.901917</td>
      <td>54.902340</td>
      <td>6.654573</td>
      <td>5.520045</td>
      <td>53.487999</td>
      <td>20.269032</td>
      <td>36.572749</td>
      <td>15.080360</td>
      <td>...</td>
      <td>0.096866</td>
      <td>0.022840</td>
      <td>0.106790</td>
      <td>0.023528</td>
      <td>0.239748</td>
      <td>0.041177</td>
      <td>0.097144</td>
      <td>0.029449</td>
      <td>0.200710</td>
      <td>0.035258</td>
    </tr>
    <tr>
      <th>woman-1</th>
      <td>17.573442</td>
      <td>26.594521</td>
      <td>16.459943</td>
      <td>34.667787</td>
      <td>3.549392</td>
      <td>3.538330</td>
      <td>33.972660</td>
      <td>16.764706</td>
      <td>27.117393</td>
      <td>17.328941</td>
      <td>...</td>
      <td>0.130860</td>
      <td>0.027785</td>
      <td>0.092994</td>
      <td>0.021836</td>
      <td>0.151388</td>
      <td>0.028849</td>
      <td>0.112650</td>
      <td>0.031445</td>
      <td>0.146931</td>
      <td>0.027798</td>
    </tr>
    <tr>
      <th>woman-2</th>
      <td>15.388711</td>
      <td>28.825025</td>
      <td>14.890288</td>
      <td>39.649111</td>
      <td>4.099760</td>
      <td>3.416167</td>
      <td>39.128572</td>
      <td>16.191324</td>
      <td>29.320655</td>
      <td>15.050938</td>
      <td>...</td>
      <td>0.108254</td>
      <td>0.021825</td>
      <td>0.084126</td>
      <td>0.018984</td>
      <td>0.173140</td>
      <td>0.029317</td>
      <td>0.098646</td>
      <td>0.023361</td>
      <td>0.159254</td>
      <td>0.027689</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 53 columns</p>
</div>



Custom feature <b>features_norm_pol</b> normalizes the feature_polar by speakers.


```python
features_norm_pol = ['norm-rr', 'norm-rtheta', 'norm-lr','norm-ltheta']
#asl.df['norm-rx'] = asl.df['right-x']/asl.df['speaker'].map(df_std['right-x'])-(df_means['right-x']/df_std['right-x'])
asl.df['norm-rr'] = (asl.df['polar-rr'] - asl.df['speaker'].map(df_means['polar-rr'])) \
    /asl.df['speaker'].map(df_std['polar-rr'])
asl.df['norm-rtheta'] = (asl.df['polar-rtheta'] - asl.df['speaker'].map(df_means['polar-rtheta'])) \
    /asl.df['speaker'].map(df_std['polar-rtheta'])
asl.df['norm-lr'] = (asl.df['polar-lr'] - asl.df['speaker'].map(df_means['polar-lr'])) \
    /asl.df['speaker'].map(df_std['polar-lr'])
asl.df['norm-ltheta'] = (asl.df['polar-ltheta'] - asl.df['speaker'].map(df_means['polar-ltheta'])) \
    /asl.df['speaker'].map(df_std['polar-ltheta'])
[asl.df.ix[98,1][v] for v in features_norm_pol]
```




    [1.5734394584886571,
     0.95941868201635139,
     0.54249851337954813,
     -0.73521895146904148]



Custom feature <b>features_norm_poldel</b> normalizes the features_podel by speakers.


```python
features_norm_poldel = ['normpd-rr', 'normpd-rtheta', 'normpd-lr','normpd-ltheta']
asl.df['normpd-rr'] = asl.df['norm-rr'] + asl.df['norm-rr'].diff().fillna(0)
asl.df['normpd-rtheta'] = asl.df['norm-rtheta'] + asl.df['norm-rtheta'].diff().fillna(0)
asl.df['normpd-lr'] = asl.df['norm-lr'] + asl.df['norm-lr'].diff().fillna(0)
asl.df['normpd-ltheta'] = asl.df['norm-ltheta'] + asl.df['norm-ltheta'].diff().fillna(0)
[asl.df.ix[98,36][v] for v in features_norm_poldel]
```




    [-1.1557046744922284,
     -1.2145301794899983,
     0.35270699978281728,
     -0.86910200185638209]



Per reviewer suggestion, add x and y distance between hands into features_grnd


```python
features_grnd_distance = ['grnd-rx', 'grnd-ry', 'grnd-lx','grnd-ly','grnd-lrx','grnd-lry']
# TODO add df columns for 'grnd-rx', 'grnd-ly', 'grnd-lx' representing differences between hand and nose locations
asl.df['grnd-lrx'] = asl.df['left-x'] - asl.df['right-x']
asl.df['grnd-lry'] = asl.df['left-y'] - asl.df['right-y']
[asl.df.ix[98,36][v] for v in features_grnd_distance]
```




    [-21, 21, -11, 116, 10, 95]




```python
polar_col = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
scale_podel = ['s-polar-rr', 's-polar-rtheta', 's-polar-lr', 's-polar-ltheta', 's-podel-rr', 's-podel-rtheta', 's-podel-lr', 's-podel-ltheta']  
# asl.df['podel-rr'] = asl.df['polar-rr'].diff().fillna(0)
# asl.df['podel-rtheta'] = asl.df['polar-rtheta'].diff().fillna(0)
# asl.df['podel-lr'] = asl.df['polar-lr'].diff().fillna(0)
# asl.df['podel-ltheta'] = asl.df['polar-ltheta'].diff().fillna(0)

x = 0
y = 4
for i in polar_col[:4]:
    asl.df[scale_podel[x]]=(asl.df[i] - asl.df[i].min())/(asl.df[i].max() - asl.df[i].min())
    asl.df[scale_podel[y]]= asl.df[scale_podel[x]].diff().fillna(0)
    x += 1
    y += 1
[asl.df.ix[98,36][v] for v in scale_podel]
```




    [0.14403510666011562,
     0.34603403843650665,
     0.53903843375864358,
     0.11052015814697957,
     0.0,
     0.0,
     -0.010250404553777459,
     -0.021076766881381548]




```python
col = ['right-x', 'right-y','left-x', 'left-y']
features_custom2 = ['scaled-rx', 'scaled-ry', 'scaled-lx', 'scaled-ly', 'delta-srx', 'delta-sry', 'delta-slx', 'delta-sly']

x = 0
y = 4
for i in col:
    asl.df[features_custom2[x]]=(asl.df[i] - asl.df[i].min())/(asl.df[i].max() - asl.df[i].min())
    asl.df[features_custom2[y]]= asl.df[features_custom2[x]].diff().fillna(0)
    x += 1
    y += 1
[asl.df.ix[98,36][v] for v in features_custom2]
```




    [0.49152542372881358,
     0.23144104803493451,
     0.10897435897435898,
     0.5524861878453039,
     0.0,
     0.0,
     -0.025641025641025633,
     -0.011049723756906049]



**Question 1:**  What custom features did you choose for the features_custom set and why?

**Answer 1:**

The feature polar performed the best for all selectors, so I customize 3 features based on feature_polar.

1. features_norm_pol: normalize the feature_polar
2. features_podel: add delta time to feature_polar
3. features_norm_poldel: normalize the features_podel
4. features_grnd_distance: features_grnd plus x and y distances between hands

The below is the test results.  

|                      |     |SelectorBIC      |     |  SelectorDIC     |
| -------------------- |:---:|:---------------:|:---:|:----------------:|
| <b>features name</b> | WER | number regonized| WER | number regonized |
| features_norm_poldel | 0.5842 | 74 | 0.5896 | 73 |
| features_podel       | 0.5    | 89 | 0.5337 | 83 |
| features_norm_pol    | 0.5842 | 74 | 0.5674 | 77 |
| features_grnd_distance    | 0.5449 | 81 | NA | NA |


<a id='part1_test'></a>
### Features Unit Testing
Run the following unit tests as a sanity check on the defined "ground", "norm", "polar", and 'delta"
feature sets.  The test simply looks for some valid values but is not exhaustive.  However, the project should not be submitted if these tests don't pass.


```python
import unittest
# import numpy as np

class TestFeatures(unittest.TestCase):

    def test_features_ground(self):
        sample = (asl.df.ix[98, 1][features_ground]).tolist()
        self.assertEqual(sample, [9, 113, -12, 119])

    def test_features_norm(self):
        sample = (asl.df.ix[98, 1][features_norm]).tolist()
        np.testing.assert_almost_equal(sample, [ 1.153,  1.663, -0.891,  0.742], 3)

    def test_features_polar(self):
        sample = (asl.df.ix[98,1][features_polar]).tolist()
        np.testing.assert_almost_equal(sample, [113.3578, 0.0794, 119.603, -0.1005], 3)

    def test_features_delta(self):
        sample = (asl.df.ix[98, 0][features_delta]).tolist()
        self.assertEqual(sample, [0, 0, 0, 0])
        sample = (asl.df.ix[98, 18][features_delta]).tolist()
        self.assertTrue(sample in [[-16, -5, -2, 4], [-14, -9, 0, 0]], "Sample value found was {}".format(sample))

suite = unittest.TestLoader().loadTestsFromModule(TestFeatures())
unittest.TextTestRunner().run(suite)

```

    ....
    ----------------------------------------------------------------------
    Ran 4 tests in 0.008s

    OK





    <unittest.runner.TextTestResult run=4 errors=0 failures=0>



<a id='part2_tutorial'></a>
## PART 2: Model Selection
### Model Selection Tutorial
The objective of Model Selection is to tune the number of states for each word HMM prior to testing on unseen data.  In this section you will explore three methods:
- Log likelihood using cross-validation folds (CV)
- Bayesian Information Criterion (BIC)
- Discriminative Information Criterion (DIC)

##### Train a single word
Now that we have built a training set with sequence data, we can "train" models for each word.  As a simple starting example, we train a single word using Gaussian hidden Markov models (HMM).   By using the `fit` method during training, the [Baum-Welch Expectation-Maximization](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm) (EM) algorithm is invoked iteratively to find the best estimate for the model *for the number of hidden states specified* from a group of sample seequences. For this example, we *assume* the correct number of hidden states is 3, but that is just a guess.  How do we know what the "best" number of states for training is?  We will need to find some model selection technique to choose the best parameter.


```python
import warnings
from hmmlearn.hmm import GaussianHMM

def train_a_word(word, num_hidden_states, features):

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    training = asl.build_training(features)  
    X, lengths = training.get_word_Xlengths(word)
    print("get_word_Xlengths(%s): lengths=", (word, lengths))
    model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
    logL = model.score(X, lengths)
    return model, logL

demoword = 'BOOK'
model, logL = train_a_word(demoword, 3, features_ground)
print("Number of states trained in model for {} is {}".format(demoword, model.n_components))
print("logL = {}".format(logL))
```

    get_word_Xlengths(%s): lengths= ('BOOK', [6, 6, 7, 7, 11, 8, 8, 8, 7, 8, 17, 15, 13, 14, 11, 8, 10, 8])
    Number of states trained in model for BOOK is 3
    logL = -2331.113812743319


The HMM model has been trained and information can be pulled from the model, including means and variances for each feature and hidden state.  The [log likelihood](http://math.stackexchange.com/questions/892832/why-we-consider-log-likelihood-instead-of-likelihood-in-gaussian-distribution) for any individual sample or group of samples can also be calculated with the `score` method.


```python
def show_model_stats(word, model):
    print("Number of states trained in model for {} is {}".format(word, model.n_components))    
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])    
    for i in range(model.n_components):  # for each hidden state
        print("hidden state #{}".format(i))
        print("mean = ", model.means_[i])
        print("variance = ", variance[i])
        print()

show_model_stats(demoword, model)
```

    Number of states trained in model for BOOK is 3
    hidden state #0
    mean =  [ -3.46504869  50.66686933  14.02391587  52.04731066]
    variance =  [ 49.12346305  43.04799144  39.35109609  47.24195772]

    hidden state #1
    mean =  [ -11.45300909   94.109178     19.03512475  102.2030162 ]
    variance =  [  77.403668    203.35441965   26.68898447  156.12444034]

    hidden state #2
    mean =  [ -1.12415027  69.44164191  17.02866283  77.7231196 ]
    variance =  [ 19.70434594  16.83041492  30.51552305  11.03678246]



##### Try it!
Experiment by changing the feature set, word, and/or num_hidden_states values in the next cell to see changes in values.  


```python
my_testword = 'CHOCOLATE'
model, logL = train_a_word(my_testword, 3, features_ground) # Experiment here with different parameters
show_model_stats(my_testword, model)
print("logL = {}".format(logL))
```

    get_word_Xlengths(%s): lengths= ('CHOCOLATE', [17, 20, 12])
    Number of states trained in model for CHOCOLATE is 3
    hidden state #0
    mean =  [   0.58333333   87.91666667   12.75        108.5       ]
    variance =  [  39.41055556   18.74388889    9.855       144.4175    ]

    hidden state #1
    mean =  [ -9.30211403  55.32333876   6.92259936  71.24057775]
    variance =  [ 16.16920957  46.50917372   3.81388185  15.79446427]

    hidden state #2
    mean =  [ -5.40587658  60.1652424    2.32479599  91.3095432 ]
    variance =  [   7.95073876   64.13103127   13.68077479  129.5912395 ]

    logL = -601.3291470028628


##### Visualize the hidden states
We can plot the means and variances for each state and feature.  Try varying the number of states trained for the HMM model and examine the variances.  Are there some models that are "better" than others?  How can you tell?  We would like to hear what you think in the classroom online.


```python
%matplotlib inline
```


```python
import math
from matplotlib import (cm, pyplot as plt, mlab)

def visualize(word, model):
    """ visualize the input model for a particular word """
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    figures = []
    for parm_idx in range(len(model.means_[0])):
        xmin = int(min(model.means_[:,parm_idx]) - max(variance[:,parm_idx]))
        xmax = int(max(model.means_[:,parm_idx]) + max(variance[:,parm_idx]))
        print(parm_idx, 'xmin=', xmin, ",xmax=",xmax)
        fig, axs = plt.subplots(model.n_components, sharex=True, sharey=False)
        colours = cm.rainbow(np.linspace(0, 1, model.n_components))
        for i, (ax, colour) in enumerate(zip(axs, colours)):
            x = np.linspace(xmin, xmax, 100)
            mu = model.means_[i,parm_idx]
            sigma = math.sqrt(np.diag(model.covars_[i])[parm_idx])
            ax.plot(x, mlab.normpdf(x, mu, sigma), c=colour)
            ax.set_title("{} feature {} hidden state #{}".format(word, parm_idx, i))

            ax.grid(True)
        figures.append(plt)
    for p in figures:
        p.show()

visualize(my_testword, model)
```

    0 xmin= -48 ,xmax= 39
    1 xmin= -8 ,xmax= 152
    2 xmin= -11 ,xmax= 26
    3 xmin= -73 ,xmax= 252



![png](output_49_1.png)



![png](output_49_2.png)



![png](output_49_3.png)



![png](output_49_4.png)


#####  ModelSelector class
Review the `ModelSelector` class from the codebase found in the `my_model_selectors.py` module.  It is designed to be a strategy pattern for choosing different model selectors.  For the project submission in this section, subclass `SelectorModel` to implement the following model selectors.  In other words, you will write your own classes/functions in the `my_model_selectors.py` module and run them from this notebook:

- `SelectorCV `:  Log likelihood with CV
- `SelectorBIC`: BIC
- `SelectorDIC`: DIC

You will train each word in the training set with a range of values for the number of hidden states, and then score these alternatives with the model selector, choosing the "best" according to each strategy. The simple case of training with a constant value for `n_components` can be called using the provided `SelectorConstant` subclass as follow:


```python
from my_model_selectors import SelectorConstant

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
word = 'VEGETABLE' # Experiment here with different words
model = SelectorConstant(training.get_all_sequences(), training.get_all_Xlengths(), word, n_constant=3, verbose=True ).select()
print("Number of states trained in model for {} is {}".format(word, model.n_components))
```

    model created for VEGETABLE with 3 states
    Number of states trained in model for VEGETABLE is 3


##### Cross-validation folds
If we simply score the model with the Log Likelihood calculated from the feature sequences it has been trained on, we should expect that more complex models will have higher likelihoods. However, that doesn't tell us which would have a better likelihood score on unseen data.  The model will likely be overfit as complexity is added.  To estimate which topology model is better using only the training data, we can compare scores using cross-validation.  One technique for cross-validation is to break the training set into "folds" and rotate which fold is left out of training.  The "left out" fold scored.  This gives us a proxy method of finding the best model to use on "unseen data". In the following example, a set of word sequences is broken into three folds using the [scikit-learn Kfold](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) class object. When you implement `SelectorCV`, you will use this technique.


```python
from sklearn.model_selection import KFold

training = asl.build_training(features_ground) # Experiment here with different feature sets
word = 'VEGETABLE' # Experiment here with different words
word_sequences = training.get_word_sequences(word)
split_method = KFold()
for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
    print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  # view indices of the folds
```

    Train fold indices:[2 3 4 5] Test fold indices:[0 1]
    Train fold indices:[0 1 4 5] Test fold indices:[2 3]
    Train fold indices:[0 1 2 3] Test fold indices:[4 5]


**Tip:** In order to run `hmmlearn` training using the X,lengths tuples on the new folds, subsets must be combined based on the indices given for the folds.  A helper utility has been provided in the `asl_utils` module named `combine_sequences` for this purpose.

##### Scoring models with other criterion
Scoring model topologies with **BIC** balances fit and complexity within the training set for each word.  In the BIC equation, a penalty term penalizes complexity to avoid overfitting, so that it is not necessary to also use cross-validation in the selection process.  There are a number of references on the internet for this criterion.  These [slides](http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf) include a formula you may find helpful for your implementation.

The advantages of scoring model topologies with **DIC** over BIC are presented by Alain Biem in this [reference](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf) (also found [here](https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf)).  DIC scores the discriminant ability of a training set for one word against competing words.  Instead of a penalty term for complexity, it provides a penalty if model liklihoods for non-matching words are too similar to model likelihoods for the correct word in the word set.

<a id='part2_submission'></a>
### Model Selection Implementation Submission
Implement `SelectorCV`, `SelectorBIC`, and `SelectorDIC` classes in the `my_model_selectors.py` module.  Run the selectors on the following five words. Then answer the questions about your results.

**Tip:** The `hmmlearn` library may not be able to train or score all models.  Implement try/except contructs as necessary to eliminate non-viable models from consideration.


```python
words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
import timeit
```


```python
training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()

X, lengths = training.get_word_Xlengths('FISH')

```


```python
X, l = training.get_word_Xlengths('JOHN')
#X, l = training.get_word_Xlengths('FISH')
len(l)
```




    113




```python
# TODO: Implement SelectorCV in my_model_selector.py
from my_model_selectors import SelectorCV

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorCV(sequences, Xlengths, word,
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))
```


```python
# TODO: Implement SelectorBIC in module my_model_selectors.py
from my_model_selectors import SelectorBIC

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorBIC(sequences, Xlengths, word,
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))
```


```python
# TODO: Implement SelectorBIC in module my_model_selectors.py
from my_model_selectors import SelectorBIC_orig

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorBIC(sequences, Xlengths, word,
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))
```


```python
# TODO: Implement SelectorDIC in module my_model_selectors.py
from my_model_selectors import SelectorDIC

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorDIC(sequences, Xlengths, word,
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))
```

**Question 2:**  Compare and contrast the possible advantages and disadvantages of the various model selectors implemented.

**Answer 2:**
* <b>SelectorBIC</b> works the best for my project. For all features, it has lower WER. The dusadvantage of BIC is the hacking-like parameter selecting, what make selector tuning difficult. I've tried  several formula that I found online but with no explanation about rationality and twisted a bit. The outcoming could be > 10% WER difference. The selected magic parameter runs the best and beat the other selectors.

* <b>SelectorDIC</b> works well. The algorithm of selection has rationality - it chooses the model that identify the current the word much better than other words. The performance is as good as SelectorBIC in my project.

* <b>SelectorCV</b> runs very slow for words with large number of sequence. Word 'JOHN' has 113 sequences and it took the selector 395 seconds to build the model while the others selectors  complete within 12 second. The SelectCV does not work with words with 1 or 2 sequences. I think the algorithm of SelectCV is the most easy to understand one.
<br>

I modified SelectorBIC as the reviewer suggested and rename the original SelectorBIC to SelectorBIC_orig.
<br>
* while the SelectorBIC improved features_grnd and custom feature features_grnd_distance, the SelectorBIC_orig performs better with features_polar and features_podel.

The reason I retain the original SelectorBIC_orig because it still recognizes the most word with features_podel.

<a id='part2_test'></a>
### Model Selector Unit Testing
Run the following unit tests as a sanity check on the implemented model selectors.  The test simply looks for valid interfaces  but is not exhaustive. However, the project should not be submitted if these tests don't pass.


```python
from asl_test_model_selectors import TestSelectors
suite = unittest.TestLoader().loadTestsFromModule(TestSelectors())
unittest.TextTestRunner().run(suite)
```

<a id='part3_tutorial'></a>
## PART 3: Recognizer
The objective of this section is to "put it all together".  Using the four feature sets created and the three model selectors, you will experiment with the models and present your results.  Instead of training only five specific words as in the previous section, train the entire set with a feature set and model selector strategy.  
### Recognizer Tutorial
##### Train the full training set
The following example trains the entire set with the example `features_ground` and `SelectorConstant` features and model selector.  Use this pattern for you experimentation and final submission cells.




```python
# autoreload for automatically reloading changes made in my_model_selectors and my_recognizer
%load_ext autoreload
%autoreload 2

from my_model_selectors import SelectorConstant

def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict

models = train_all_words(features_ground, SelectorConstant)
print("Number of word models returned = {}".format(len(models)))
```

    Number of word models returned = 112


##### Load the test set
The `build_test` method in `ASLdb` is similar to the `build_training` method already presented, but there are a few differences:
- the object is type `SinglesData`
- the internal dictionary keys are the index of the test word rather than the word itself
- the getter methods are `get_all_sequences`, `get_all_Xlengths`, `get_item_sequences` and `get_item_Xlengths`


```python
test_set = asl.build_test(features_ground)
print("Number of test set items: {}".format(test_set.num_items))
print("Number of test set sentences: {}".format(len(test_set.sentences_index)))
```

    Number of test set items: 178
    Number of test set sentences: 40


<a id='part3_submission'></a>
### Recognizer Implementation Submission
For the final project submission, students must implement a recognizer following guidance in the `my_recognizer.py` module.  Experiment with the four feature sets and the three model selection methods (that's 12 possible combinations). You can add and remove cells for experimentation or run the recognizers locally in some other way during your experiments, but retain the results for your discussion.  For submission, you will provide code cells of **only three** interesting combinations for your discussion (see questions below). At least one of these should produce a word error rate of less than 60%, i.e. WER < 0.60 .

**Tip:** The hmmlearn library may not be able to train or score all models.  Implement try/except contructs as necessary to eliminate non-viable models from consideration.


```python
from my_recognizer import recognize
from asl_utils import show_errors
```


```python
# TODO Choose a feature set and model selector
features = features_ground # change as needed
model_selector = SelectorConstant # change as needed

# TODO Recognize the test set and display the result with the show_errors method
models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
show_errors(guesses, test_set)
```


    **** WER = 0.6741573033707865
    Total correct: 58 out of 178
    Video  Recognized                                                    Correct
    =====================================================================================================
      100: POSS NEW CAR BREAK-DOWN                                       POSS NEW CAR BREAK-DOWN
        2: *GO *BOOK *ARRIVE                                             JOHN WRITE HOMEWORK
       67: *LIKE FUTURE NOT BUY HOUSE                                    JOHN FUTURE NOT BUY HOUSE
        7: *SOMETHING-ONE *GO1 *IX CAN                                   JOHN CAN GO CAN
      201: JOHN *GIVE *GIVE *LOVE *ARRIVE HOUSE                          JOHN TELL MARY IX-1P BUY HOUSE
       74: *IX *VISIT *GO *GO                                            JOHN NOT VISIT MARY
      119: *PREFER *BUY1 IX *BLAME *IX                                   SUE BUY IX CAR BLUE
       12: JOHN *HAVE *WHAT CAN                                          JOHN CAN GO CAN
       77: *JOHN BLAME *LOVE                                             ANN BLAME MARY
      142: *FRANK *STUDENT YESTERDAY *TEACHER BOOK                       JOHN BUY YESTERDAY WHAT BOOK
      107: *SHOULD *IX FRIEND *GO *JANA                                  JOHN POSS FRIEND HAVE CANDY
       84: *LOVE *ARRIVE *HOMEWORK BOOK                                  IX-1P FIND SOMETHING-ONE BOOK
       21: JOHN *HOMEWORK *NEW *PREFER *CAR *CAR *FUTURE *EAT            JOHN FISH WONT EAT BUT CAN EAT CHICKEN
       25: *FRANK *TELL *LOVE *TELL *LOVE                                JOHN LIKE IX IX IX
       89: *GIVE *GIVE GIVE *IX IX *ARRIVE *BOOK                         JOHN IX GIVE MAN IX NEW COAT
       71: JOHN *FINISH VISIT MARY                                       JOHN WILL VISIT MARY
       92: *FRANK GIVE *WOMAN *WOMAN WOMAN BOOK                          JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
       90: *SOMETHING-ONE *SOMETHING-ONE IX *IX WOMAN *COAT              JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
       30: *SHOULD LIKE *GO *GO *GO                                      JOHN LIKE IX IX IX
      193: JOHN *SOMETHING-ONE *YESTERDAY BOX                            JOHN GIVE GIRL BOX
       36: *VISIT VEGETABLE *YESTERDAY *GIVE *MARY *MARY                 MARY VEGETABLE KNOW IX LIKE CORN1
      139: *SHOULD *BUY1 *CAR *BLAME BOOK                                JOHN BUY WHAT YESTERDAY BOOK
      167: *MARY IX *VISIT *WOMAN *LOVE                                  JOHN IX SAY LOVE MARY
       40: *SUE *GIVE *CORN *VEGETABLE *GO                               JOHN IX THINK MARY LOVE
       28: *FRANK *TELL *LOVE *TELL *LOVE                                JOHN LIKE IX IX IX
      171: *VISIT *VISIT BLAME                                           JOHN MARY BLAME
       43: *FRANK *GO BUY HOUSE                                          JOHN MUST BUY HOUSE
      108: *GIVE *LOVE                                                   WOMAN ARRIVE
      174: *CAN *GIVE3 GIVE1 *APPLE *WHAT                                PEOPLE GROUP GIVE1 JANA TOY
      113: IX CAR *CAR *IX *IX                                           IX CAR BLUE SUE BUY
       50: *FRANK *SEE BUY CAR *SOMETHING-ONE                            FUTURE JOHN BUY CAR SHOULD
      199: *LOVE CHOCOLATE WHO                                           LIKE CHOCOLATE WHO
      158: LOVE *MARY WHO                                                LOVE JOHN WHO
       54: JOHN SHOULD *WHO BUY HOUSE                                    JOHN SHOULD NOT BUY HOUSE
      105: *FRANK *VEGETABLE                                             JOHN LEG
      184: *GIVE1 BOY *GIVE1 TEACHER APPLE                               ALL BOY GIVE TEACHER APPLE
       57: *MARY *VISIT VISIT *VISIT                                     JOHN DECIDE VISIT MARY
      122: JOHN *GIVE1 *COAT                                             JOHN READ BOOK
      189: *JANA *SOMETHING-ONE *YESTERDAY *WHAT                         JOHN GIVE GIRL BOX
      181: *BLAME ARRIVE                                                 JOHN ARRIVE



```python
from my_model_selectors import SelectorDIC, SelectorBIC_orig, SelectorCV, SelectorBIC
np.seterr(invalid='ignore')

# customized featurs
myfeatures_names = ["features_grnd_distance", "features_norm_poldel", "features_podel", "features_norm_pol"]
myfeatures_list = [features_grnd_distance, features_norm_poldel, features_podel, features_norm_pol]   # change as needed
# selected features
features_names = ["features_grnd_distance", 'features_podel', 'features_ground', 'features_polar', 'features_norm', 'features_delta']
features_list = [features_grnd_distance, features_podel, features_ground, features_polar, features_norm, features_delta]   
# after review: features_custom2 from reviewer, features_podel2 from features_podel
newfeature_names = ['features_podel', "features_custom2", "scale_podel"]
newfeature_list = [features_podel, features_custom2, scale_podel]

# model_selectors
model_selector_names = ['SelectorBIC', 'SelectorBIC_orig', 'SelectorDIC', 'SelectorCV']
model_selector_list = [SelectorBIC, SelectorBIC_orig, SelectorDIC, SelectorCV]

sel_features_list = newfeature_list           # features_list[:3]
sel_model_list = model_selector_list[:2]
# pre-build test_set
test_set_list = [asl.build_test(features) for j, features in enumerate(sel_features_list)]

for i, model_selector in enumerate(sel_model_list):
    for j, features in enumerate(sel_features_list):
        print("model_selector=%s, features=%s" % (model_selector_names[i], newfeature_names[j]))
        models = train_all_words(features, model_selector_list[i])
        test_set = test_set_list[j]
        probabilities, guesses = recognize(models, test_set)
        show_errors(guesses, test_set)
```

    model_selector=SelectorBIC, features=features_podel

    **** WER = 0.5674157303370787
    Total correct: 77 out of 178
    Video  Recognized                                                    Correct
    =====================================================================================================


    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:251: RuntimeWarning: invalid value encountered in true_divide
      / (means_weight + denom))
    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:265: RuntimeWarning: invalid value encountered in maximum
      (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:251: RuntimeWarning: invalid value encountered in true_divide
      / (means_weight + denom))
    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:265: RuntimeWarning: invalid value encountered in maximum
      (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:251: RuntimeWarning: invalid value encountered in true_divide
      / (means_weight + denom))
    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:265: RuntimeWarning: invalid value encountered in maximum
      (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:251: RuntimeWarning: invalid value encountered in true_divide
      / (means_weight + denom))
    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:265: RuntimeWarning: invalid value encountered in maximum
      (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)


      100: POSS NEW CAR BREAK-DOWN                                       POSS NEW CAR BREAK-DOWN
        2: *GO *COAT HOMEWORK                                            JOHN WRITE HOMEWORK
       67: *SHOULD FUTURE *MARY BUY HOUSE                                JOHN FUTURE NOT BUY HOUSE
        7: JOHN CAN *IX CAN                                              JOHN CAN GO CAN
      201: JOHN *MAN *MAN *LIKE BUY HOUSE                                JOHN TELL MARY IX-1P BUY HOUSE
       74: *IX *VISIT *GIVE *GO                                          JOHN NOT VISIT MARY
      119: *PREFER *BUY1 *HAVE *BOX *IX                                  SUE BUY IX CAR BLUE
       12: JOHN *WHAT *GO1 *HOUSE                                        JOHN CAN GO CAN
       77: *JOHN BLAME *LOVE                                             ANN BLAME MARY
      142: JOHN BUY YESTERDAY WHAT BOOK                                  JOHN BUY YESTERDAY WHAT BOOK
      107: *LIKE POSS *HAVE *GO *SAY                                     JOHN POSS FRIEND HAVE CANDY
       84: *LOVE *NEW *HOMEWORK BOOK                                     IX-1P FIND SOMETHING-ONE BOOK
       21: JOHN *NEW WONT *WHO *CAR *TEACHER *MOTHER *TOMORROW           JOHN FISH WONT EAT BUT CAN EAT CHICKEN
       25: JOHN LIKE *LOVE *MARY IX                                      JOHN LIKE IX IX IX
       89: *GIVE *SOMETHING-ONE *WOMAN *IX IX NEW *BREAK-DOWN            JOHN IX GIVE MAN IX NEW COAT
       71: JOHN *FUTURE *GIVE1 MARY                                      JOHN WILL VISIT MARY
       92: JOHN *WOMAN *WOMAN *WOMAN WOMAN BOOK                          JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
       90: *SOMETHING-ONE *SOMETHING-ONE IX *ALL WOMAN BOOK              JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
       30: JOHN LIKE *MARY *MARY IX                                      JOHN LIKE IX IX IX
      193: JOHN *POSS *GIVE BOX                                          JOHN GIVE GIRL BOX
       36: *VISIT VEGETABLE *YESTERDAY *GIVE *MARY *MARY                 MARY VEGETABLE KNOW IX LIKE CORN1
      139: JOHN *BUY1 WHAT *BLAME BOOK                                   JOHN BUY WHAT YESTERDAY BOOK
      167: *MARY IX *VISIT LOVE MARY                                     JOHN IX SAY LOVE MARY
       40: *BILL *VISIT *FUTURE1 MARY *MARY                              JOHN IX THINK MARY LOVE
       28: JOHN *MARY *FUTURE *MARY IX                                   JOHN LIKE IX IX IX
      171: *MARY MARY BLAME                                              JOHN MARY BLAME
       43: JOHN *JOHN BUY HOUSE                                          JOHN MUST BUY HOUSE
      108: *SOMETHING-ONE *BOOK                                          WOMAN ARRIVE
      174: *JOHN *GIVE3 GIVE1 *APPLE *WHAT                               PEOPLE GROUP GIVE1 JANA TOY
      113: *BLUE CAR *IX *IX *BUY1                                       IX CAR BLUE SUE BUY
       50: *SOMETHING-ONE *SEE BUY CAR *ARRIVE                           FUTURE JOHN BUY CAR SHOULD
      199: LIKE *HOMEWORK *TELL                                          LIKE CHOCOLATE WHO
      158: LOVE JOHN *VEGETABLE                                          LOVE JOHN WHO
       54: JOHN SHOULD *WHO BUY HOUSE                                    JOHN SHOULD NOT BUY HOUSE
      105: JOHN *VEGETABLE                                               JOHN LEG
      184: *IX BOY *GIVE1 TEACHER *GIRL                                  ALL BOY GIVE TEACHER APPLE
       57: *MARY *PREFER VISIT *IX                                       JOHN DECIDE VISIT MARY
      122: JOHN *GIVE1 BOOK                                              JOHN READ BOOK
      189: *MARY *GO *PREFER BOX                                         JOHN GIVE GIRL BOX
      181: *EAT ARRIVE                                                   JOHN ARRIVE
    model_selector=SelectorBIC, features=features_custom2

    **** WER = 0.42134831460674155
    Total correct: 103 out of 178
    Video  Recognized                                                    Correct
    =====================================================================================================
      100: POSS NEW CAR BREAK-DOWN                                       POSS NEW CAR BREAK-DOWN
        2: JOHN WRITE HOMEWORK                                           JOHN WRITE HOMEWORK


    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:251: RuntimeWarning: invalid value encountered in true_divide
      / (means_weight + denom))
    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:251: RuntimeWarning: invalid value encountered in true_divide
      / (means_weight + denom))
    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:265: RuntimeWarning: invalid value encountered in maximum
      (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)


       67: JOHN FUTURE NOT BUY HOUSE                                     JOHN FUTURE NOT BUY HOUSE
        7: JOHN *PEOPLE *IX *TOY                                         JOHN CAN GO CAN
      201: JOHN *SHOULD *WOMAN *LOVE BUY HOUSE                           JOHN TELL MARY IX-1P BUY HOUSE
       74: *IX *MARY *MARY MARY                                          JOHN NOT VISIT MARY
      119: *WHO *BUY1 IX *TOY *JANA                                      SUE BUY IX CAR BLUE
       12: JOHN CAN *GO1 CAN                                             JOHN CAN GO CAN
       77: *JOHN BLAME MARY                                              ANN BLAME MARY
      142: JOHN BUY YESTERDAY WHAT BOOK                                  JOHN BUY YESTERDAY WHAT BOOK
      107: JOHN POSS FRIEND *MARY *JOHN                                  JOHN POSS FRIEND HAVE CANDY
       84: *ANN *LIVE *HOMEWORK BOOK                                     IX-1P FIND SOMETHING-ONE BOOK
       21: JOHN FISH WONT *MARY BUT *CAR *CHICKEN *EAT                   JOHN FISH WONT EAT BUT CAN EAT CHICKEN
       25: JOHN LIKE *MARY *CORN1 IX                                     JOHN LIKE IX IX IX
       89: *THINK *SOMETHING-ONE *SOMETHING-ONE *SOMETHING-ONE *SOMETHING-ONE *NEW-YORK COAT  JOHN IX GIVE MAN IX NEW COAT
       71: JOHN WILL VISIT MARY                                          JOHN WILL VISIT MARY
       92: JOHN *SOMETHING-ONE IX *IX WOMAN BOOK                         JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
       90: JOHN *IX IX *IX *MARY BOOK                                    JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
       30: JOHN *MARY *MARY IX IX                                        JOHN LIKE IX IX IX
      193: JOHN *SELF *CORN BOX                                          JOHN GIVE GIRL BOX
       36: MARY *JOHN *IX *WOMAN *MARY *MARY                             MARY VEGETABLE KNOW IX LIKE CORN1
      139: JOHN *BUY1 WHAT *WHAT BOOK                                    JOHN BUY WHAT YESTERDAY BOOK
      167: JOHN IX *MARY LOVE MARY                                       JOHN IX SAY LOVE MARY
       40: JOHN *MARY *JOHN MARY *MARY                                   JOHN IX THINK MARY LOVE
       28: JOHN *WHO IX IX IX                                            JOHN LIKE IX IX IX
      171: JOHN *JOHN BLAME                                              JOHN MARY BLAME
       43: JOHN *FUTURE BUY HOUSE                                        JOHN MUST BUY HOUSE
      108: *LOVE *VIDEOTAPE                                              WOMAN ARRIVE
      174: *GROUP GROUP GIVE1 *WHO TOY                                   PEOPLE GROUP GIVE1 JANA TOY
      113: IX CAR BLUE *JOHN *BUY1                                       IX CAR BLUE SUE BUY
       50: *JOHN *SEE *WRITE CAR SHOULD                                  FUTURE JOHN BUY CAR SHOULD
      199: *JOHN CHOCOLATE WHO                                           LIKE CHOCOLATE WHO
      158: LOVE JOHN WHO                                                 LOVE JOHN WHO
       54: JOHN SHOULD *FUTURE BUY HOUSE                                 JOHN SHOULD NOT BUY HOUSE
      105: JOHN *SEE                                                     JOHN LEG
      184: *IX BOY *GIVE1 TEACHER APPLE                                  ALL BOY GIVE TEACHER APPLE
       57: *IX *VEGETABLE VISIT MARY                                     JOHN DECIDE VISIT MARY
      122: JOHN READ BOOK                                                JOHN READ BOOK
      189: JOHN *JOHN *CORN *BUY1                                        JOHN GIVE GIRL BOX
      181: JOHN *BOX                                                     JOHN ARRIVE
    model_selector=SelectorBIC, features=scale_podel

    **** WER = 0.46629213483146065
    Total correct: 95 out of 178
    Video  Recognized                                                    Correct
    =====================================================================================================
      100: POSS NEW CAR BREAK-DOWN                                       POSS NEW CAR BREAK-DOWN
        2: JOHN WRITE HOMEWORK                                           JOHN WRITE HOMEWORK
       67: JOHN *YESTERDAY NOT BUY HOUSE                                 JOHN FUTURE NOT BUY HOUSE
        7: JOHN *HAVE *IX *ARRIVE                                        JOHN CAN GO CAN
      201: JOHN *MARY *WOMAN *JOHN BUY HOUSE                             JOHN TELL MARY IX-1P BUY HOUSE
       74: *IX *WHO *MARY MARY                                           JOHN NOT VISIT MARY
      119: *JOHN *BUY1 *BLUE *TOY *JANA                                  SUE BUY IX CAR BLUE
       12: JOHN CAN *GO1 CAN                                             JOHN CAN GO CAN
       77: *JOHN BLAME MARY                                              ANN BLAME MARY
      142: JOHN BUY YESTERDAY WHAT BOOK                                  JOHN BUY YESTERDAY WHAT BOOK
      107: JOHN *IX FRIEND *MARY *WHO                                    JOHN POSS FRIEND HAVE CANDY
       84: *MARY *NEW *HOMEWORK BOOK                                     IX-1P FIND SOMETHING-ONE BOOK
       21: JOHN *HOMEWORK WONT *WHO BUT *CAR *CHICKEN CHICKEN            JOHN FISH WONT EAT BUT CAN EAT CHICKEN
       25: JOHN LIKE *LOVE *WHO IX                                       JOHN LIKE IX IX IX
       89: *SAY *GIVE *WOMAN *OLD IX *BUY *BOOK                          JOHN IX GIVE MAN IX NEW COAT
       71: JOHN *FUTURE VISIT MARY                                       JOHN WILL VISIT MARY
       92: JOHN *WOMAN IX *WOMAN WOMAN BOOK                              JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
       90: JOHN *GIVE1 IX *IX WOMAN BOOK                                 JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
       30: JOHN *MARY *MARY *MARY *MARY                                  JOHN LIKE IX IX IX
      193: JOHN *SOMETHING-ONE *GIVE1 BOX                                JOHN GIVE GIRL BOX
       36: MARY *WHO *GIRL *GIVE *MARY *MARY                             MARY VEGETABLE KNOW IX LIKE CORN1
      139: JOHN *BUY1 WHAT YESTERDAY BOOK                                JOHN BUY WHAT YESTERDAY BOOK
      167: JOHN *TOY1 *MARY LOVE MARY                                    JOHN IX SAY LOVE MARY
       40: JOHN *BILL *CORN *JOHN *MARY                                  JOHN IX THINK MARY LOVE
       28: JOHN *WHO *MARY *MARY IX                                      JOHN LIKE IX IX IX
      171: JOHN MARY BLAME                                               JOHN MARY BLAME
       43: JOHN *POSS BUY HOUSE                                          JOHN MUST BUY HOUSE
      108: *MAN *BOOK                                                    WOMAN ARRIVE
      174: PEOPLE GROUP GIVE1 *CORN TOY                                  PEOPLE GROUP GIVE1 JANA TOY
      113: IX CAR BLUE *MARY *IX-1P                                      IX CAR BLUE SUE BUY
       50: *JOHN *SEE BUY CAR *JOHN                                      FUTURE JOHN BUY CAR SHOULD
      199: *JOHN CHOCOLATE WHO                                           LIKE CHOCOLATE WHO
      158: LOVE JOHN WHO                                                 LOVE JOHN WHO
       54: JOHN *JOHN *MARY BUY HOUSE                                    JOHN SHOULD NOT BUY HOUSE
      105: JOHN *VEGETABLE                                               JOHN LEG
      184: *GIVE3 BOY *GIVE1 TEACHER *GIRL                               ALL BOY GIVE TEACHER APPLE
       57: JOHN *PREFER VISIT MARY                                       JOHN DECIDE VISIT MARY
      122: JOHN *GIVE1 BOOK                                              JOHN READ BOOK
      189: JOHN *SELF *CORN *BUY1                                        JOHN GIVE GIRL BOX
      181: *SUE ARRIVE                                                   JOHN ARRIVE
    model_selector=SelectorBIC_orig, features=features_podel

    **** WER = 0.5
    Total correct: 89 out of 178
    Video  Recognized                                                    Correct
    =====================================================================================================


    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:251: RuntimeWarning: invalid value encountered in true_divide
      / (means_weight + denom))
    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:265: RuntimeWarning: invalid value encountered in maximum
      (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:251: RuntimeWarning: invalid value encountered in true_divide
      / (means_weight + denom))
    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:265: RuntimeWarning: invalid value encountered in maximum
      (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:251: RuntimeWarning: invalid value encountered in true_divide
      / (means_weight + denom))
    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:265: RuntimeWarning: invalid value encountered in maximum
      (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:251: RuntimeWarning: invalid value encountered in true_divide
      / (means_weight + denom))
    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:265: RuntimeWarning: invalid value encountered in maximum
      (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)


      100: POSS NEW CAR BREAK-DOWN                                       POSS NEW CAR BREAK-DOWN
        2: *GO *BOOK HOMEWORK                                            JOHN WRITE HOMEWORK
       67: *SHOULD FUTURE *MARY BUY HOUSE                                JOHN FUTURE NOT BUY HOUSE
        7: JOHN CAN *JOHN CAN                                            JOHN CAN GO CAN
      201: JOHN *MAN *WOMAN *LIKE BUY HOUSE                              JOHN TELL MARY IX-1P BUY HOUSE
       74: *IX *GO VISIT *GO                                             JOHN NOT VISIT MARY
      119: *MARY *BUY1 IX *JOHN *IX                                      SUE BUY IX CAR BLUE
       12: JOHN *WHAT *GO1 *HOUSE                                        JOHN CAN GO CAN
       77: *JOHN BLAME *LOVE                                             ANN BLAME MARY
      142: JOHN BUY YESTERDAY WHAT BOOK                                  JOHN BUY YESTERDAY WHAT BOOK
      107: *LIKE POSS *HAVE *GO *GO                                      JOHN POSS FRIEND HAVE CANDY
       84: *LOVE *GIVE1 *GIVE1 BOOK                                      IX-1P FIND SOMETHING-ONE BOOK
       21: JOHN *ARRIVE *FUTURE *WHO *CAR *TEACHER *MOTHER *FUTURE       JOHN FISH WONT EAT BUT CAN EAT CHICKEN
       25: JOHN LIKE *LOVE *MARY IX                                      JOHN LIKE IX IX IX
       89: JOHN *GIVE *IX *IX IX NEW *BOOK                               JOHN IX GIVE MAN IX NEW COAT
       71: JOHN *FUTURE *GIVE1 MARY                                      JOHN WILL VISIT MARY
       92: JOHN GIVE IX *IX WOMAN BOOK                                   JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
       90: JOHN *HAVE IX SOMETHING-ONE WOMAN BOOK                        JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
       30: JOHN LIKE *MARY *MARY IX                                      JOHN LIKE IX IX IX
      193: JOHN *HAVE *YESTERDAY BOX                                     JOHN GIVE GIRL BOX
       36: MARY VEGETABLE *YESTERDAY *GIVE *MARY *MARY                   MARY VEGETABLE KNOW IX LIKE CORN1
      139: JOHN *BUY1 WHAT *GIVE1 BOOK                                   JOHN BUY WHAT YESTERDAY BOOK
      167: *MARY IX *VISIT LOVE MARY                                     JOHN IX SAY LOVE MARY
       40: *BILL *GO *FUTURE1 MARY *MARY                                 JOHN IX THINK MARY LOVE
       28: JOHN *MARY *FUTURE *MARY IX                                   JOHN LIKE IX IX IX
      171: *MARY MARY BLAME                                              JOHN MARY BLAME
       43: JOHN *JOHN BUY HOUSE                                          JOHN MUST BUY HOUSE
      108: WOMAN *BOOK                                                   WOMAN ARRIVE
      174: *JOHN *GIVE3 GIVE1 *GIRL *BLAME                               PEOPLE GROUP GIVE1 JANA TOY
      113: IX CAR BLUE *IX *BUY1                                         IX CAR BLUE SUE BUY
       50: *POSS *SEE BUY CAR *ARRIVE                                    FUTURE JOHN BUY CAR SHOULD
      199: LIKE CHOCOLATE *TELL                                          LIKE CHOCOLATE WHO
      158: LOVE JOHN *NOT                                                LOVE JOHN WHO
       54: JOHN SHOULD *WHO BUY HOUSE                                    JOHN SHOULD NOT BUY HOUSE
      105: JOHN *VEGETABLE                                               JOHN LEG
      184: *IX BOY *GIVE1 TEACHER APPLE                                  ALL BOY GIVE TEACHER APPLE
       57: *MARY *VISIT VISIT *IX                                        JOHN DECIDE VISIT MARY
      122: JOHN *GIVE1 BOOK                                              JOHN READ BOOK
      189: *MARY *VISIT *VISIT BOX                                       JOHN GIVE GIRL BOX
      181: *APPLE ARRIVE                                                 JOHN ARRIVE
    model_selector=SelectorBIC_orig, features=features_custom2

    **** WER = 0.4044943820224719
    Total correct: 106 out of 178
    Video  Recognized                                                    Correct
    =====================================================================================================
      100: POSS NEW CAR BREAK-DOWN                                       POSS NEW CAR BREAK-DOWN
        2: JOHN WRITE HOMEWORK                                           JOHN WRITE HOMEWORK

    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:251: RuntimeWarning: invalid value encountered in true_divide
      / (means_weight + denom))
    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:251: RuntimeWarning: invalid value encountered in true_divide
      / (means_weight + denom))
    C:\Users\linna\AppData\Roaming\Python\Python35\site-packages\hmmlearn\hmm.py:265: RuntimeWarning: invalid value encountered in maximum
      (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)



       67: JOHN FUTURE NOT BUY HOUSE                                     JOHN FUTURE NOT BUY HOUSE
        7: JOHN *PEOPLE GO *TOY                                          JOHN CAN GO CAN
      201: JOHN *SHOULD *WOMAN *LIKE BUY HOUSE                           JOHN TELL MARY IX-1P BUY HOUSE
       74: *IX *MARY *MARY MARY                                          JOHN NOT VISIT MARY
      119: *WHO *BUY1 IX *TOY *JANA                                      SUE BUY IX CAR BLUE
       12: JOHN CAN *GO1 CAN                                             JOHN CAN GO CAN
       77: *JOHN BLAME MARY                                              ANN BLAME MARY
      142: JOHN BUY YESTERDAY WHAT BOOK                                  JOHN BUY YESTERDAY WHAT BOOK
      107: JOHN POSS FRIEND *MARY *JOHN                                  JOHN POSS FRIEND HAVE CANDY
       84: *ANN *STUDENT *HOMEWORK BOOK                                  IX-1P FIND SOMETHING-ONE BOOK
       21: JOHN FISH WONT *MARY BUT *CAR *CHICKEN *MARY                  JOHN FISH WONT EAT BUT CAN EAT CHICKEN
       25: JOHN LIKE *MARY *LIKE IX                                      JOHN LIKE IX IX IX
       89: *THINK *SHOULD *SAY *WOMAN IX *NEW-YORK COAT                  JOHN IX GIVE MAN IX NEW COAT
       71: JOHN WILL VISIT MARY                                          JOHN WILL VISIT MARY
       92: JOHN *GIVE1 IX SOMETHING-ONE WOMAN BOOK                       JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
       90: JOHN *IX IX *IX *MARY BOOK                                    JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
       30: JOHN *MARY *MARY IX IX                                        JOHN LIKE IX IX IX
      193: JOHN *SELF *CORN BOX                                          JOHN GIVE GIRL BOX
       36: MARY *JOHN *IX *WOMAN *MARY *MARY                             MARY VEGETABLE KNOW IX LIKE CORN1
      139: JOHN *BUY1 WHAT YESTERDAY BOOK                                JOHN BUY WHAT YESTERDAY BOOK
      167: JOHN IX *MARY LOVE MARY                                       JOHN IX SAY LOVE MARY
       40: JOHN *MARY *JOHN MARY *MARY                                   JOHN IX THINK MARY LOVE
       28: JOHN LIKE IX *LIKE IX                                         JOHN LIKE IX IX IX
      171: JOHN *JOHN BLAME                                              JOHN MARY BLAME
       43: JOHN *SHOULD BUY HOUSE                                        JOHN MUST BUY HOUSE
      108: *LOVE *VIDEOTAPE                                              WOMAN ARRIVE
      174: *GROUP GROUP GIVE1 *WHO TOY                                   PEOPLE GROUP GIVE1 JANA TOY
      113: IX CAR BLUE *JOHN *BUY1                                       IX CAR BLUE SUE BUY
       50: *JOHN *SEE BUY *WHAT SHOULD                                   FUTURE JOHN BUY CAR SHOULD
      199: *JOHN CHOCOLATE *MARY                                         LIKE CHOCOLATE WHO
      158: LOVE JOHN WHO                                                 LOVE JOHN WHO
       54: JOHN SHOULD *FUTURE BUY HOUSE                                 JOHN SHOULD NOT BUY HOUSE
      105: JOHN *SEE                                                     JOHN LEG
      184: *IX BOY *GIVE1 TEACHER APPLE                                  ALL BOY GIVE TEACHER APPLE
       57: *IX *VEGETABLE VISIT MARY                                     JOHN DECIDE VISIT MARY
      122: JOHN READ BOOK                                                JOHN READ BOOK
      189: JOHN *JOHN *CORN *BUY1                                        JOHN GIVE GIRL BOX
      181: JOHN *BOX                                                     JOHN ARRIVE
    model_selector=SelectorBIC_orig, features=scale_podel

    **** WER = 0.43258426966292135
    Total correct: 101 out of 178
    Video  Recognized                                                    Correct
    =====================================================================================================
      100: POSS NEW CAR BREAK-DOWN                                       POSS NEW CAR BREAK-DOWN
        2: JOHN WRITE HOMEWORK                                           JOHN WRITE HOMEWORK
       67: JOHN *YESTERDAY NOT BUY HOUSE                                 JOHN FUTURE NOT BUY HOUSE
        7: JOHN *HAVE GO *ARRIVE                                         JOHN CAN GO CAN
      201: JOHN *MARY *WOMAN *JOHN BUY HOUSE                             JOHN TELL MARY IX-1P BUY HOUSE
       74: *IX *MARY *MARY MARY                                          JOHN NOT VISIT MARY
      119: *JOHN *BUY1 *BLUE *TOY *JANA                                  SUE BUY IX CAR BLUE
       12: JOHN *WHAT *GO1 CAN                                           JOHN CAN GO CAN
       77: *JOHN BLAME MARY                                              ANN BLAME MARY
      142: JOHN BUY YESTERDAY WHAT BOOK                                  JOHN BUY YESTERDAY WHAT BOOK
      107: JOHN POSS FRIEND *MARY *MARY                                  JOHN POSS FRIEND HAVE CANDY
       84: *MARY *NEW *HOMEWORK BOOK                                     IX-1P FIND SOMETHING-ONE BOOK
       21: JOHN FISH WONT *WHO BUT *CAR *CHICKEN CHICKEN                 JOHN FISH WONT EAT BUT CAN EAT CHICKEN
       25: JOHN *TELL *LOVE *WHO IX                                      JOHN LIKE IX IX IX
       89: *SOMETHING-ONE *GIVE *WOMAN *FRED IX NEW COAT                 JOHN IX GIVE MAN IX NEW COAT
       71: JOHN *FUTURE VISIT MARY                                       JOHN WILL VISIT MARY
       92: JOHN *WOMAN IX *WOMAN WOMAN BOOK                              JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
       90: JOHN *GIVE1 IX *IX WOMAN *ARRIVE                              JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
       30: JOHN *MARY *MARY *MARY *MARY                                  JOHN LIKE IX IX IX
      193: JOHN *SELF *CORN BOX                                          JOHN GIVE GIRL BOX
       36: MARY VEGETABLE *GIRL *GIVE *MARY *MARY                        MARY VEGETABLE KNOW IX LIKE CORN1
      139: JOHN *BUY1 WHAT YESTERDAY BOOK                                JOHN BUY WHAT YESTERDAY BOOK
      167: JOHN *TOY1 *MARY LOVE MARY                                    JOHN IX SAY LOVE MARY
       40: JOHN *VISIT *CORN *JOHN *MARY                                 JOHN IX THINK MARY LOVE
       28: JOHN *WHO *WHO *WHO IX                                        JOHN LIKE IX IX IX
      171: JOHN MARY BLAME                                               JOHN MARY BLAME
       43: JOHN *SHOULD BUY HOUSE                                        JOHN MUST BUY HOUSE
      108: *MAN *BOOK                                                    WOMAN ARRIVE
      174: PEOPLE GROUP GIVE1 *CORN TOY                                  PEOPLE GROUP GIVE1 JANA TOY
      113: IX CAR BLUE *MARY *BUY1                                       IX CAR BLUE SUE BUY
       50: *JOHN *SEE BUY CAR SHOULD                                     FUTURE JOHN BUY CAR SHOULD
      199: *JOHN CHOCOLATE WHO                                           LIKE CHOCOLATE WHO
      158: LOVE JOHN WHO                                                 LOVE JOHN WHO
       54: JOHN *JOHN *MARY BUY HOUSE                                    JOHN SHOULD NOT BUY HOUSE
      105: JOHN *VEGETABLE                                               JOHN LEG
      184: ALL BOY *GIVE1 TEACHER *GIRL                                  ALL BOY GIVE TEACHER APPLE
       57: JOHN *PREFER VISIT MARY                                       JOHN DECIDE VISIT MARY
      122: JOHN READ BOOK                                                JOHN READ BOOK
      189: JOHN *SELF *CORN *BUY1                                        JOHN GIVE GIRL BOX
      181: *SUE ARRIVE                                                   JOHN ARRIVE



```python
from my_model_selectors import SelectorDIC, SelectorBIC_orig, SelectorCV, SelectorBIC
np.seterr(invalid='ignore')

# selected features
features_names = ["features_grnd_distance", 'features_podel', 'features_ground', 'features_polar', 'features_norm', 'features_delta']
features_list = [features_grnd_distance, features_podel, features_ground, features_polar, features_norm, features_delta]   
# model_selectors
model_selector_names = ['SelectorBIC', 'SelectorBIC_orig', 'SelectorDIC', 'SelectorCV']
model_selector_list = [SelectorBIC, SelectorBIC_orig, SelectorDIC, SelectorCV]

sel_features_list = features_list[:3]
sel_model_list = model_selector_list[:2]
# pre-build test_set
test_set_list = [asl.build_test(features) for j, features in enumerate(sel_features_list)]

for i, model_selector in enumerate(sel_model_list):
    for j, features in enumerate(sel_features_list):
        print("model_selector=%s, features=%s" % (model_selector_names[i], newfeature_names[j]))
        models = train_all_words(features, model_selector_list[i])
        test_set = test_set_list[j]
        probabilities, guesses = recognize(models, test_set)
        show_errors(guesses, test_set)
```

**Question 3:**  Summarize the error results from three combinations of features and model selectors.  What was the "best" combination and why?  What additional information might we use to improve our WER?  For more insight on improving WER, take a look at the introduction to Part 4.

**Answer 3:**
I run all selectors and features, includes 3 custom features. Both 'features_norm', 'features_delta', did not scored under 60% WER. Two of custom features "features_norm_poldel" and "features_norm_pol" scored little under 60%, see Answer 1. The final selected features are 'features_podel' (custom features), 'features_ground', 'features_polar'.

The <b>features_podel</b> with SelectorBIC performed the best, yield 50% success rate. The features_podel is features_polar plus features_delta. I believe that the aditional delta time information provide the model more 'clues' for identifying the sequences.

I notice that the selectorCV performed better with features_poral than features_podel. My guess is that delta occurances may not consistantly among sequences. If the differences are significant enough between the training and test folds, the selector is confused. Does this make sense?

The polar coordinates features are performed better than Cartesian coordinate. I believe that the main reason is that the values between polar coordinates, distance and angle, are more distinct than the values between Cartesian coordinates x and y. Does this make sense?

Below is the output of the tests. The total number of test_set sequence is 178.

|                      |     |SelectorBIC      |     |  SelectorDIC     |     | SelectorCV       |     | SelectorBIC_orig   |
| -------------------- |:---:|:---------------:|:---:|:----------------:|:---:|:----------------:|:---:|:------------------:|
| <b>features name</b> | WER | number regonized| WER | number regonized | WER | number regonized | WER | number regonized   |
| features_podel       | 0.5674 | 77 | 0.5337 | 83 | 0.5618 | 79 | 0.5 | 89
| features_ground      | 0.5505 | 80 | 0.5786 | 75 | 0.5618 |78 | 0.5618 | 78 |
| features_polar       | 0.5393 | 82 | 0.5449 | 81 | 0.5337 |83 | 0.5281 | 84 |
| features_grnd_distance | 0.5393 | 82 | 0.5449 | 81 | 0.5842 | 74 | 0.5618 | 78 |

Selector Improvement I have tried

1. I had tried on improving selectorBIC by making the parameter as 'param = num_state x num_state + 2 x num_state  - 1' (just a hacking) but only make it worse.

2. I tried use 2 selectors. The word selection criteria:
    a. if both selectors top word are the same, choose the word
    b. choose the word with the best score
 this strategy performed between 2 selectors. I think the reason is that the probability of choose the poor selector's word is 50%, so the the outcoming is worse than the good selector but better than the poor selector.

3. I tried use 2 selectors. Compare the top 3 words of each selector and choose the word
    a. if both selectors top word are the same, choose the word
    b. if both selectors top 3 has common words, choose the best common word.
    c. if no word share by both selector top 3, choose the word with highest score.
 this strategy performed very poor. I believe that the reason is that often both selector's 2nd and/or 3nd guesses the same word. By boost the selectability of common words ends up chosen a wrong word.  

Selector Improvement I like to try

1. I think for selectorDIC, instead calculate the mean of the model on other word, calculate the diviation of scores of other words and this word. This way, we select the model that could best identify this_word. Does that make sense?

2. I skip the Part 4 because not clear about the data. I downloaded the SLM data but do not know how to use the files (10 total). For example in ukn.3.lm:

```
    \1-grams:
    -0.9031478	</s>
    -99	<s>	-0.947363
    -2.51825	ALL	-0.1719009
    -2.51825	ANN	-0.1719009
    ...
    \2-grams:
    -2.066325	<s> ALL	-0.554368
    -2.624788	<s> ANN
    -1.95889	<s> ARRIVE	-0.554368
    ...
    \3-grams:
    -0.08973494	<s> ALL BOY
    -0.1023415	TEACHER APPLE </s>
    -0.07130415	THROW APPLE WHO
    -0.1209455	<s> ARRIVE WHO
    -0.07609817	BROTHER ARRIVE </s>
    -0.1374041	JOHN ARRIVE </s>

```
    * what does the number in front of a word mean?
    * what does the number after a word mean?
    * what does <s> means?

I did not find any documentation explaining the data and usage.

<a id='part3_test'></a>
### Recognizer Unit Tests
Run the following unit tests as a sanity check on the defined recognizer.  The test simply looks for some valid values but is not exhaustive. However, the project should not be submitted if these tests don't pass.


```python
from asl_test_recognizer import TestRecognize
suite = unittest.TestLoader().loadTestsFromModule(TestRecognize())
unittest.TextTestRunner().run(suite)
```

<a id='part4_info'></a>
## PART 4: (OPTIONAL)  Improve the WER with Language Models
We've squeezed just about as much as we can out of the model and still only get about 50% of the words right! Surely we can do better than that.  Probability to the rescue again in the form of [statistical language models (SLM)](https://en.wikipedia.org/wiki/Language_model).  The basic idea is that each word has some probability of occurrence within the set, and some probability that it is adjacent to specific other words. We can use that additional information to make better choices.

##### Additional reading and resources
- [Introduction to N-grams (Stanford Jurafsky slides)](https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf)
- [Speech Recognition Techniques for a Sign Language Recognition System, Philippe Dreuw et al](https://www-i6.informatik.rwth-aachen.de/publications/download/154/Dreuw--2007.pdf) see the improved results of applying LM on *this* data!
- [SLM data for *this* ASL dataset](ftp://wasserstoff.informatik.rwth-aachen.de/pub/rwth-boston-104/lm/)

##### Optional challenge
The recognizer you implemented in Part 3 is equivalent to a "0-gram" SLM.  Improve the WER with the SLM data provided with the data set in the link above using "1-gram", "2-gram", and/or "3-gram" statistics. The `probabilities` data you've already calculated will be useful and can be turned into a pandas DataFrame if desired (see next cell).  
Good luck!  Share your results with the class!


```python
# create a DataFrame of log likelihoods for the test word items
df_probs = pd.DataFrame(data=probabilities)
df_probs.head()
```
