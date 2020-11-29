# Using CMU-Multimodal SDK

"""
Credits: https://github.com/Justin1904/CMU-MultimodalSDK-Tutorials

This file uses ***CMU-Multimodal SDK*** to load and process multimodal time-series dataset CMU-MOSEI.

We specify some constants in `./constans/paths.py`. Please first take a look and modify the paths to point to the correct folders.

## Downloading the data

We start off by (down)loading the datasets. In the SDK each dataset has three sets of content: `highlevel`, `raw` and `labels`. `highlevel` contains the extracted features for each modality (e.g OpenFace facial landmarks, openSMILE acoustic features) while `raw` contains the raw transctripts, phonemes. `labels` are self-explanatory. Note that some datasets have more than just one set of annotations so `labels` could also give you multiple files.

Currently there's a caveat that the SDK will not automatically detect if you have downloaded the data already. In event of that it will throw a `RuntimeError`. We work around that by `try/except`. This is not ideal but it will work for now.
"""

from Data.constants import SDK_PATH, DATA_PATH_HIGH_LEVEL, DATA_PATH_LABELS, ALIGNED_DATA_PATH_HIGH_LEVEL, ALIGNED_DATA_PATH_LABELS, WORD_EMB_PATH, CACHE_PATH, DATA_PATH_RAW
#from constants import SDK_PATH, DATA_PATH_HIGH_LEVEL, DATA_PATH_LABELS, ALIGNED_DATA_PATH_HIGH_LEVEL, ALIGNED_DATA_PATH_LABELS, WORD_EMB_PATH, CACHE_PATH, DATA_PATH_RAW
import os
import sys

if SDK_PATH is None:
    print("SDK path is not specified! Please specify first in constants/paths.py")
    exit(0)
else:
    sys.path.append(SDK_PATH)

if not os.path.exists(SDK_PATH):
    print("Check the relative address of SDK in the constants/paths.py, current address is: ", SDK_PATH)

if not os.path.exists(DATA_PATH_HIGH_LEVEL):
    print("Check the relative address of high level features of the data in the constants/paths.py, current address is: ", DATA_PATH_HIGH_LEVEL)

if not os.path.exists(DATA_PATH_LABELS):
    print("Check the relative address of labels of the high level features in the constants/paths.py, current address is: ", DATA_PATH_LABELS)

if not os.path.exists(ALIGNED_DATA_PATH_HIGH_LEVEL):
    print("Check the relative address of aligned high level features of the data in the constants/paths.py, current address is: ", ALIGNED_DATA_PATH_HIGH_LEVEL)

if not os.path.exists(ALIGNED_DATA_PATH_LABELS):
    print("Check the relative address of aligned labels of the high level features in the constants/paths.py, current address is: ", ALIGNED_DATA_PATH_LABELS)

if not os.path.exists(DATA_PATH_RAW):
    print("Check the relative address of raw features in the constants/paths.py, current address is: ", DATA_PATH_RAW)

import mmsdk
import os
import re
import numpy as np
from mmsdk import mmdatasdk as md
from subprocess import check_call, CalledProcessError

# create folders for storing the data
#if not os.path.exists(DATA_PATH):
#    check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

# download highlevel features, low-level (raw) data and labels for the dataset MOSEI
# if the files are already present, instead of downloading it you just load it yourself.
# here we use CMU_MOSEI dataset as example.

DATASET = md.cmu_mosei
SETUP = False # Set this to True if you are downloading and aligning the dataset for the first time 

#try:
#    md.mmdataset(DATASET.highlevel, DATA_PATH_HIGH_LEVEL)
#except RuntimeError:
#    print("High-level features have been downloaded previously.")

#try:
#    md.mmdataset(DATASET.raw, DATA_PATH)
#except RuntimeError:
#    print("Raw data have been downloaded previously.")
    
#try:
#    md.mmdataset(DATASET.labels, DATA_PATH_LABELS)
#except RuntimeError:
#    print("Labels have been downloaded previously.")

"""## Inspecting the downloaded files

We can print the files in the target data folder to see what files are there.

We can observe a bunch of files ending with `.csd` extension. This stands for ***computational sequences***, which is the underlying data structure for all features in the SDK. We will come back to that later when we load the data. For now we just print out what computational sequences we have downloaded.
"""

# list the directory contents... let's see what features there are
data_files = os.listdir(ALIGNED_DATA_PATH_HIGH_LEVEL)
print('\n'.join(data_files))
data_files = os.listdir(DATA_PATH_RAW)
print('\n'.join(data_files))
data_files = os.listdir(ALIGNED_DATA_PATH_LABELS)
print('\n'.join(data_files))

"""## Loading a multimodal dataset

Loading the dataset is as simple as telling the SDK what are the features you need and where are their computational sequences. You can construct a dictionary with format `{feature_name: csd_path}` and feed it to `mmdataset` object in the SDK.
"""

# define your different modalities - refer to the filenames of the CSD files
#visual_field = 'CMU_MOSEI_VisualFacet42'
#acoustic_field = 'CMU_MOSEI_COVAREP'
#text_field = 'CMU_MOSEI_TimestampedWordVectors'

# We have use different names for the files, make sure the names are same for your version of the data
visual_field = 'FACET 4.2'
acoustic_field = 'COAVAREP'
text_field = 'glove_vectors'
word_field = 'CMU_MOSEI_TimestampedWords'

features = [
    #text_field, 
    visual_field, 
    acoustic_field
]

raw_features = [word_field]

# Use the line below if you have just downloaded the high level unaligned features and comment out the line below that
# recipe = {feat: os.path.join(DATA_PATH_HIGH_LEVEL, feat) + '.csd' for feat in features}

recipe = {feat: os.path.join(ALIGNED_DATA_PATH_HIGH_LEVEL, feat) + '.csd' for feat in features}
recipe[word_field] = os.path.join(DATA_PATH_RAW, word_field) + '.csd'
print(recipe)
dataset = md.mmdataset(recipe)

"""## A peek into the dataset

The multimodal dataset, after loaded, has the following hierarchy:


```
            computational_sequence_1 ---...
           /                                   ...
          /                                    /
         /                          first_video     features -- T X N array
        /                          /               /
dataset ---computational_sequence_2 -- second_video
        \                          \               \
         \                          third_video     intervals -- T X 2 array
          \                                    \...
           \
            computational_sequence_3 ---...
```

It looks like a nested dictionary and can be indexed as if it is a nested dictionary. A dataset contains multiple computational sequences whose key is the `text_field`, `visual_field`, `acoustic_field` as defined above. Each computational sequence, however, has multiple video IDs in it, and different computational sequences are supposed to have the same set of video IDs. Within each video, there are two arrays: `features` and `intervals`, denoting the feature values at each time step and the start and end timestamp for each step. We can take a look at its content.
"""

print(list(dataset.keys()))
print("=" * 80)

print(list(dataset[visual_field].keys())[:10])
print("=" * 80)

some_id = list(dataset[visual_field].keys())[15]
print(list(dataset[visual_field][some_id].keys()))
print("=" * 80)

word_id = list(dataset[word_field].keys())[15]
print(list(dataset[word_field][word_id].keys()))
print("=" * 80)
print(dataset[word_field].keys())

print('Intervals')
print(list(dataset[visual_field][some_id]['intervals'].shape))
#print(list(dataset[text_field][some_id]['intervals'].shape))
print(list(dataset[word_field][word_id]['intervals'].shape))
print(list(dataset[acoustic_field][some_id]['intervals'].shape))
print("=" * 80)

print('Features')
print(list(dataset[visual_field][some_id]['features'].shape))
#print(list(dataset[text_field][some_id]['features'].shape))
print(list(dataset[word_field][word_id]['features'].shape))
print(list(dataset[acoustic_field][some_id]['features'].shape))
print("Different modalities have different number of time steps!")

"""## Alignment of multimodal time series

To work with multimodal time series that contains multiple views of data with different frequencies, we have to first align them to a ***pivot*** modality. The convention is to align to ***words***. Alignment groups feature vectors from other modalities into bins denoted by the timestamps of the pivot modality, and apply a certain processing function to each bin. We call this function ***collapse function***, because usually it is a pooling function that collapses multiple feature vectors from another modality into one single vector. This will give you sequences of same lengths in each modality (as the length of the pivot modality) for all videos.

Here we define our collapse funtion as simple averaging. We feed the function to the SDK when we invoke `align` method. Note that the SDK always expect collapse functions with two arguments: `intervals` and `features`. Even if you don't use intervals (as is in the case below) you still need to define your function in the following way.

***Note: Currently the SDK applies the collapse function to all modalities including the pivot, and obviously text modality cannot be "averaged", causing some errors. My solution is to define the avg function such that it averages the features when it can, and return the content as is when it cannot average.***
"""

# Don't run this and the next cell if you have already aligned the data
if SETUP:
# we define a simple averaging function that does not depend on intervals
    def avg(intervals: np.array, features: np.array) -> np.array:
        try:
            return np.average(features, axis=0)
        except:
            return features

    # first we align to words with averaging, collapse_function receives a list of functions
    dataset.align(text_field, collapse_functions=[avg])

if SETUP:
    deploy_files={x:x for x in dataset.keys()}
    dataset.deploy("hl1",deploy_files)

"""## Append annotations to the dataset and get the data points

Now that we have a preprocessed dataset, all we need to do is to apply annotations to the data. Annotations are also computational sequences, since they are also just some values distributed on different time spans (e.g 1-3s is 'angry', 12-26s is 'neutral'). Hence, we just add the label computational sequence to the dataset and then align to the labels. Since we (may) want to preserve the whole sequences, this time we don't specify any collapse functions when aligning. 

Note that after alignment, the keys in the dataset changes from `video_id` to `video_id[segment_no]`, because alignment will segment each datapoint based on the segmentation of the pivot modality (in this case, it is segmented based on labels, which is what we need, and yes, one code block ago they are segmented to word level, which I didn't show you).

***Important: DO NOT add the labels together at the beginning, the labels will be segmented during the first alignment to words. This also holds for any situation where you want to do multiple levels of alignment.***
"""

#label_field = 'CMU_MOSEI_Labels'
label_field = 'All Labels'

# we add and align to lables to obtain labeled segments
# this time we don't apply collapse functions so that the temporal sequences are preserved
label_recipe = {label_field: os.path.join(ALIGNED_DATA_PATH_LABELS, label_field + '.csd')}
dataset.add_computational_sequences(label_recipe, destination=None)

# Uncomment the line below if you haven't aligned the labels 
#dataset.align(label_field)

# Only run this cell if you just aligned the data with the labels and now you want to store the alined data
if SETUP:
    deploy_files={x:x for x in dataset.keys()}
    dataset.deploy("fl1",deploy_files)

# check out what the keys look like now
print(list(dataset[word_field].keys())[55])

"""## Splitting the dataset

Now it comes to our final step: splitting the dataset into train/dev/test splits. This code block is a bit long in itself, so be patience and step through carefully with the explanatory comments.

The SDK provides the splits in terms of video IDs (which video belong to which split), however, after alignment our dataset keys already changed from `video_id` to `video_id[segment_no]`. Hence, we need to extract the video ID when looping through the data to determine which split each data point belongs to.

In the following data processing, I also include instance-wise Z-normalization (subtract by mean and divide by standard dev) and converted words to unique IDs.

This example is based on PyTorch so I am using PyTorch related utils, but the same procedure should be easy to adapt to other frameworks.
"""

# obtain the train/dev/test splits - these splits are based on video IDs
#train_split = DATASET.standard_folds.standard_train_fold
#dev_split = DATASET.standard_folds.standard_valid_fold
#test_split = DATASET.standard_folds.standard_test_fold

import math
import random

identifiers = list(dataset[label_field].keys())
#print(identifiers[:10])
random.shuffle(identifiers)
print(len(identifiers))

train_index = math.floor(len(identifiers)*0.8)
#dev_index = math.floor(len(identifiers)*0.85)
#print(train_index)
#print(dev_index)

train_split = identifiers[:train_index]
#dev_split = identifiers[train_index:dev_index]
test_split = identifiers[train_index:]

train_split = identifiers[:10]
#dev_split = identifiers[10:15]
test_split = identifiers[10:300]

# inspect the splits: they only contain video IDs
#print(test_split)

# we can see they are in the format of 'video_id[segment_no]', but the splits was specified with video_id only
# we need to use regex or something to match the video IDs...
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm_notebook
from collections import defaultdict

from pytorch_transformers import XLNetTokenizer

# a sentinel epsilon for safe division, without it we will replace illegal values with a constant
EPS = 0

# construct a word2id mapping that automatically takes increment when new words are encountered
word2id = defaultdict(lambda: len(word2id))
sep = word2id['[SEP]']
cls = word2id['[CLS]']
pad = word2id['[PAD]']
sp = word2id['[SP]']

# place holders for the final train/dev/test dataset
train = []
#dev = []
test = []

# define a regular expression to extract the video ID out of the keys
pattern = re.compile('(.*)\[.*\]')
num_drop = 0 # a counter to count how many data points went into some processing issues

#for segment in dataset[label_field].keys():
for segment in identifiers[:300]:

    print(segment)
    
    # get the video ID and the features out of the aligned dataset
    vid = re.search(pattern, segment).group(1)
    label = dataset[label_field][segment]['features']
    #_words = dataset[text_field][segment]['features']
    _words = dataset[word_field][vid]['features']
    ##_visual = dataset[visual_field][segment]['features']
    ##_acoustic = dataset[acoustic_field][segment]['features']

    #print(label.shape)
    #print(_words.shape)
    #print(_visual.shape)
    #print(_acoustic.shape)

    # if the sequences are not same length after alignment, there must be some problem with some modalities
    # we should drop it or inspect the data again
    #if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]:
    #    print(f"Encountered datapoint {vid} with text shape {_words.shape}, visual shape {_visual.shape}, acoustic shape {_acoustic.shape}")
    #    num_drop += 1
    #    continue

    # remove nan values
    label = np.nan_to_num(label)
    ##_visual = np.nan_to_num(_visual)
    ##_acoustic = np.nan_to_num(_acoustic)

    print(label.shape)
    print(_words.shape)
    ##print(_visual.shape)
    ##print(_acoustic.shape)

    # remove speech pause tokens - this is in general helpful
    # we should remove speech pauses and corresponding visual/acoustic features together
    # otherwise modalities would no longer be aligned
    words = []
    ##visual = []
    ##acoustic = []
    for i, word in enumerate(_words):
        if word[0] == b'sp':
            word[0] = '[SP]'
        #print(len(_words))
        #print(word[0])
        words.append(word2id[word[0].decode('utf-8')]) # SDK stores strings as bytes, decode into strings here
        #words.append(word2id[word[0]])

    words.append(word2id['[SEP]'])
    words.append(word2id['[CLS]'])
    ##visual.append(_visual)
    ##acoustic.append(_acoustic)

    words = np.asarray(words)
    ##visual = np.asarray(visual)
    ##acoustic = np.asarray(acoustic)

    # z-normalization per instance and remove nan/infs
    ##visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
    ##acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))

    if segment in train_split:
        ##train.append(((words, visual, acoustic), label, segment))
        train.append((words, label, segment))
    #elif segment in dev_split:
    #    dev.append(((words, visual, acoustic), label, segment))
    elif segment in test_split:
        test.append((words, label, segment))
    else:
        print(f"Found video that doesn't belong to any splits: {vid}")

    #input('Enter: ')

print(f"Total number of {num_drop} datapoints have been dropped.")

# turn off the word2id - define a named function here to allow for pickling
def return_unk():
    return UNK
word2id.default_factory = return_unk

"""## Inspect the dataset

Now that we have loaded the data, we can check the sizes of each split, data point shapes, vocabulary size, etc.
"""

# let's see the size of each set and shape of data
print(len(train))
#print(len(dev))
print(len(test))

print(train[0][0].shape)
##print(train[0][0][1].shape)
##print(train[0][0][2].shape)
print(train[0][1].shape)
print(train[0][1])

print(f"Total vocab size: {len(word2id)}")

"""## Collate function in PyTorch

Collate functions are functions used by PyTorch dataloader to gather batched data from dataset. It loads multiple data points from an iterable dataset object and put them in a certain format. Here we just use the lists we've constructed as the dataset and assume PyTorch dataloader will operate on that.
"""

def multi_collate(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
    ##sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=pad)
    sentences = pad_sequence([torch.LongTensor(sample[0]) for sample in batch])
    #print([sample[0][0] for sample in batch])
    #attention_masks = pad_sequence([torch.FloatTensor(sample[0][0] != pad) for sample in batch])
    ##visual = pad_sequence([torch.FloatTensor(sample[0][1].reshape(-1, sample[0][1].shape[1]).T) for sample in batch])
    ##acoustic = pad_sequence([torch.FloatTensor(sample[0][2].reshape(-1, sample[0][2].shape[1]).T) for sample in batch])
    
    # lengths are useful later in using RNNs
    lengths = torch.LongTensor([sample[0].shape[0] for sample in batch])
    #return sentences, visual, acoustic, labels, attention_masks, lengths
    ##return sentences, visual, acoustic, labels, lengths
    return sentences, labels, lengths

# construct dataloaders, dev and test could use around ~X3 times batch size since no_grad is used during eval
batch_sz = 1
train_loader = DataLoader(train, shuffle=True, batch_size=batch_sz, collate_fn=multi_collate)
#dev_loader = DataLoader(dev, shuffle=False, batch_size=batch_sz*3, collate_fn=multi_collate)
test_loader = DataLoader(test, shuffle=False, batch_size=batch_sz, collate_fn=multi_collate)

# let's create a temporary dataloader just to see how the batch looks like
temp_loader = iter(DataLoader(train, shuffle=True, batch_size=8, collate_fn=multi_collate))
batch = next(temp_loader)

print(batch[0].shape) # word vectors, padded to maxlen
print(batch[1].shape) # visual features
print(batch[2].shape) # acoustic features
#print(batch[3]) # labels
#print(batch[4]) # Attention Masks
#print(batch[4]) # lengths

# Let's actually inspect the transcripts to ensure it's correct
#if SETUP:
id2word = {v:k for k, v in word2id.items()}
examine_target = train
idx = np.random.randint(0, len(examine_target))
print(' '.join(list(map(lambda x: id2word[x], examine_target[idx][0].tolist()))))
# print(' '.join(examine_target[idx][0]))
print(examine_target[idx][1])
print(examine_target[idx][2])

"""## Define a function which returns all 3 loaders for communication with models in the other files

The function will return train, dev (validation) and test dataloaders
"""

def get_dataloaders():
    #return train_loader, dev_loader, test_loader
    return train_loader, test_loader

def get_id_word():
    return id2word, word2id