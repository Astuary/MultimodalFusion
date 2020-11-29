# Hierarcical Attention for Multimodal Fusion
A novel Hierarchical Attention model to classify 6 classes of emotions from visual, verbal, and vocal features of a person’s speech in CMU-MOSEI dataset.

# File Structure
Let's look at the file structure of the entire project
## 🗄️  Data [Stores all the paths to the separete datasets and scripts to load & preprocess them]
 -🗃️ __init__.py [refers to all the environment variables]
 
 -🗃️ DataLoaderCreator.py [Script to align and preprocess the data, convert that data into dataloaders]
### 🗄️  Constants
  -🗃️ __init__.py [need this to let interpreter know that this is a package]
  
  -🗃️ paths.py [paths and variables are stored here]
## 🗄️  Dataset
-🗃️ __init__.py [need this to let interpreter know that this is a package]
### 🗄️  AlignedData [Need to run the Data/DataLoaderCreator.py script in order to align the data first]
 🗄️ 🗄️ high_level [all the high-level features]
 - COVAREP.csd [acoustics]
 - FACET 4.2.csd [visuals]
 - glove_vectors.csd [textual]
 - OpenFace_2.0.csd [visuals]
 - OpenSMILE.csd [visuals]
 
 🗄️ 🗄️ labels
 - All Labels.csd
### 🗄️  CMU-MultimodalSDK [Provides the functions to download, format, align the data; split it into train-valid-test splits]
Download this from https://github.com/A2Zadeh/CMU-MultimodalSDK/
### 🗄️  HighLevelData [Unaligned data with high level features]
🗄️ 🗄️ high_level [all the high-level features]
 - COVAREP.csd [acoustics]
 - FACET 4.2.csd [visuals]
 - glove_vectors.csd [textual]
 - OpenFace_2.0.csd [visuals]
 - OpenSMILE.csd [visuals]
 
 🗄️ 🗄️ labels
 - All Labels.csd
### 🗄️  RawData [We only needed one modality's raw data]
Can be found at http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/language/
- CMU_MOSEI_TimestampedWords.csd [textual]
## 🗄️  Models [all models have both .py and .ipynb versions]
- __init__.py [need this to let interpreter know that this is a package]
- AcousticsEmbdder.py/.ipynb [Extracts embeddings from COVAREP high level features with the DeltaSelfAttention module]
- TextEmbdder.py/.ipynb [Extracts embeddings from raw sentence level words with the DeltaSelfAttention module]
- VisualsEmbdder.py/.ipynb [Extracts embeddings from FaceT4.2 high level words with the DeltaSelfAttention module]
- DeltaSelfAttention.py/ipynb [Temporally and contextually attends to features within a modality]
- DCCA.py/.ipynb [Gathers all modality's extracted self-attended embeddings and run a deep canonical correlation analysis; CrossAttention module for the canonically fused embeddings also included in this script; Final results can be derived after the CrossAttention]
- TextClassifierXLNet.py/XLNet_CMU_MOSEI_Text.ipynb/TextEmbedderXLNet.ipynb/TextClassifierXLNetPrototype.ipynb [Misc models used for prototyping and experimenting]
🗄️ 🗄️ model_constants
  -🗃️ __init__.py [need this to let interpreter know that this is a package]
  
  -🗃️ paths.py [paths and variables for the models are stored here]
🗄️ 🗄️ saved_models [models you run and save the weights of will be found here]

# Pre-requisite Libraries
- PyTorch 1.7.0
- Scikit-learn 0.23
- Numpy 1.19
- Huggingface Transformers 3.5.0
- Ray Tune 1.0.1

# How to make it work
1. Download the dataset as per the instructions from https://github.com/A2Zadeh/CMU-MultimodalSDK/. See the Dataset folder contents for more details.
2. Change Data/constants and Models/model_constants path variables as per your local directory setup. Check out the files for more instructions and information on how they are currently setup.
3. For the modality alignment, run Data/DataLoaderCreator.py script. Always run all the scripts from the root folder as the relative paths are set like that.
4. Run Models/TextEmbedder.py, Models/AcousticEmbedder.py, and Models/VisualEmbedder.py to save the embeddings.
5. Run Models/DCCA (running this will start running the models in step 4 also, it's a long process, so to fragment into memory-manageable pieces, we can run it one by one.)
Note: If you are running step 4 first, make sure you are saving the hidden representations/Embeddings somewhere first. And Models/DCCA should load those instead of running the models again. 

Each section in the scripts or notebooks are commented with what they do and how to run those.
