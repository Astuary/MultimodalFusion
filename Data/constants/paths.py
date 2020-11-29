from typing import Optional

# path to the SDK folder
SDK_PATH: Optional[str] = "..\Dataset\CMU-MultimodalSDK"

# path to the folder where you want to store data
DATA_PATH_HIGH_LEVEL: Optional[str] = "..\Dataset\HighLevelData\high_level"
DATA_PATH_LABELS: Optional[str] = "..\Dataset\HighLevelData\labels"
DATA_PATH_RAW: Optional[str] = "..\Dataset\RawData"
ALIGNED_DATA_PATH_HIGH_LEVEL: Optional[str] = "..\Dataset\AlignedData\high_level"
ALIGNED_DATA_PATH_LABELS: Optional[str] = "..\Dataset\AlignedData\labels"

# path to a pretrained word embedding file
WORD_EMB_PATH: Optional[str] = None

# path to loaded word embedding matrix and corresponding word2id mapping
CACHE_PATH: Optional[str] = '..\embedding_and_mapping.pt'

