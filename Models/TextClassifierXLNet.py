import os
import sys
import import_ipynb
from constants import LOADER_PATH
if LOADER_PATH not in sys.path:
    sys.path.append(os.path.abspath(LOADER_PATH))
print(sys.path)
if not os.path.exists(LOADER_PATH):
    print("Check the relative address of data loaders in the constants/paths.py, current address is: ", LOADER_PATH)
from Data.dataloadercreator import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
config = XLNetConfig.from_pretrained('xlnet-base-cased')
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=6)
model.to(device)