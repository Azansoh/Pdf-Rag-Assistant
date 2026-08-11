import os
import warnings
from langchain_community.document_loaders import TextLoader
# Hide deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Get absolute path of the folder where test.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Point directly to notex.txt inside the document loader folder
file_path = os.path.join(script_dir, "notes.txt")

# Load file with proper encoding
data = TextLoader(file_path, encoding="utf-8")
doc = data.load()

print("--- Document Loaded Successfully ---")
print(doc[0])