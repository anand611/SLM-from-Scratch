"""## Step 1: Get Dataset

In this step, we will do the following:

(1) We'll load the 'TinyStories dataset' from the 'datasets' module

"""

from datasets import load_dataset

# Load the TinyStories dataset
ds = load_dataset("roneneldan/TinyStories")
print("Dataset loaded")
#endregion

