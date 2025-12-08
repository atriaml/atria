# %% [markdown]
# # Atria Datasets Examples
#
# This notebook demonstrates how to use the `atria_datasets` library.
# We will explore various types of datasets one by one, load them, and check their documentation.

# %%
# Enable autoreload so changes in library are reflected automatically
%load_ext autoreload
%autoreload 2

# %% [markdown]
# ## List Available Dataset Modules
# Let's see what datasets are available in the library.

# %%
from atria_datasets import DATASET

available_modules = DATASET.list_all_modules()
print("Available dataset modules:")
for module in available_modules:
    print(f"- {module}")

# %% [markdown]
# ## 1. Image Classification Datasets
# Let's start with an image classification dataset example.

# %%
from atria_datasets import load_dataset_config

# Load dataset configuration
dataset_name = "tobacco3482/image_only"
dataset_config = load_dataset_config(dataset_name)

# %% [markdown]
# ### Dataset Documentation
# Display the documentation for this dataset.

# %%
help(dataset_config)

# %% [markdown]
# ### Build Dataset
# The dataset can be built locally using the `build` method. You can specify a directory where the data should be stored.

# %%
dataset_config.build(data_dir="./data/tobacco3482")

# %% [markdown]
# ### Explore Dataset
# After building, you can explore the dataset objects, e.g., images, labels, etc.

# %%
# Example: access train/test splits if available
if hasattr(dataset_config, "train"):
    print(f"Number of training samples: {len(dataset_config.train)}")
if hasattr(dataset_config, "test"):
    print(f"Number of test samples: {len(dataset_config.test)}")
