# Fully Parameterized Example Fine Tuning T5 Large Language Model on Databricks for Summarization<br><br>

A T5 large language model is a type of natural language processing (NLP) model that is based on the Transformer architecture. It is a neural network that has been trained on a large corpus of text data to generate human-like responses to various language tasks, such as summarization, translation, question-answering, and text completion.  The "T5" in the name refers to the "Text-to-Text Transfer Transformer," which is the framework that the model is built on. This framework uses a single neural network architecture to handle a wide variety of NLP tasks, which allows for faster and more efficient training and inference.  T5 large language models are particularly useful for generating high-quality text in a variety of contexts, such as chatbots, language translation systems, and content generation tools. They are capable of understanding complex language structures and generating responses that are both grammatically correct and semantically meaningful.

### Clone Down the Repo into Databricks Workspace: <br>
- git clone https://github.com/robert-altmiller/dbricks_llms.git

## Step 1: Update User Defined Parameters
## Navigate to the following folder: <br>

- databricks_llm folder --> data_preparation.py
- Update any of the following parameters below.  You do not need to update the Databricks Personal Access Token (PAT) or Workspace Instance because they are created or fetched automatically.  If you want to speed up the time it takes to do the fine tuning of the T5 model you can test different gpu spark driver types.<br><br>
![user_parameters.png](/readme_images/user_parameters.png)

## Step 2: Update the Requirements File Location
## Navigate to the following folder: <br>

- databricks_llm folder --> run folder --> summarization-(t5-11b) folder --> main.py
- Update the location of the requirements file.  This file loads local notebook libraries needing for fine tuning.<br><br>
![update_requirements_path.png](/readme_images/update_requirements_path.png)


## Step 3: Create and Switch to GPU Cluster
## Navigate to the following folder: <br>

- databricks_llm folder --> run folder --> summarization-(t5-11b) folder --> main.py
- Run the first two cells with any existing cluster.  After the GPU cluster is created switch to it and continue execution in the main.py notebook.<br><br>
![switch_cluster1.png](/readme_images/switch_cluster1.png)
![switch_cluster2.png](/readme_images/switch_cluster2.png)