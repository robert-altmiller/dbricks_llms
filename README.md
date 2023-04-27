# Simple Fully Parameterized Example Fine Tuning T5 Large Language Model on Databricks for Summarization<br><br>

A T5 large language model is a type of natural language processing (NLP) model that is based on the Transformer architecture. It is a neural network that has been trained on a large corpus of text data to generate human-like responses to various language tasks, such as summarization, translation, question-answering, and text completion.  The "T5" in the name refers to the "Text-to-Text Transfer Transformer," which is the framework that the model is built on. This framework uses a single neural network architecture to handle a wide variety of NLP tasks, which allows for faster and more efficient training and inference.  T5 large language models are particularly useful for generating high-quality text in a variety of contexts, such as chatbots, language translation systems, and content generation tools. They are capable of understanding complex language structures and generating responses that are both grammatically correct and semantically meaningful.

## Step 1: Add Databricks Workspace Instance and PAT
## Navigate to the following folder:
- databricks_llm folder --> databricks_api folder --> cluster_base.py, and add your Databricks Workspace Instance and Databricks Person Access Token (PAT)<br>
![databricks_instance_pat.png](/readme_images/databricks_instance_pat.png)

## Step 2: Update User Defined Parameters
## Navigate to the following folder: 
- databricks_llm folder --> summarization folder --> t5-11b folder --> data_preparation.py, and update any of the following parameters below:<br>
![user_parameters.png](/readme_images/user_parameters.png)