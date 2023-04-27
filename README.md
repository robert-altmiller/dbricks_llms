# Simple Fully Parameterized Example Fine Tuning T5 Large Language Model on Databricks for Summarization<br><br>

A T5 large language model is a type of natural language processing (NLP) model that is based on the Transformer architecture. It is a neural network that has been trained on a large corpus of text data to generate human-like responses to various language tasks, such as summarization, translation, question-answering, and text completion.  The "T5" in the name refers to the "Text-to-Text Transfer Transformer," which is the framework that the model is built on. This framework uses a single neural network architecture to handle a wide variety of NLP tasks, which allows for faster and more efficient training and inference.  T5 large language models are particularly useful for generating high-quality text in a variety of contexts, such as chatbots, language translation systems, and content generation tools. They are capable of understanding complex language structures and generating responses that are both grammatically correct and semantically meaningful.


## Step 1: Update User Defined Parameters
## Navigate to the following folder: 
- databricks_llm folder --> data_preparation.py, and update any of the following parameters below that have a 'change' comment.  For example, we recommend updating cluster settings if you want to use a larger sized GPU to enable the fine tuning of the T5 model to go faster<br>
![user_parameters.png](/readme_images/user_parameters.png)

## Step 2: Change Location of Requirements.txt
## Navigate to the following folder: 
- databricks_llm folder --> run folder --> summarization-(t5-11b) folder --> main.py, and update the location of the requirements.txt file:<br>
![update_requirements_path.png](/readme_images/update_requirements_path.png)

## Step 3: Run the Main Program 
## Navigate to the following folder: 
- databricks_llm folder --> run folder --> summarization-(t5-11b) folder --> main.py, and run the first two cells to create the gpu cluster and switch to the gpu cluster.  You will need to wait for the gpu cluster to warm up before switching to it.<br>
![switch_cluster1.png](/readme_images/switch_cluster1.png)
![switch_cluster2.png](/readme_images/switch_cluster2.png)
