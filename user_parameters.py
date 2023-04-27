# Databricks notebook source
# DBTITLE 1,User Defined Parameters
# authentication parameters (don't change)
databricks_instance = spark.conf.get("spark.databricks.workspaceUrl") # format "adb-723483445396.18.azuredatabricks.net" (no https://) (don't change)
databricks_pat = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get() # format "dapi*****" (don't change)

# user parameters (don't change)
username1 = spark.sql('select current_user() as user').collect()[0]['user'] # firstname.lastname@email.com (don't change)
username2 = username1.split("@")[0] # firstname.lastname (don't change)

# gpu cluster parameters (change)
cluster_name = f"{username2.replace('.', ' ')}'s cluster (g5 gpu)" # don't change
spark_min_workers = 4 # change
spark_max_workers = 8 # change
spark_auto_terminate_mins = 120 # change
spark_version = "12.2.x-gpu-ml-scala2.12" # change
spark_driver_type = "g5.4xlarge" # change

# externanal data preparation parameters (change)
externaldatafilename = "amazon_reviews_us_Camera_v1_00.tsv.gz" # external location filename (change)
externaldatafilepath = f"https://s3.amazonaws.com/amazon-reviews-pds/tsv/{externaldatafilename}" # external data location (change)

# dbfs file and folder path parameters (change)
localdbfsbasepath = "/dbfs/temp" # dbfs base folder location (change)
localdbfsfoldername = "review" # name of dbfs folder where all data will be store in databricks (change)
localdbfscleanedfoldername = "cleaned" # name of dbfs folder where cleaned dataset with be stored in databricks (change)
localtraindatafilename = "camera_reviews_train.csv" # training dataset filename (change)
localvalidatedatafilename = "camera_reviews_val.csv" # validation dataset filename (change)


# train, test, split parameter for training and validation datasets (change)
train_test_split_val = .8 # train, test, split parameters (change)
seed_val = 42

# cleaned data parameters
# these parameters are used to get a sample of the original dataset to clean and run fine tuning on
camera_reviews_sample_percentage = .1 # sample this percent of the original dataset (change)

# tuned model parameters (change)
fine_tune_requirements_path = f"/Workspace/Repos/{username1}/dbricks_llms/run/summarization-(t5-11b)/requirements/requirements.txt" # change
tuned_model_path = f"{localdbfsbasepath}/{username1}/{localdbfsfoldername}/t5-small-summary" # change
pipeline_desc = "summarization" # hugging face pipeline parameters (change)
num_beams = 10 # change
min_new_tokens = 50 # change
batch_size = 8 # change

# environment variable settings for using with %sh (don't change)
# hf = hugging face
os.environ["DATABRICKS_TOKEN"] = databricks_pat # don't change
os.environ["DATABRICKS_HOST"] = f"https://{databricks_instance}" # don't change
os.environ["MLFLOW_EXPERIMENT_NAME"] = f"/Users/{username1}/fine-tuning-t5" # don't change
os.environ["MLFLOW_FLATTEN_PARAMS"] = "true"
os.environ["TRANSFORMERS_CACHE"] = f"{localdbfsbasepath}/{username1}/cache/hf" # don't change
os.environ["huggingface_tuned_model_path"] = tuned_model_path # don't change
os.environ["huggingface_tuning_script_path"] = f"/Workspace/Repos/{username1}/dbricks_llms/run/summarization-(t5-11b)/hf_fine_tuning_script/run_summarization.py" # don't change
