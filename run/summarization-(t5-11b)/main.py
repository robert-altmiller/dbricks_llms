# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-Tuning with T5-Small Transformer Model
# MAGIC ## This demonstrates basic fine-tuning with the `t5-small` model.  GPU cluster is created automatically and deleted at the end using the databricks cluster api 2.0.

# COMMAND ----------

# MAGIC %md
# MAGIC # Databricks 2.0 API Configuration (Cluster)

# COMMAND ----------

# DBTITLE 1,Databricks API Configuration
# MAGIC %run "../dbricks_api/cluster_api"

# COMMAND ----------

# MAGIC %md
# MAGIC # Check for GPU Cluster or Create GPU Cluster For Fine Tuning - Continue to Next Step After Switching to GPU Cluster

# COMMAND ----------

# DBTITLE 1,Create or Use Existing GPU Cluster
# check if gpu cluster already exists and if it doesn't create it

gpu_cluster_exists = False
response = list_clusters(databricks_instance, databricks_pat)
clusters_json = json.loads(response.text)["clusters"]

# search for gpu cluster with your name and if it exists start it
for cluster in clusters_json:
  if cluster["cluster_name"] == cluster_name: # gpu cluster exists
    clusterid = cluster["cluster_id"]
    start_cluster(databricks_instance, databricks_pat, clusterid)
    print(f"gpu '{cluster_name}' starting.....")
    print(f"gpu '{cluster_name}' already exists so please switch to it after it starts.....")
    gpu_cluster_exists = True
    break

# if gpu cluster does not exist create it and start it
if gpu_cluster_exists == False: # create gpu cluster
  response_create_cluster = create_cluster(databricks_instance, databricks_pat)
  clusterid = response_create_cluster.text 
  response_settings = get_cluster_settings(databricks_instance, databricks_pat, clusterid)
  print(f"gpu '{cluster_name}' created and starting so please switch to it.....")

print(f"gpu '{cluster_name}' cluster id: {clusterid}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Install Requirements File - Make Sure You Have Switched to the GPU Cluster Before Running This Step

# COMMAND ----------

# DBTITLE 1,Install Requirements
# MAGIC %pip install -r "/Workspace/Repos/robert.altmiller@databricks.com/dbricks_llms/run/summarization-(t5-11b)/requirements/requirements.txt"

# COMMAND ----------

# MAGIC %md
# MAGIC # Library Imports

# COMMAND ----------

# DBTITLE 1,Library Imports
# pandas and numpy libraries
import pandas as pd
# mlflow libraries
import mlflow, torch
# spark and hugging face libraries
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, udf
from transformers import pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preparation

# COMMAND ----------

# DBTITLE 1,Data Preparation
# MAGIC %run "../summarization-(t5-11b)/data_preparation/data_preparation"

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Fine Tuning Step
# MAGIC ## The `run_summarization.py` script is simply obtained from [transformers examples](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py)
# MAGIC ## There is a copy of it locally for you already in dbricks_llms folder --> summarization (t5-11b) --> hf_fine_tuning_script folder

# COMMAND ----------

# DBTITLE 1,Run Model Fine Tuning (Takes 55 - 60 Minutes to Complete With Current GPU Cluster Settings)
# MAGIC %sh
# MAGIC 
# MAGIC # huggingface tuning script path
# MAGIC for path in $huggingface_tuning_script_path 
# MAGIC   do 
# MAGIC     huggingface_tuning_script_path="$path" 
# MAGIC   done
# MAGIC echo $huggingface_tuning_script_path
# MAGIC 
# MAGIC # huggingface tuned model path
# MAGIC for path in $huggingface_tuned_model_path 
# MAGIC   do 
# MAGIC     huggingface_tuned_model_path="$path" 
# MAGIC   done
# MAGIC echo $huggingface_tuned_model_path
# MAGIC 
# MAGIC # local train data filepath
# MAGIC for path in $localtraindatafilepath 
# MAGIC   do 
# MAGIC     localtraindatafilepath="$path" 
# MAGIC   done
# MAGIC echo $localtraindatafilepath
# MAGIC 
# MAGIC # local validate data filepath
# MAGIC for path in $localvalidatedatafilepath 
# MAGIC   do 
# MAGIC     localvalidatedatafilepath="$path" 
# MAGIC   done
# MAGIC echo $localvalidatedatafilepath
# MAGIC 
# MAGIC python \
# MAGIC   $huggingface_tuning_script_path \
# MAGIC   --model_name_or_path t5-small \
# MAGIC   --do_train \
# MAGIC   --do_eval \
# MAGIC   --train_file $localtraindatafilepath \
# MAGIC   --validation_file $localvalidatedatafilepath \
# MAGIC   --source_prefix "summarize: " \
# MAGIC   --output_dir $huggingface_tuned_model_path \
# MAGIC   --optim adafactor \
# MAGIC   --num_train_epochs 8 \
# MAGIC   --bf16 \
# MAGIC   --per_device_train_batch_size 64 \
# MAGIC   --per_device_eval_batch_size 64 \
# MAGIC   --predict_with_generate \
# MAGIC   --run_name "t5-small-fine-tune-reviews"

# COMMAND ----------

# MAGIC %md
# MAGIC # View Created Artifacts in DBFS From Model Fine Tuning in Previous Step
# MAGIC ## note: checkpoint folders (see below) are not needed to use fine tuned model with mlflow if you decide to do so

# COMMAND ----------

# DBTITLE 1,View Created Artifacts
dbutils.fs.ls(tuned_model_path.replace("/dbfs", ""))

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Inference Using Fine Tuned Model From Previous Step

# COMMAND ----------

# DBTITLE 1,Run Inference and Display Results
# summarizer pipeline
summarizer_pipeline = pipeline(pipeline_desc, \
  model = tuned_model_path, \
  tokenizer = tuned_model_path, \
  num_beams = num_beams, min_new_tokens = min_new_tokens)

# summarizer broadcast
summarizer_broadcast = sc.broadcast(summarizer_pipeline)

# pandas user defined function for summarizing reviews
@pandas_udf('string')
def summarize_review(reviews):
  pipe = summarizer_broadcast.value(("summarize: " + reviews).to_list(), batch_size = batch_size, truncation=True)
  return pd.Series([s['summary_text'] for s in pipe])

# read in cleaned camera reviews
camera_reviews_df = spark.read.format("delta").load(localdatacleanedfolderpath.replace("/dbfs", ""))

# review by product dataframe
review_by_product_df = camera_reviews_df.groupBy("product_id") \
  .agg(F.collect_list("review_body").alias("review_array"), F.count("*").alias("n")) \
  .filter("n >= 10") \
  .select("product_id", "n", F.concat_ws(" ", F.col("review_array")).alias("reviews")) \
  .withColumn("summary", summarize_review("reviews"))


# display reviews and summary
display(review_by_product_df.select("reviews", "summary").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC # Fine Tuned Model Can Be Deployed as a Real Time Endpoint
# MAGIC ## This model can then be deployed as a real-time endpoint! Check the `Models` and `Endpoints` tabs to the left in Databricks.

# COMMAND ----------

# DBTITLE 1,Serve Model as Real Time Endpoint
# get one sample review
sample_review = "summarize: " + review_by_product_df.select("reviews").head(1)[0]["reviews"]
print(sample_review)

# serve model as a real time endpoint
summarizer_pipeline = pipeline(pipeline_desc, model = tuned_model_path, tokenizer = tuned_model_path, num_beams = num_beams, min_new_tokens = min_new_tokens, device = "cuda:0")

# get real time endpoint prediction on one summary
summarizer_pipeline(sample_review, truncation = True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Terminate and Remove Databricks GPU Cluster

# COMMAND ----------

response = delete_cluster(databricks_instance, databricks_pat, clusterid)
print(f"cluster id: '{clusterid}' permanently deleted; response: {response}")
