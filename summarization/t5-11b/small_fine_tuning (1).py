# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-Tuning with t5-small
# MAGIC ## This demonstrates basic fine-tuning with the `t5-small` model. This notebook should be run on an instance with 1 Ampere architecture GPU, such as an A10. Use Databricks Runtime 12.2 ML GPU or higher.  
# MAGIC ## GPU cluster is created automatically and deleted at the end using the databricks cluster api 2.0.

# COMMAND ----------

# MAGIC %md
# MAGIC # Databricks 2.0 API Configuration + Library Imports

# COMMAND ----------

# DBTITLE 1,Databricks API Configuration
# MAGIC %run "../../dbricks_api/cluster_base"

# COMMAND ----------

# MAGIC %md
# MAGIC # Check for GPU Cluster or Create GPU Cluster For Fine Tuning

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

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preparation - Proceed With This Step After Switching to GPU Cluster From Previous Step

# COMMAND ----------

# DBTITLE 1,Data Preparation
# MAGIC %run "./data_preparation"

# COMMAND ----------

# MAGIC %md
# MAGIC # Set additional environment variables to enable integration between Hugging Face's training and MLflow hosted in Databricks (and make sure to use the shared cache again!).  
# MAGIC ## You can also set `HF_MLFLOW_LOG_ARTIFACTS` to have it log all checkpoints to MLflow, but they can be large.

# COMMAND ----------

# DBTITLE 1,Notebook Parameters (Don't Change)
# tuned model parameters
tuned_model_path = f"/dbfs/tmp/{username}/review/t5-small-summary"
num_beams = 10
min_new_tokens = 50
batch_size = 8

# hugging face and ml flow parameters
pipeline_desc = "summarization"
ml_flow_run_name = "t5-small-fine-tune-reviews"
ml_flow_experiment_path = "/Users/sean.owen@databricks.com/fine-tuning-t5"
ml_flow_registered_model_name = "sean_t5_small_fine_tune_reviews"
ml_flow_artifact_name = "review_summarizer"

# environment variables (do not change)
os.environ["huggingface_tuned_model_path"] = tuned_model_path
os.environ["huggingface_tuning_script_path"] = f"/Workspace/Repos/{username}/dbricks_llms/summarization/t5-11b/run_summarization.py"
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ['DATABRICKS_HOST'] = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['TRANSFORMERS_CACHE'] = f"/dbfs/tmp/{username}/cache/hf"
os.environ['MLFLOW_EXPERIMENT_NAME'] = f"/Users/{username}/fine-tuning-t5"
os.environ['MLFLOW_FLATTEN_PARAMS'] = "true"

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Fine Tuning Step
# MAGIC ## The `run_summarization.py` script is simply obtained from [transformers examples](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py)
# MAGIC ## There is a copy of it locally for you already in dbricks_llms folder --> summarization folder

# COMMAND ----------

# DBTITLE 1,Run Model Fine Tuning (Takes 55 - 60 Minutes to Complete)
# MAGIC %sh
# MAGIC 
# MAGIC # huggingface tuning script path
# MAGIC for path in $huggingface_tuning_script_path 
# MAGIC   do 
# MAGIC     huggingface_tuning_script_path="$path" 
# MAGIC   done
# MAGIC 
# MAGIC # huggingface tuned model path
# MAGIC for path in $huggingface_tuned_model_path 
# MAGIC   do 
# MAGIC     huggingface_tuned_model_path="$path" 
# MAGIC   done
# MAGIC 
# MAGIC # local train data filepath
# MAGIC for path in $localtraindatafilepath 
# MAGIC   do 
# MAGIC     localtraindatafilepath="$path" 
# MAGIC   done
# MAGIC 
# MAGIC # local validate data filepath
# MAGIC for path in $localvalidatedatafilepath 
# MAGIC   do 
# MAGIC     localvalidatedatafilepath="$path" 
# MAGIC   done
# MAGIC 
# MAGIC export DATABRICKS_TOKEN && export DATABRICKS_HOST && export MLFLOW_EXPERIMENT_NAME && export MLFLOW_FLATTEN_PARAMS && python \
# MAGIC     $huggingface_tuning_script_path \
# MAGIC     --model_name_or_path t5-small \
# MAGIC     --do_train \
# MAGIC     --do_eval \
# MAGIC     --train_file $localtraindatafilepath \
# MAGIC     --validation_file $localvalidatedatafilepath \
# MAGIC     --source_prefix "summarize: " \
# MAGIC     --output_dir $huggingface_tuned_model_path \
# MAGIC     --optim adafactor \
# MAGIC     --num_train_epochs 8 \
# MAGIC     --bf16 \
# MAGIC     --per_device_train_batch_size 64 \
# MAGIC     --per_device_eval_batch_size 64 \
# MAGIC     --predict_with_generate \
# MAGIC     --run_name "t5-small-fine-tune-reviews"

# COMMAND ----------

# MAGIC %md
# MAGIC # View Created Artifacts in DBFS From Model Fine Tuning in Previous Step

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

# MAGIC %sh 
# MAGIC 
# MAGIC rm -r /tmp/t5-small-summary; 
# MAGIC mkdir -p /tmp/t5-small-summary
# MAGIC cp /dbfs/tmp/sean.owen@databricks.com/review/t5-small-summary/* /tmp/t5-small-summary

# COMMAND ----------

# DBTITLE 1,This model can even be managed by MLFlow by wrapping up its usage in a simple custom `PythonModel`:
# review model class
class ReviewModel(mlflow.pyfunc.PythonModel):

  # load context function
  def load_context(self, context):
    self.pipeline = pipeline(pipeline_desc, \
      model = context.artifacts["pipeline"], tokenizer = context.artifacts["pipeline"], \
      num_beams = num_beams, min_new_tokens = min_new_tokens, \
      device = 0 if torch.cuda.is_available() else -1)

  # predict function
  def predict(self, context, model_input): 
    texts = ("summarize: " + model_input.iloc[:, 0]).to_list()
    pipe = self.pipeline(texts, truncation=True, batch_size = batch_size)
    return pd.Series([s['summary_text'] for s in pipe])


# setup mlflow experiment
mlflow.set_experiment(ml_flow_experiment_path)
last_run_id = mlflow.search_runs(filter_string = f"tags.mlflow.runName = {ml_flow_run_name}")['run_id'].item()

# mlfow start run
with mlflow.start_run(run_id = last_run_id):
  mlflow.pyfunc.log_model
  (
    artifacts = {"pipeline": "/tmp/t5-small-summary"}, 
    artifact_path = ml_flow_artifact_name, 
    python_model = ReviewModel(),
    registered_model_name = ml_flow_registered_model_name
  )

# COMMAND ----------

# MAGIC %md Copy everything but the checkpoints, which are large and not necessary to serve the model
