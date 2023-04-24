# Databricks notebook source
# MAGIC %md
# MAGIC ## Fine-Tuning with t5-small
# MAGIC 
# MAGIC This demonstrates basic fine-tuning with the `t5-small` model. This notebook should be run on an instance with 1 Ampere architecture GPU, such as an A10. Use Databricks Runtime 12.2 ML GPU or higher.
# MAGIC 
# MAGIC This requires a few additional Python libraries, including an update to the very latest `transformers`, and additional CUDA tools:

# COMMAND ----------

# MAGIC %pip install 'transformers>=4.26.0' datasets evaluate rouge-score

# COMMAND ----------

# MAGIC %md
# MAGIC Set additional environment variables to enable integration between Hugging Face's training and MLflow hosted in Databricks (and make sure to use the shared cache again!)
# MAGIC You can also set `HF_MLFLOW_LOG_ARTIFACTS` to have it log all checkpoints to MLflow, but they can be large.

# COMMAND ----------

import os

os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ['DATABRICKS_HOST'] = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['TRANSFORMERS_CACHE'] = "/dbfs/tmp/sean.owen@databricks.com/cache/hf"
os.environ['MLFLOW_EXPERIMENT_NAME'] = "/Users/sean.owen@databricks.com/fine-tuning-t5"
os.environ['MLFLOW_FLATTEN_PARAMS'] = "true"

# COMMAND ----------

# MAGIC %md
# MAGIC The `run_summarization.py` script is simply obtained from [transformers examples](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py). Copy it into a repo of your choice, or simply sparse check-out the transformers repo and include only `examples/pytorch/summarization`. Either way, edit the paths below to correspond to the location of the runner script.

# COMMAND ----------

# MAGIC %sh export DATABRICKS_TOKEN && export DATABRICKS_HOST && export MLFLOW_EXPERIMENT_NAME && export MLFLOW_FLATTEN_PARAMS && python \
# MAGIC     /Workspace/Repos/sean.owen@databricks.com/summarization/run_summarization.py \
# MAGIC     --model_name_or_path t5-small \
# MAGIC     --do_train \
# MAGIC     --do_eval \
# MAGIC     --train_file /dbfs/tmp/sean.owen@databricks.com/review/camera_reviews_train.csv \
# MAGIC     --validation_file /dbfs/tmp/sean.owen@databricks.com/review/camera_reviews_val.csv \
# MAGIC     --source_prefix "summarize: " \
# MAGIC     --output_dir /dbfs/tmp/sean.owen@databricks.com/review/t5-small-summary \
# MAGIC     --optim adafactor \
# MAGIC     --num_train_epochs 8 \
# MAGIC     --bf16 \
# MAGIC     --per_device_train_batch_size 64 \
# MAGIC     --per_device_eval_batch_size 64 \
# MAGIC     --predict_with_generate \
# MAGIC     --run_name "t5-small-fine-tune-reviews"

# COMMAND ----------

# MAGIC %md
# MAGIC Same inference code as before, just built using the fine-tuned model that was produced above:

# COMMAND ----------

from pyspark.sql.functions import collect_list, concat_ws, col, count, pandas_udf
from transformers import pipeline
import pandas as pd

summarizer_pipeline = pipeline("summarization",\
  model="/dbfs/tmp/sean.owen@databricks.com/review/t5-small-summary",\
  tokenizer="/dbfs/tmp/sean.owen@databricks.com/review/t5-small-summary",\
  num_beams=10, min_new_tokens=50)
summarizer_broadcast = sc.broadcast(summarizer_pipeline)

@pandas_udf('string')
def summarize_review(reviews):
  pipe = summarizer_broadcast.value(("summarize: " + reviews).to_list(), batch_size=8, truncation=True)
  return pd.Series([s['summary_text'] for s in pipe])

camera_reviews_df = spark.read.format("delta").load("/tmp/sean.owen@databricks.com/review/cleaned")

review_by_product_df = camera_reviews_df.groupBy("product_id").\
  agg(collect_list("review_body").alias("review_array"), count("*").alias("n")).\
  filter("n >= 10").\
  select("product_id", "n", concat_ws(" ", col("review_array")).alias("reviews")).\
  withColumn("summary", summarize_review("reviews"))

display(review_by_product_df.select("reviews", "summary").limit(10))

# COMMAND ----------

from pyspark.sql.functions import collect_list, concat_ws, col, count, pandas_udf
from transformers import pipeline
import pandas as pd

summarizer_pipeline = pipeline("summarization",\
  model="/dbfs/tmp/sean.owen@databricks.com/review/t5-small-summary",\
  tokenizer="/dbfs/tmp/sean.owen@databricks.com/review/t5-small-summary",\
  num_beams=10, min_new_tokens=50)
summarizer_broadcast = sc.broadcast(summarizer_pipeline)

@pandas_udf('string')
def summarize_review(reviews):
  pipe = summarizer_broadcast.value(("summarize: " + reviews).to_list(), batch_size=8, truncation=True)
  return pd.Series([s['summary_text'] for s in pipe])

camera_reviews_df = spark.read.format("delta").load("/tmp/sean.owen@databricks.com/review/cleaned")

review_by_product_df = camera_reviews_df.groupBy("product_id").\
  agg(collect_list("review_body").alias("review_array"), count("*").alias("n")).\
  filter("n >= 10").\
  select("product_id", "n", concat_ws(" ", col("review_array")).alias("reviews")).\
  withColumn("summary", summarize_review("reviews"))

# COMMAND ----------

# MAGIC %md
# MAGIC This model can even be managed by MLFlow by wrapping up its usage in a simple custom `PythonModel`:

# COMMAND ----------

import mlflow
import torch

class ReviewModel(mlflow.pyfunc.PythonModel):
  
  def load_context(self, context):
    self.pipeline = pipeline("summarization", \
      model=context.artifacts["pipeline"], tokenizer=context.artifacts["pipeline"], \
      num_beams=10, min_new_tokens=50, \
      device=0 if torch.cuda.is_available() else -1)
    
  def predict(self, context, model_input): 
    texts = ("summarize: " + model_input.iloc[:,0]).to_list()
    pipe = self.pipeline(texts, truncation=True, batch_size=8)
    return pd.Series([s['summary_text'] for s in pipe])

# COMMAND ----------

# MAGIC %md Copy everything but the checkpoints, which are large and not necessary to serve the model

# COMMAND ----------

# MAGIC %sh rm -r /tmp/t5-small-summary ; mkdir -p /tmp/t5-small-summary ; cp /dbfs/tmp/sean.owen@databricks.com/review/t5-small-summary/* /tmp/t5-small-summary

# COMMAND ----------

mlflow.set_experiment("/Users/sean.owen@databricks.com/fine-tuning-t5")
last_run_id = mlflow.search_runs(filter_string="tags.mlflow.runName	= 't5-small-fine-tune-reviews'")['run_id'].item()

with mlflow.start_run(run_id=last_run_id):
  mlflow.pyfunc.log_model(artifacts={"pipeline": "/tmp/t5-small-summary"}, 
    artifact_path="review_summarizer", 
    python_model=ReviewModel(),
    registered_model_name="sean_t5_small_fine_tune_reviews")

# COMMAND ----------

# MAGIC %md
# MAGIC This model can then be deployed as a real-time endpoint! Check the `Models` and `Endpoints` tabs to the left in Databricks.
# MAGIC 
# MAGIC What would the latency be like for such a model? if latency is important, then one might serve the model using GPUs (coming soon to Databricks Model Serving). Test latency on a single input, and run this on a GPU cluster:

# COMMAND ----------

sample_review = "summarize: " + review_by_product_df.select("reviews").head(1)[0]["reviews"]

summarizer_pipeline = pipeline("summarization",\
  model="/dbfs/tmp/sean.owen@databricks.com/review/t5-small-summary",\
  tokenizer="/dbfs/tmp/sean.owen@databricks.com/review/t5-small-summary",\
  num_beams=10, min_new_tokens=50, device="cuda:0")

# COMMAND ----------

# MAGIC %time summarizer_pipeline(sample_review, truncation=True)

# COMMAND ----------


