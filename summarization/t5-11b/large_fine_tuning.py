# Databricks notebook source
# MAGIC %md
# MAGIC ## Fine-Tuning with t5-large
# MAGIC 
# MAGIC This proceeds just like the tuning of `t5-small`, but requires a larger instance type such as the `g5.12xlarge` on AWS (4 x A10 GPUs), and a smaller batch size per device.

# COMMAND ----------

# MAGIC %pip install 'transformers>=4.26.0' datasets evaluate rouge-score

# COMMAND ----------

import os

os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ['DATABRICKS_HOST'] = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['TRANSFORMERS_CACHE'] = "/dbfs/tmp/sean.owen@databricks.com/cache/hf"
os.environ['MLFLOW_EXPERIMENT_NAME'] = "/Users/sean.owen@databricks.com/fine-tuning-t5"
os.environ['MLFLOW_FLATTEN_PARAMS'] = "true"

# COMMAND ----------

# MAGIC %sh export DATABRICKS_TOKEN && export DATABRICKS_HOST && export MLFLOW_EXPERIMENT_NAME && export MLFLOW_FLATTEN_PARAMS && python \
# MAGIC     /Workspace/Repos/sean.owen@databricks.com/summarization/run_summarization.py \
# MAGIC     --model_name_or_path t5-large \
# MAGIC     --do_train \
# MAGIC     --do_eval \
# MAGIC     --train_file /dbfs/tmp/sean.owen@databricks.com/review/camera_reviews_train.csv \
# MAGIC     --validation_file /dbfs/tmp/sean.owen@databricks.com/review/camera_reviews_val.csv \
# MAGIC     --source_prefix "summarize: " \
# MAGIC     --output_dir /dbfs/tmp/sean.owen@databricks.com/review/t5-large-summary \
# MAGIC     --optim adafactor \
# MAGIC     --num_train_epochs 4 \
# MAGIC     --bf16 \
# MAGIC     --per_device_train_batch_size 12 \
# MAGIC     --per_device_eval_batch_size 12 \
# MAGIC     --predict_with_generate \
# MAGIC     --run_name "t5-large-fine-tune-reviews"

# COMMAND ----------


