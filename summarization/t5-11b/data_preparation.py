# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-Tuning Billion-Parameter Models with Hugging Face and DeepSpeed
# MAGIC 
# MAGIC These notebooks accompany the blog of the same name, with more complete listings and basic commentary about the steps. The blog gives fuller context about what is happening.
# MAGIC 
# MAGIC **Note:** Throughout these examples, various temp paths are used to store results, under `/dbfs/tmp/`. Change them to whatever location you desire.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation
# MAGIC 
# MAGIC This example uses data from the [Amazon Customer Review Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html), or rather just the camera product reviews, as a stand-in for "your" e-commerce site's camera reviews. Simply download it and display:

# COMMAND ----------

pip install -r  /Workspace/Repos/robert.altmiller@databricks.com/dbricks_llms/summarization/requirements.txt

# COMMAND ----------

# library imports
import os

# user defined parameters
externaldatafilename = "amazon_reviews_us_Camera_v1_00.tsv.gz" # external location filename
externaldatafilepath = f"https://s3.amazonaws.com/amazon-reviews-pds/tsv/{externaldatafilename}" # external data location
localdbfsfoldername = "review" # name of dbfs folder where all data will be store in databricks
localdbfscleanedfoldername = "cleaned" # name of dbfs folder where cleaned dataset with be stored in databricks
localtraindatafilename = "camera_reviews_train.csv" # training dataset filename
localvalidatedatafilename = "camera_reviews_val.csv" # validation dataset filename
train_test_split_val = .8 # train, test, split parameters
seed_val = 42 # train, test, split parameters

# dyanmic databricks user name
username = spark.sql('select current_user() as user').collect()[0]['user'] # email address or unique username

# external data paths
print(f"externaldatafilename: {externaldatafilename}")
print(f"externaldatafilepath: {externaldatafilepath}\n")

# local dbfs raw data paths
localdatafolderpath = f"/dbfs/tmp/{username}/{localdbfsfoldername}"
print(f"localdatafolderpath: {localdatafolderpath}")
localdatafilepath = f"{localdatafolderpath}/{externaldatafilename.replace('.gz', '')}"
print(f"localdatafilepath: {localdatafilepath}\n")

# local dbfs cleaned data paths
localdatacleanedfolderpath = f"{localdatafolderpath}/{localdbfscleanedfoldername}"
print(f"localdatacleanedfolderpath: {localdatacleanedfolderpath}\n")

# local dbfs training data paths
print(f"localtraindatafilename: {localtraindatafilename}")
localtraindatafilepath = f"{localdatafolderpath}/{localtraindatafilename}"
print(f"localtraindatafilepath: {localtraindatafilepath}\n")

# local dbfs validation data paths
print(f"localvalidatedatafilename: {localvalidatedatafilename}")
localvalidatedatafilepath = f"{localdatafolderpath}/{localvalidatedatafilename}"
print(f"localvalidatedatafilepath: {localvalidatedatafilepath}\n")

# set environment variables for bash script
os.environ["externaldatafilepath"] = externaldatafilepath
os.environ["localtraindatafilepath"] = localtraindatafilepath
os.environ["localvalidatedatafilepath"] = localvalidatedatafilepath



# COMMAND ----------

# DBTITLE 1,Remove Starting DBFS Directory if it Already Exists
# true removes folder and files recursively and ignores if current data exists
print(f'removed directory "{localdatafolderpath}": {dbutils.fs.rm(localdatafolderpath, True)}')
print(f'created directory "{localdatafolderpath}": {dbutils.fs.mkdirs(localdatafolderpath)}')

# COMMAND ----------

# DBTITLE 1,Download Starting Dataset Using Curl Command
# MAGIC %sh
# MAGIC 
# MAGIC for path in $externaldatafilepath 
# MAGIC   do 
# MAGIC     externalfilepath="$path" 
# MAGIC   done
# MAGIC echo $externaldatafilepath
# MAGIC 
# MAGIC curl -s $externaldatafilepath | gunzip > localdatafilepath

# COMMAND ----------

# DBTITLE 1,Verify Starting Dataset Exists in Databricks File System (DBFS)
dbutils.fs.ls(localdatafolderpath.replace("/dbfs", ""))

# COMMAND ----------

# DBTITLE 1,Read in the Starting Dataset in Spark Dataframe and Print Dataframe Row Count
camera_reviews_df = spark.read.options(delimiter="\t", header=True).\
  csv(localdatafilepath.replace("/dbfs", "dbfs:"))
print(f"total records: {camera_reviews_df.count()}")

# COMMAND ----------

# DBTITLE 1,Clean the Starting Dataset of HTML Tags, Escapes and Other Markdown Using Pandas UDFs
# The data needs a little cleaning because it contains HTML tags, escapes, and other markdown that isn't worth handling further. Simply replace these with spaces in a UDF.
# The functions below also limit the number of tokens in the result, and try to truncate the review to end on a sentence boundary. 
# This makes the resulting review more realistic to learn from; it shouldn't end in the middle of a sentence.
# The result is just saved as a Delta table.

# library imports
import re
from pyspark.sql.functions import udf


# Some simple (simplistic) cleaning: remove tags, escapes, newlines
# Also keep only the first N tokens to avoid problems with long reviews
remove_regex = re.compile(r"(&[#0-9]+;|<[^>]+>|\[\[[^\]]+\]\]|[\r\n]+)")
split_regex = re.compile(r"([?!.]\s+)")


def clean_text(text, max_tokens):
  if not text:
    return ""
  text = remove_regex.sub(" ", text.strip()).strip()
  approx_tokens = 0
  cleaned = ""
  for fragment in split_regex.split(text):
    approx_tokens += len(fragment.split(" "))
    if (approx_tokens > max_tokens):
      break
    cleaned += fragment
  return cleaned.strip()


@udf('string')
def clean_review_udf(review):
  return clean_text(review, 100)


@udf('string')
def clean_summary_udf(summary):
  return clean_text(summary, 20)


# Pick examples that have sufficiently long review and headline
camera_reviews_df.select("product_id", "review_body", "review_headline").\
  sample(0.1, seed=42).\
  withColumn("review_body", clean_review_udf("review_body")).\
  withColumn("review_headline", clean_summary_udf("review_headline")).\
  filter("LENGTH(review_body) > 0 AND LENGTH(review_headline) > 0").\
  write.format("delta").mode("overwrite").save(localdatacleanedfolderpath.replace("/dbfs", ""))

# COMMAND ----------

# DBTITLE 1,Verify Cleaned Dataset Exists in Databricks File System (DBFS)
dbutils.fs.ls(localdatacleanedfolderpath.replace("/dbfs",""))

# COMMAND ----------

# DBTITLE 1,Read the Cleaned Dataset in Spark Dataframe and Display Results
camera_reviews_cleaned_df = spark.read.format("delta").load(localdatacleanedfolderpath.replace("/dbfs", "")).\
  select("review_body", "review_headline").toDF("text", "summary")
display(camera_reviews_cleaned_df.limit(10))

# COMMAND ----------

# DBTITLE 1,Split the Cleaned Dataset into Training and Validation Datasets
# Fine-tuning will need this data as simple csv files. Split the data into train/validation sets and write as CSV for later

train_df, val_df = camera_reviews_cleaned_df.randomSplit([train_test_split_val, 1 - train_test_split_val], seed = seed_val)
train_df.toPandas().to_csv(localtraindatafilepath, index = False)
val_df.toPandas().to_csv(localvalidatedatafilepath, index = False)

# COMMAND ----------

# DBTITLE 1,Verify Training, and Validation Datasets Exists in Databricks File System (DBFS)
dbutils.fs.ls(localdatafolderpath.replace("/dbfs",""))
