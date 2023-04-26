# Databricks notebook source
# MAGIC %md
# MAGIC ## Data Preparation
# MAGIC 
# MAGIC This example uses data from the [Amazon Customer Review Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html), or rather just the camera product reviews, as a stand-in for "your" e-commerce site's camera reviews. Simply download it and display:

# COMMAND ----------

# MAGIC %md
# MAGIC # Install Requirements File

# COMMAND ----------

# DBTITLE 1,Install Requirements
pip install -r "/Workspace/Repos/robert.altmiller@databricks.com/dbricks_llms/run/summarization (t5-11b)/requirements/requirements.txt"

# COMMAND ----------

# MAGIC %md
# MAGIC # Databricks 2.0 API Configuration (Cluster) + Library Imports

# COMMAND ----------

# DBTITLE 1,Cluster and Libraries
# MAGIC %run ../../dbricks_api/cluster_base"

# COMMAND ----------

# MAGIC %md
# MAGIC # User Defined Parameters

# COMMAND ----------

# DBTITLE 1,Parameters
# external data paths
print(f"externaldatafilename: {externaldatafilename}")
print(f"externaldatafilepath: {externaldatafilepath}\n")

# local dbfs raw data paths
localdatafolderpath = f"{localdbfsbasepath}/{username1}/{localdbfsfoldername}"
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
os.environ["localdatafilepath"] = localdatafilepath
os.environ["externaldatafilepath"] = externaldatafilepath
os.environ["localtraindatafilepath"] = localtraindatafilepath
os.environ["localvalidatedatafilepath"] = localvalidatedatafilepath

# COMMAND ----------

# MAGIC %md
# MAGIC # Remove Starting DBFS Directory if it Already Exists

# COMMAND ----------

# DBTITLE 1,Cleanup and Recreate Directories
# true removes folder and files recursively and ignores if current data exists
print("Remove Starting DBFS Directory if it Already Exists.....")
print(f'removed directory "{localdatafolderpath}": {dbutils.fs.rm(localdatafolderpath.replace("/dbfs", ""), True)}')
print(f'created directory "{localdatafolderpath}": {dbutils.fs.mkdirs(localdatafolderpath.replace("/dbfs", ""))}')

# COMMAND ----------

# MAGIC %md
# MAGIC # Download Starting Dataset Using Curl Command

# COMMAND ----------

# DBTITLE 1,Download Starting Dataset
# MAGIC %sh
# MAGIC 
# MAGIC for path in $externaldatafilepath 
# MAGIC   do 
# MAGIC     externalfilepath="$path" 
# MAGIC   done
# MAGIC 
# MAGIC for path in $localdatafilepath 
# MAGIC   do 
# MAGIC     localdatafilepath="$path" 
# MAGIC   done
# MAGIC echo downloading $externaldatafilepath Using Curl Command.....
# MAGIC 
# MAGIC curl -s $externalfilepath | gunzip > $localdatafilepath
# MAGIC 
# MAGIC echo $externaldatafilepath downloaded successfully.....

# COMMAND ----------

# MAGIC %md
# MAGIC # Verify Starting Dataset Exists in Databricks File System (DBFS)

# COMMAND ----------

# DBTITLE 1,Verify Starting Dataset Exists
print("Verify Starting Dataset Exists in Databricks File System (DBFS).....")
dbutils.fs.ls(localdatafolderpath.replace("/dbfs", ""))

# COMMAND ----------

# MAGIC %md
# MAGIC # Read in the Starting Dataset in Spark Dataframe and Display Results

# COMMAND ----------

# DBTITLE 1,Read in the Starting Dataset in Spark Dataframe
print("Read in the Starting Dataset in Spark Dataframe and Print Dataframe Row Count.....")
camera_reviews_df = spark.read.options(delimiter="\t", header=True).\
  csv(localdatafilepath.replace("/dbfs", "dbfs:"))
print(f"total records: {camera_reviews_df.count()}")
display(camera_reviews_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Clean the Starting Dataset of HTML Tags, Escapes and Other Markdown and Display Results

# COMMAND ----------

# DBTITLE 1,Clean Starting Dataset
# The data needs a little cleaning because it contains HTML tags, escapes, and other markdown that isn't worth handling further. Simply replace these with spaces in a UDF.
# The functions below also limit the number of tokens in the result, and try to truncate the review to end on a sentence boundary. 
# This makes the resulting review more realistic to learn from; it shouldn't end in the middle of a sentence.
# The result is just saved as a Delta table.


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


# Pick examples that have sufficiently long review and headline (e.g. sample)
camera_reviews_cleaned_df = camera_reviews_df \
  .select("product_id", "review_headline", "review_body",) \
  .sample(fraction = camera_reviews_sample_percentage, seed = camera_reviews_sample_seed) \
  .withColumn("review_headline", clean_summary_udf("review_headline")) \
  .withColumn("review_body", clean_review_udf("review_body")) \
  .filter("LENGTH(review_body) > 0 AND LENGTH(review_headline) > 0")
  

# write out the cleaned reviews to DBFS  as a Delta table
camera_reviews_cleaned_df.write.format("delta").mode("overwrite").save(localdatacleanedfolderpath.replace("/dbfs", ""))


print(f"Starting Dataset Has Been Cleaned and Written to DBFS: {localdatacleanedfolderpath}.....")
print(f"total records: {camera_reviews_cleaned_df.count()}")
display(camera_reviews_cleaned_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Verify Cleaned Dataset Exists in Databricks File System (DBFS)

# COMMAND ----------

# DBTITLE 1,Verify Cleaned Dataset Exists
print("Verify Cleaned Dataset Exists in Databricks File System (DBFS).....")
dbutils.fs.ls(localdatacleanedfolderpath.replace("/dbfs",""))

# COMMAND ----------

# MAGIC %md
# MAGIC # Read the Cleaned Dataset in Spark Dataframe and Display Results

# COMMAND ----------

# DBTITLE 1,Read the Cleaned Dataset in Spark DF
print("Read the Cleaned Dataset in Spark Dataframe and Display Results.....")
camera_reviews_cleaned_df = spark.read.format("delta").load(localdatacleanedfolderpath.replace("/dbfs", "")).\
  select("review_body", "review_headline").toDF("text", "summary")
print(f"total records: {camera_reviews_cleaned_df.count()}")
display(camera_reviews_cleaned_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC # Split the Cleaned Dataset into Training and Validation Datasets

# COMMAND ----------

# DBTITLE 1,Create Training and Validation Datasets
# Fine-tuning will need this data as simple csv files. Split the data into train/validation sets and write as CSV for later

train_df, val_df = camera_reviews_cleaned_df.randomSplit([train_test_split_val, 1 - train_test_split_val], seed = seed_val)
train_df.toPandas().to_csv(localtraindatafilepath, index = False)
val_df.toPandas().to_csv(localvalidatedatafilepath, index = False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Verify Training, and Validation Datasets Exists in Databricks File System (DBFS)

# COMMAND ----------

# DBTITLE 1,Verify Training and Validation Datasets Exist
print("Verify Training, and Validation Datasets Exists in Databricks File System (DBFS).....")
dbutils.fs.ls(localdatafolderpath.replace("/dbfs",""))
