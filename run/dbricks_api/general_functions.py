# Databricks notebook source
# DBTITLE 1,Create Databricks File System Folder
def create_dbfs_folder(folderpath = None):
  """create databricks file system folder"""
  try:
    result = dbutils.fs.mkdirs(folderpath)
    return f"{folderpath} created successfully...."
  except: return f"{folderpath} could not be created...."

# COMMAND ----------

# DBTITLE 1,Delete Databricks File System Folder
def delete_dbfs_folder(folderpath = None):
  """delete databricks file system folder"""
  try:
    result = dbutils.fs.rm(folderpath)
    return f"{folderpath} removed successfully...."
  except: return f"{folderpath} could not be removed...."

# COMMAND ----------

# DBTITLE 1,Get Databricks File System File Name
def get_dbfs_file_name(dbfsfilepath = None, file_ext = None):
  """get a dbfs file name"""
  files = dbutils.fs.ls(dbfsfilepath)
  for file in files:
    if file_ext in file[1]: return file[1]
  else: return None

# COMMAND ----------

# DBTITLE 1,Remove a Substring From a String Input
def check_str_for_substr_and_replace(inputstr = None, substr = None):
    """remove a substring from a string input"""
    if substr in inputstr:
        return inputstr.replace(substr, '')
    else: return inputstr
