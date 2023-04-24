# Databricks notebook source
# MAGIC %md
# MAGIC ## Fine-Tuning with t5-11b and DeepSpeed
# MAGIC 
# MAGIC This proceeds just like the tuning of `t5-large` with DeepSpeed, but requires an even larger instance type such as the `g5.48xlarge` on AWS (8 x A10 GPUs, 768GB of RAM), and a smaller batch size per device again.

# COMMAND ----------

# MAGIC %pip install 'transformers>=4.26.0' datasets evaluate rouge-score git+https://github.com/microsoft/DeepSpeed accelerate 

# COMMAND ----------

import os

os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ['DATABRICKS_HOST'] = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['TRANSFORMERS_CACHE'] = "/dbfs/tmp/sean.owen@databricks.com/cache/hf"
os.environ['MLFLOW_EXPERIMENT_NAME'] = "/Users/sean.owen@databricks.com/fine-tuning-t5"
os.environ['MLFLOW_FLATTEN_PARAMS'] = "true"

# COMMAND ----------

# MAGIC %md
# MAGIC Now, the DeepSpeed configuration enables full ZeRO stage 3 optimization:
# MAGIC 
# MAGIC ```
# MAGIC {
# MAGIC     "fp16": {
# MAGIC         "enabled": false
# MAGIC     },
# MAGIC 
# MAGIC     "bf16": {
# MAGIC         "enabled": true
# MAGIC     },
# MAGIC     
# MAGIC     "scheduler": {
# MAGIC         "type": "WarmupLR",
# MAGIC         "params": {
# MAGIC             "warmup_min_lr": "auto",
# MAGIC             "warmup_max_lr": "auto",
# MAGIC             "warmup_num_steps": "auto"
# MAGIC         }
# MAGIC     },
# MAGIC 
# MAGIC     "zero_optimization": {
# MAGIC         "stage": 3,
# MAGIC         "offload_optimizer": {
# MAGIC             "device": "cpu",
# MAGIC             "pin_memory": true
# MAGIC         },
# MAGIC         "offload_param": {
# MAGIC             "device": "cpu",
# MAGIC             "pin_memory": true
# MAGIC         },
# MAGIC         "overlap_comm": true,
# MAGIC         "contiguous_gradients": true,
# MAGIC         "sub_group_size": 1e9,
# MAGIC         "reduce_bucket_size": "auto",
# MAGIC         "stage3_prefetch_bucket_size": "auto",
# MAGIC         "stage3_param_persistence_threshold": "auto",
# MAGIC         "stage3_max_live_parameters": 1e9,
# MAGIC         "stage3_max_reuse_distance": 1e9,
# MAGIC         "stage3_gather_16bit_weights_on_model_save": true
# MAGIC     },
# MAGIC 
# MAGIC     "gradient_accumulation_steps": "auto",
# MAGIC     "gradient_clipping": "auto",
# MAGIC     "steps_per_print": 1000,
# MAGIC     "train_batch_size": "auto",
# MAGIC     "train_micro_batch_size_per_gpu": "auto",
# MAGIC     "wall_clock_breakdown": false
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %sh export DATABRICKS_TOKEN && export DATABRICKS_HOST && export MLFLOW_EXPERIMENT_NAME && export MLFLOW_FLATTEN_PARAMS && deepspeed \
# MAGIC     /Workspace/Repos/sean.owen@databricks.com/summarization/run_summarization.py \
# MAGIC     --deepspeed /Workspace/Repos/sean.owen@databricks.com/summarization/ds_config_zero3_adafactor.json \
# MAGIC     --model_name_or_path t5-11b \
# MAGIC     --do_train \
# MAGIC     --train_file /dbfs/tmp/sean.owen@databricks.com/review/camera_reviews_train.csv \
# MAGIC     --validation_file /dbfs/tmp/sean.owen@databricks.com/review/camera_reviews_val.csv \
# MAGIC     --source_prefix "summarize: " \
# MAGIC     --output_dir /dbfs/tmp/sean.owen@databricks.com/review/t5-11b-summary \
# MAGIC     --optim adafactor \
# MAGIC     --num_train_epochs 0.01 \
# MAGIC     --gradient_checkpointing \
# MAGIC     --bf16 \
# MAGIC     --per_device_train_batch_size 8 \
# MAGIC     --per_device_eval_batch_size 8 \
# MAGIC     --predict_with_generate \
# MAGIC     --run_name "t5-11b-fine-tune-reviews"

# COMMAND ----------

sample_review = """Nothing was wrong with this item. All its functionalities work perfectly. I recommend this item for anyone that want to take black and white photos. This camera wasn't exactly what I had expected, it was much lighter and seemed a bit flimsy, but it was in very good condition and it arrived very quickly, just as the sender advertised it would. It is a very easy to use camera and I am happy I have it to learn on, but the quality of the first role of film I developed was not great. Overall, I was happy with my purchase and would most definitely buy from this seller again. Great Camera, great condition LIKE NEW. It takes amazing photographs and is easy to handle and work with. The price was very low considering the great quality photos I can take with it. I couldn't be happier and I take it everywhere. The only disadvantage I can find is that the battery that this camera needs to work is very hard to find, and maybe the seller should have specified it with the information regarding the camera. It works with 2 CR2 2 V Batteries. THis camera was everything the shipper promised! works very well, is in amazing condition and is a lot of fun! Practically as good as new! I had a misunderstanding and thought that the camera should come with batteries so the shipper went ahead and shopped them to me even though that wasnt in the description! We received the camera, lens, film, and a camera case - all in excellent condition. Camera works well for the photography student for which it was purchased. Would buy from this site again! Love the shoes. Love it Ok so I didn't read that this was not a digital camera but I am so pleased with this camera. I love it! It came in great condition and it was very easy to use. I would definitely buy it again. Dont Work Very Good Great camera lesson a long time too I sold it recently and got the same value that I paid for I received the camera and it didn't work. I need to find out how to rectify Excellent camera. My daughter need this type of camera for class and has been great! Great camera for a photography class!! I am very happy with my purchase. Delivery was quick. I received the camera a couple days after purchase. I am taking a black and white film photography class and so far am happy with the results my camera gives me. I am no camera expert and for this reason appreciate how easy this camera is to use. I recommend this camera to amateur photographers like myself for its ease of use and and quality for it's price. I bought this camera for my wife for Christmas 04' I thought I ordered it too late for it to arrive on time for Christmas, but these people are great! they got it to me on time and we love this camera we use it all the time - we have several accs. item for it and the quality of the picture are wonderful . Thanks Mark Christian I had been searching Amazon for a camera for about a year and had decided on a camera to get. But then one night I was searching through amazon again and came across the rebel 2000 and bought it that night. The design was my first concern when deciding on what to get, I have bigger hands so little cameras are hard for me to deal with. While this camera is not a HUGE one it is just the right size for me and I love it. I have been extremely impressed with this camera. Stock lens and flash are outstanding. Have done professional work with it. Would recommend it to anyone. Photos are very clear and sharp. Easy to use. When I first bought this camera about a year ago I had no clue what I was doing. But the manual is fairly good at explaining how this camera worked, and you get accustomed to the way it functions fast and easy. After only a few months I was enjoying the creative freedom that you don't get in regular snap shot cameras, and the pictures were great. But now I find myself dreaming about buying a digital EOS. I know very little about taking pitures, granted I understand \\"how\\" to, but doing it is a total different thing. This Camera is VERY good. Manual or Auto, you control all over it. It is idiot proof, or you can take control of every aspect. Like I said, I am new at all of this, but if your a beginner like myself you cann't go wrong with this camera. This was the Camera that was recommended to me over and over again. I received this for Christmas and quickly burned through 4 rolls of film. It is easy to use - the manual and auto features are great for the professional wannabe. I have some trouble getting the pictures to come out natural looking, though. I think it is more a function of my lack of technical photographic knowledge. I am looking forward to the Tiffen polarizing lense I ordered from Amazon to help me in that area. Overall a good camera that can grow with you. I have this kit for over a year and is happy with the results of the few rolls of film I shot. The body is extremely light for a SLR. The major criticism I have about the rebel 2000 body is in its manual mode setting. There's a dial on the right side of the camera top (where your index finger is) for changing f/stops or exposure time. The Canon Rebel 2000 is a superb camera for everybody--whether you are a professional photographer or not. I am an amature photographer, aspriring to be somewhat of a professional and I have taken some excellent pictures with this camera--all of which I intend to include in a portfolio to colleges portraying one of my extracurricular activities. The camera is extremely easy to use and I have not had one problem with it for the year thus far that I have owned it. It and its accessories are worth every penny. This is the best camera I've ever owned. First of all, if you are not a camera buff it is easy to use. If offers both manual and programmed options and even an in between option where the user gives it one setting and it determines the other. Its a fun camera, anyone considering a SLR camera should really consider this one. Just started exploring the camera. I love the light weight and the auto focus/flash that pops up when required. My only disappointment was that when I got the unit I didn't realize there was a slightly different model available that lets you time/date stamp your photos. If I had known about that other model I would have ordered it. Bought this camera after having used my sisters. Can be as simple or complex to use as needed."""

# COMMAND ----------

from transformers import pipeline

summarizer_pipeline = pipeline("summarization",\
  model="/dbfs/tmp/sean.owen@databricks.com/review/t5-11b-summary",\
  tokenizer="/dbfs/tmp/sean.owen@databricks.com/review/t5-11b-summary",\
  num_beams=10, min_new_tokens=50,\
  batch_size=1)

# COMMAND ----------

# MAGIC %time summarizer_pipeline(sample_review, truncation=True)

# COMMAND ----------


