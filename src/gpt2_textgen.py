import gpt_2_simple as gpt2
import os
import csv

model_name = "355M"
if not os.path.isdir(os.path.join("models", model_name)):
	print(f"Downloading {model_name} model...")
	gpt2.download_gpt2(model_name = model_name)

file_name = "finneganswake.txt"

sess = gpt2.start_tf_sess()

# use this when loading a trained model
#gpt2.load_gpt2(sess)

gpt2.finetune(sess,
			  file_name,
			  model_name = model_name,
			  steps = 1000)

texts = gpt2.generate(sess, return_as_list=True)
print(texts[0])

with open("fw_tweets.csv", "w") as f:

	writer = csv.writer(f)

	for text in texts:
		writer.writerow([text])