#this package runs on gcloud the main module is trainer/kerascNN 
#install gcloud first
#then create a bucket and upload the train_x.csv train_y.csv
# note --config=trainer/cloudml-gpu.yaml  this line will get you a GPU on 
open terminal
cd to this directory and do 
export JOB_DIR=gs://your/path/to/google/cloud/bucket
#you can change those to yours
export JOB_NAME=kcNN1
export REGION=us-east1



gcloud ml-engine jobs submit training $JOB_NAME --job-dir $JOB_DIR --runtime-version 1.0 
--module-name trainer.kerascNN --package-path ./trainer --region $REGION 
--config=trainer/cloudml-gpu.yaml -- 
--x_file gs://your/path/to/google/cloud/bucket/train_x.csv 
--y_file gs://your/path/to/google/cloud/bucket/train_y.csv

#after the training finished, download the saved model and put it in predict folder and run prediction
#adjust model naming accordingly
