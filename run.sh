#/bin/zsh
MLFLOW_TRACKING_URI=http://localhost:8080/ MLFLOW_EXPERIMENT_NAME="MNIST_TESTING" python main.py --data MNIST\
							 --data-path .\
							 --epoch 3\
               --optimizer adam\
							 --test-size 0.95
#							 --use-gpu



#echo $MLFLOW_EXPERIMENT_NAME


