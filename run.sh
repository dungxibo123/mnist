#/bin/zsh
MLFLOW_TRACKING_URI=http://localhost:5000/ MLFLOW_EXPERIMENT_NAME="MNIST_TESTING" python main.py --data MNIST\
							 --data-path .\
							 --epoch 100\
               --optimizer adam\
							 --test-size 0.999\
							 --use-gpu



#echo $MLFLOW_EXPERIMENT_NAME


