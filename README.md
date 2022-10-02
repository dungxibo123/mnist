usage: main.py [-h] [-b BATCH_SIZE] [-e EPOCH] [--weight-decay WEIGHT_DECAY]
               [--optimizer {sgd,adam,adamw}] [-lr LEARNING_RATE]
               [-vb VAL_BATCH_SIZE] [--use-gpu] [--test-size TEST_SIZE]
               [--workers WORKERS] [--channels [CHANNELS ...]] [--augment]
               [--dropout DROPOUT] [--batch-norm] [--kernel-size KERNEL_SIZE]
               [--maxpool-kernel-size MAXPOOL_KERNEL_SIZE]
               [--dense-size DENSE_SIZE] [--num-classes NUM_CLASSES]
               [--checkpoint] [--checkpoint-path CHECKPOINT_PATH] --data
               {MNIST,FMNIST} --data-path DATA_PATH

options:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
  -e EPOCH, --epoch EPOCH
  --weight-decay WEIGHT_DECAY
  --optimizer {sgd,adam,adamw}
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
  -vb VAL_BATCH_SIZE, --val-batch-size VAL_BATCH_SIZE
  --use-gpu
  --test-size TEST_SIZE
  --workers WORKERS
  --channels [CHANNELS ...]
  --augment
  --dropout DROPOUT
  --batch-norm
  --kernel-size KERNEL_SIZE
  --maxpool-kernel-size MAXPOOL_KERNEL_SIZE
  --dense-size DENSE_SIZE
  --num-classes NUM_CLASSES
  --checkpoint
  --checkpoint-path CHECKPOINT_PATH
  --data {MNIST,FMNIST}
  --data-path DATA_PATH
