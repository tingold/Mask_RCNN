source activate tensorflow_p36
export PYTHONPATH=$PYTHONPATH:.
python samples/buidings/buildings.py && tensorboard --logdir logs