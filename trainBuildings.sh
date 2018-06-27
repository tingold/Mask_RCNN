source activate tensorflow_p36
export PYTHONPATH=$PYTHONPATH:.
python samples/buildings/buildings.py && tensorboard --logdir logs &