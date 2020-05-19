# Action Recognition 
This repo will host a collection of code examples and resources for using various methods to recognize human actions in videos.

## I3D (Inflated 3D ConvNet)
[I3D architecture](https://arxiv.org/pdf/1705.07750) was released by the researchers at [DeepMind](https://deepmind.com/research/publications/quo-vadis-action-recognition-new-model-and-kinetics-dataset) for video classification. They have also released the [Kinetics](https://deepmind.com/research/open-source/kinetics) dataset which contains 600 different human actions with at least 600 video clips for each class.

To use i3d trained on Kinetics-600 dataset, clone this repo and follow the below commands.
```
git clone https://github.com/arunponnusamy/action-recognition.git
cd action-recognition/i3d/
python3 i3d_kinetics_600_tf_hub.py --label label_map_600.txt --video v_CricketBowling_g05_c02.avi 
```

You should get the below result or similar
```
playing cricket 0.8835184
brushing teeth 0.020662978
hammer throw 0.009957399
hopscotch 0.009599756
robot dancing 0.0078740865
```

You can also try with a different video clip containing any activity from the 600 classes listed in the [label map](i3d/label_map_600.txt).

### Dependencies
- tensorflow 2.x
- tensorflow_hub
- opencv-python
- numpy
