# rgbd-ssd
The source code for my research 'Multi Modal Single Shot MultiBox Detector for RGB-D Object Detection(tentative name)' supervised by Professor Sakurai in Keio University.

## SSD Testing
I used `VOCdevkit dataset` which can be downloaded from following commands in order to test the SSD model.

```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```

## NYU Data Conversion 
```
python convert_nyu_from_mat_to_pkl.py --dataset_path ../dataset/ --mat_path ../dataset/SUNRGBDMeta2DBB_v2.mat
```
