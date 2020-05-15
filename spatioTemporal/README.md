### Multimodal Video Prediction Work using Satellite and CityScape Images

Directory for collected satellite images from Google Maps : `/atlas/u/buzkent/CityScapes/GoogleMaps/`

Directory for the CityScapes Video Dataset : `/atlas/u/buzkent/CityScapes/`
- It includes 30 frames from each sequence.
- It includes ground truth segmentation map for the 20th frame of each sequence.

Train with ```python train.py --train_dir video_train_filelist.csv --test_dir video_val_filelist.csv```

