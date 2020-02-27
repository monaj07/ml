#### ImageNet Classification
Run `test_imagenet.py` to classify sample images into one of the 1000 ImageNet categories.
MobileNet-v2 model is used for classification.

Put the sample images in `data/randomTest/images` directory.

The model is pre-trained on ImageNet dataset
(which is the case for all experiments in this repo).

#### Transfer Learning for Face/Person Classification
Fine-tune the MobileNet-v2 model on the CaltechFaces dataset.

Run `train_faces.py` and then `test_faces.py` for evaluating
the trained model snapshots on the test split of the dataset.

#### Model Quantization
Run `train_faces_quant.py` to train, quantize and serialize the MobileNet-v2 model.

Test the serialized model by running `test_faces_quant.py`.

#### Model Prunning
Run `prune.py`.
(Description in the file docstring).

