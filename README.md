# SfMNeXt: The NeXt Series of Learning Structure Prior from Motion

## 👩‍⚖️ Demo

Online demo is available at [HERE](http://cn-nd-plc-1.openfrp.top:56789/)

## 👀 Training

To train on KITTI, run:

```bash
python train.py ./args_files/args_res50_kitti_192x640_train.txt
```
For instructions on downloading the KITTI dataset, see [Monodepth2](https://github.com/nianticlabs/monodepth2)

To train on CityScapes, run:

```bash
python train.py ./args_files/args_cityscapes_train.txt
```
To finetune on CityScapes, run:

```bash
python train.py ./args_files/args_cityscapes_finetune.txt
```

For preparing cityscapes dataset, please refer to SfMLearner's [prepare_train_data.py](https://github.com/tinghuiz/SfMLearner/blob/master/data/prepare_train_data.py) script.
We used the following command:

```bash
python prepare_train_data.py \
    --img_height 512 \
    --img_width 1024 \
    --dataset_dir <path_to_downloaded_cityscapes_data> \
    --dataset_name cityscapes \
    --dump_root <your_preprocessed_cityscapes_path> \
    --seq_length 3 \
    --num_threads 8
```

## 💾 Pretrained weights and evaluation

You can download weights for some pretrained models here:

* [KITTI (640x192)](https://drive.google.com/file/d/1IRo-k56mO0glRuCyHJu2p16RBJGDIx59/view?usp=sharing)
* [KITTI (1024x320)](https://drive.google.com/file/d/1VH9hMN59eIMjVhUwjYOOxwFN1hsKPIp0/view?usp=sharing)
* [CityScapes (512x192)](https://drive.google.com/file/d/1nLwTQnXV_9IURUqfCfoGZyVHb4U5XwYI/view?usp=sharing)

To evaluate a model on KITTI, run:

```bash
python evaluate_depth_config.py args_files/args_kitti_320x1024_evaluate.config
```

Make sure you have first run `export_gt_depth.py` to extract ground truth files.

And to evaluate a model on Cityscapes, run:

```bash
python ./tools/evaluate_depth_cityscapes_config.py args_files/args_res50_cityscapes_finetune_192x640_eval.txt
```

The ground truth depth files can be found at [HERE](https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip),
Download this and unzip into `splits/cityscapes`.

## 🖼 Inference with your own iamges

```bash
python test_simple_SQL_config.py ./args_files/args_test_simple_kitti_320x1024.txt
```
## Future Works

- [x] release code for training in outdoor scenes (KITTI, Cityscapes)
- [x] model release (KITTI, Cityscapes)
- [ ] code for training in indoor scenes (NYU-Depth-v2, MannequinChallenge)
- [ ] code for finetuning self-supervised model using metric depth
- [ ] model release for indoor scenes and metric fine-tuned model

## License

All rights reserved.
Please see the [license file](LICENSE) for terms.
