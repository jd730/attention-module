python train_pascal.py \
			--ngpu 1 \
			--workers 0 \
			--arch resnet --depth 50 \
			--epochs 100 \
			--batch-size 32 --lr 0.1 \
			--att-type CBAM \
			--prefix RESNET50_IMAGENET_CBAM \
			./VOCdevkit2007/VOC2007/JPEGImages/

