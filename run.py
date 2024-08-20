#classifier train terminal 1
TRAIN_FLAGS="--iterations 150000 --image_size 80 --anneal_lr True --batch_size 32 --lr 3e-3 --save_interval 5000 --weight_decay 0.01 --data_dir /data/yjpak/Dataset/audio_mnist/ --doc model_analysis"
CLASSIFIER_FLAGS="--image_size 80"
mpiexec -n 2 python classifier_train.py $TRAIN_FLAGS $CLASSIFIER_FLAGS

# --weight_decay 0.05
# --classifier_attention_resolutions 32,16,8

#classifier train CNN
TRAIN_FLAGS="--iterations 150000 --image_size 80 --anneal_lr True --batch_size 32 --lr 3e-3 --save_interval 5000 --weight_decay 0.01 --data_dir /data/yjpak/Dataset/audio_mnist/ --doc CNN_classifier02"
CLASSIFIER_FLAGS="--image_size 80"
mpiexec -n 2 python model_analysis.py $TRAIN_FLAGS $CLASSIFIER_FLAGS



#diffusion train terminal2
MODEL_FLAGS="--image_size 80 --class_cond True --learn_sigma False --weight_decay 0.05 --num_head_channels 64 --data_dir /data/yjpak/Dataset/audio_mnist/ --save_interval 5000" 
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 3e-4 --batch_size 64 --doc diffusion01"
mpiexec -n 2 python image_train.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS



#CNN_classifier smaple 
MODEL_FLAGS="--image_size 80 --class_cond True --learn_sigma False --num_head_channels 64" 
CLASSIFIER_FLAGS="--image_size 80"
SAMPLE_FLAGS="--batch_size 4 --num_samples 50 --timestep_respacing 1000 --use_ddim False --doc sample/CNN/"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
mpiexec -n 2 python CNN_sample.py \
    --model_path /data/yjpak/guided-diffusion/logs/diffusion01/model045000.pt\
    --classifier_path /data/yjpak/guided-diffusion/logs/classifier/CNN_classifier02/classifier_model065000.pt\
    $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS




#diffusion train image terminal2
MODEL_FLAGS="--image_size 32 --class_cond True --learn_sigma False --weight_decay 0.05 --num_head_channels 64 --data_dir /data/yjpak/Dataset/Cifar10/cifar_train/ --save_interval 5000" 
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 2e-5 --batch_size 64 --doc diffusion01"
mpiexec -n 2 python image_train.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS


#image classifier train terminal 1
TRAIN_FLAGS="--iterations 150000 --image_size 32 --anneal_lr True --batch_size 64 --lr 1e-3 --save_interval 3000 --weight_decay 0.05 --data_dir /data/yjpak/Dataset/Cifar10/cifar_train/ --doc image_classifier02"
CLASSIFIER_FLAGS="--image_size 32"
mpiexec -n 2 python classifier_train.py $TRAIN_FLAGS $CLASSIFIER_FLAGS

# mpi 설치
# apt update
# apt apt-get install libopenmpi-dev
# pip install mpi4py-mpich
