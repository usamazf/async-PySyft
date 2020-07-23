#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L I B R A R I E S                                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global parameters to be used through out the program                                 #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
def init( ):
    
    #-------------------------------------------------------------------------------------------#
    #                                                                                           #
    #   Define global parameters related to the training process to be used throughout program  #
    #                                                                                           #
    #-------------------------------------------------------------------------------------------#
    
    # define the model to use for training
    global MODEL
    MODEL = "mnist-small"       # vgg-13, vgg-16, vgg-19, resnet-18, resnet-50, resnet-101, resnet-152
    
    # define the model to use for training
    global MODEL_PARAM_ID
    MODEL_PARAM_ID = "model-params"
    
    # define the dateset to use for model training
    global DATASET
    DATASET = "mnist"           # stl-10, cifar-10, cifar-100, imagenet-1k-64
    
    # define the dateset to use for model training
    global DATASET_ID
    DATASET_ID = "mnist_train"           # stl-10, cifar-10, cifar-100, imagenet-1k-64

    # define the id for the created training plan
    global PLAN_ID
    PLAN_ID = "train_plan_mnist"
    
    # define the batch size you want to use
    global BATCH_SIZE
    BATCH_SIZE = 32 
    
    # max batches to process before carrying out the syncrhonization
    global MAX_NR_BATCHES
    MAX_NR_BATCHES = 50
    
    # define if you want the batches to be sampled randomly
    global RANDOM_SAMPLE_BATCHES
    RANDOM_SAMPLE_BATCHES = False
    
    # define the total number of epochs you want to train
    global NUM_EPOCHS
    NUM_EPOCHS = 50

    # define the initial learning rate to start the training with
    global INITIAL_LR
    INITIAL_LR = 0.01
    
    # define the criterion be used
    global CRITERION
    CRITERION = "CrossEntropyLoss"
    
    # define the optimizer that the workers should use
    global OPTIMIZER
    OPTIMIZER = "SGD"
    
    #-------------------------------------------------------------------------------------------#
    #                                                                                           #
    #   Define process related information to be used by the program.                           #
    #                                                                                           #
    #-------------------------------------------------------------------------------------------#
    
    # define the manual seed for common model initializations.
    global MANUAL_SEED
    MANUAL_SEED = 42        
    
    # other process information about current process.
    #global GLOBAL_RANK
    #GLOBAL_RANK = global_rank
    
    #global LOCAL_RANK
    #LOCAL_RANK = local_rank
    
    #global WORLD_SIZE 
    #WORLD_SIZE = world_size
