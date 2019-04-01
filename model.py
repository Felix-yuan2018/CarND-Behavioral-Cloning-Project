
import csv
import cv2
import numpy as np
from keras.models import Sequential
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from tqdm import tqdm



###############################################################################
##############LOAD DATA AND SPLIT THE DATA TO TRAIN AND VALIDA SAMPLE#############
###############################################################################
def load_log_file(file_path):
    '''
    read the log file in the path folder
    return a list contain the log information
    '''
    lines = []
    with open(file_path+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) #skip the headers
        for line in reader:
            lines.append(line) # line content: center/left/right/steering/throttle/brake/speed
        #lines = lines[1:]
    return lines

def load_images(images, angles,file_path,sample_balance=False):
    '''
    append the center image to the list 'images'.
    append the angle data to list 'angles'
    path: the data folder
    sample_balance: False/0 or int (1, 101), balance the steering data 
    '''
    print("Loading Centeral image from: ", file_path)
    lines = load_log_file(file_path)
    # check the split
    if ('/' in lines[0][0]):
        split = '/'
    else:
        split = '\\'
    num_before = len(images)

    if sample_balance:
        print("Sample balance, {} percent steering<0.02 image will be keeped.".format(sample_balance))

    for indx, line in enumerate(lines):
        
        image_path = file_path +'IMG/'+ line[0].split(split)[-1]
        image= process_image(image_path)
    
        
        angle = float(line[3])
        # check the angle and decide if need append the data to the list, if zero, skip
        if sample_balance:
            if abs(angle) < 0.02 and np.random.randint(0, 101) < 100 - sample_balance: #and np.random.randint(0,10) < 10: # controal the percent, 10 100% get rid of
           
                continue

        images.append(image)
        angles.append(angle)
    
    num_after = len(images)
    print("Actual loading {} images.".format(num_after-num_before))
    print()

def load_sides_images(images, angles, file_path,  correction, sub_read=False, sample_balance=False):
    '''
    append the left/right image to the list 'images'.
    append the offseted steering data to list 'angles'
    file_path: the data folder
    correction: offset the steering of light/right images
    sample_balance: False or int (0, 101), balance the steering data 
                    
    '''
    
    print("Loading Left/Right image from: ", file_path)
    print("Left image will offset: {}, Right images will offset: {}".format(-correction,correction))
    
    lines = load_log_file(file_path)    
    # check the split
    if ('/' in lines[0][0]):
        split = '/'
    else:
        split = '\\'
        
    num_before = len(images)

    if sample_balance:
        print("Sample balance, {} percent steering<0.02 image will be keeped.".format(sample_balance))    

    
    for indx, line in tqdm(enumerate(lines)):
        
        #left_image_path = file_path + line[1].strip()
        #right_image_path = file_path + line[2].strip()
        left_image_path = file_path + 'IMG/' + line[1].split(split)[-1]
        right_image_path = file_path + 'IMG/' + line[2].split(split)[-1]

        image_left = process_image( left_image_path)  

        image_right = process_image( right_image_path)    
     

        angle = float(line[3])
       
        if sample_balance:
            if abs(angle + correction) < 0.02 and np.random.randint(0, 101) < 100 - sample_balance:
           
                continue
        images.append(image_left)
        angles.append(angle + correction)

        if sample_balance:
            if abs(angle - correction) < 0.02 and np.random.randint(0, 101) < 100 - sample_balance:
            
                continue
        images.append(image_right)
        angles.append(angle - correction)
    
    num_after = len(images)
    print("Actual loading {} images.".format(num_after-num_before))
    print()
       


###############################################################################
#############DATA PREPROCESS BY CROP/RESIZE/COLOR CHANNEL TRANSFORM#############
###############################################################################

def process_image(image_file):
    """
    read the image from image_file and preprocess the image
    """
    #image = cv2.imread(image_file)
    image = mpimg.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (200,100)) # (160,320,3)-->(100,200,3)
    image = image[25:91,:,:] # (100,200,3)-->(66,200,3)

    return image   



###############################################################################
#################DATA AUGUMENTED BY RANDOM SHEAR/BRIGHTNESS/FLIP###############
###############################################################################
def image_rotate(image, angle):
    """
    rotate the image 
    """
    image_shape = np.array(image).shape
    center = (image_shape[1]//2, image_shape[0]//2)
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    image = cv2.warpAffine(image, M, (image_shape[1], image_shape[0]))
    return image

def image_shift(image, dx, dy):
    '''
    shift the image dx and dy
    '''
    image_shape = np.array(image).shape
    M = np.float32([[1,0,dx], [0,1,dy]])
    image = cv2.warpAffine(image, M, (image_shape[1], image_shape[0]))
    return image

def images_augmentation(images, angles, factor=0, rotate=15, shift=0.2):
    """
    rotate and shift the images randomly
    factor: the muli number of augmentation numbers 1.0 = 1.0x, 2.0=2.0x
    rotate: rotate angle range >0
    shift: shift range, muli the width and height
    """
    print("Begin augemenation the images {factor}X")
    image_shape = images[0].shape
    num_images = len(images)
    num_aug = int(num_images * factor)

    # (160, 320, 3)
    width = int(image_shape[1] * shift)
    height = int(image_shape[0] * shift)

    for indx in tqdm(range(num_aug)):
        if indx % 2 == 0: # rotate
            rot_angle = np.random.randint(-rotate, rotate+1)
            image = image_rotate(images[indx%num_images], rot_angle)
            images.append(image)
            angles.append(angles[indx%num_images])
        else:
            dx = np.random.randint(-width, width+1)
            dy = np.random.randint(-height, height+1)
            image = image_shift(images[indx%num_images], dx, dy)
            images.append(image)
            angles.append(angles[indx%num_images])
            
    print("Data augmentation Done!!!") 
       

    


###############################################################################
#################################DNN NETWORK##################################
###############################################################################
def model_nvidia_modifed(X_train, y_train,epoch):
    # compile and train the model using the generator function
    # Model based on  end-to-end architecture
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66,200,3))) 

    # Conv1 (66, 200, 3) --> (31, 98, 24)
    model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))

    # Conv2 (31, 98, 24) --> (14, 47, 36)
    model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
 
    # Conv3 (14, 47, 36) --> (48, 5, 22)
    model.add(Conv2D(48, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
    
    # Conv4 (48, 5, 22) --> (64, 3, 20)
    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
  
    # Conv5, (64, 3, 20) --> (64, 1, 18)
    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))   
    # FC1
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(0.45))
    # FC2
    model.add(Dense(100))
    model.add(Dropout(0.45))
    # FC3
    model.add(Dense(50))
    model.add(Dropout(0.45))
    #FC4
    model.add(Dense(10))
    model.add(Dropout(0.45))
    # Out put
    model.add(Dense(1))
    model.summary()


    # Compile model with adam optimizer,learning rate = 0.0001
    model.compile(loss='mse', optimizer='adam')


    history_object =model.fit(X_train, y_train, batch_size=256, validation_split=0.2, shuffle=True, epochs=epoch)
    
 



    # plot the training and validation loss for each epoch
    plt.figure()

    plt.plot(history_object.history['loss'])

    plt.plot(history_object.history['val_loss'])

    plt.title('model mean squared error loss')

    plt.ylabel('mean squared error loss')

    plt.xlabel('epoch')

    plt.legend(["training set", "validation_set"], loc="upper right")

    plt.savefig('history_track2.png')

    model.save('model_track2.h5')
    print('Model saved!!!')



###############################################################################
##############################TRAIN  PROCEDURE##############################
###############################################################################    
def train_model(file_path,balanced_par,angle_correction,epoch):
    
    
    # cread the list to hold the features and labels
    images = []
    angles = []
     # read data
    for path in file_path:
        load_images(images, angles,path,sample_balance=balanced_par)
        load_sides_images(images, angles, path,sample_balance=balanced_par, correction = angle_correction)
    images_augmentation(images, angles, factor=0, rotate=15, shift=0.2)
    # Generate the training data
    print("Generate X_train, y_train array from loaded images")
    X_train = np.array(images)
    y_train = np.array(angles)
    
     # print the samples information
    print('Sample number: ', len(X_train))
    print("Sample shape: ", X_train.shape)
    print("Image shape: ", X_train[0].shape)
    print('Sample label number: ', len(y_train))
    print()
    
    model_nvidia_modifed(X_train,y_train,epoch)


if __name__ == '__main__':
    train_model(file_path=('./data/','./data_2/'),balanced_par=4,angle_correction=0.21,epoch=10)
    

