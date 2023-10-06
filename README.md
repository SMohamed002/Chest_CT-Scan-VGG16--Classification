# Chest_CT-Scan-VGG16--Classification
Chest cancer classification using chest CT images using VGG16 model and data augmentation to avoid overfitting
About Dataset
---------------
Data Story
-----------
It was a project about chest cancer detection using machine learning and deep leaning (CNN) .
we classify and diagnose if the patient have cancer or not using AI model .
We give them the information about the type of cancer and the way of treatment.
we tried to collect all data we need to make the model classify the images easily.
so i had to fetch data from many resources to start the project .
I researched a lot to collect all the data from many resources and cleaned it for the CNN .

Data
-----
Images are not in dcm format, the images are in jpg or png to fit the model
Data contain 3 chest cancer types which are Adenocarcinoma,Large cell carcinoma, Squamous cell carcinoma , and 1 folder for the normal cell
Data folder is the main folder that contain all the step folders
inside Data folder are test , train , valid

test represent testing set
train represent training set
valid represent validation set
training set is 70%
testing set is 20%
validation set is 10%

Adenocarcinoma
---------------
Adenocarcinoma of the lung: Lung adenocarcinoma is the most common form of lung cancer
accounting for 30 percent of all cases overall and about 40 percent
of all non-small cell lung cancer occurrences. Adenocarcinomas are
found in several common cancers, including breast, prostate and colorectal.
Adenocarcinomas of the lung are found in the outer region of the lung
in glands that secrete mucus and help us breathe.
Symptoms include coughing, hoarseness, weight loss and weakness.

Large cell carcinoma
---------------------
Large-cell undifferentiated carcinoma: Large-cell undifferentiated carcinoma lung cancer grows and spreads quickly and can
be found anywhere in the lung. This type of lung cancer usually accounts for 10
to 15 percent of all cases of NSCLC.
Large-cell undifferentiated carcinoma tends to grow and spread quickly.

Squamous cell carcinoma
------------------------
Squamous cell: This type of lung cancer is found centrally in the lung,
where the larger bronchi join the trachea to the lung,
or in one of the main airway branches.
Squamous cell lung cancer is responsible for about 30 percent of all non-small
cell lung cancers, and is generally linked to smoking.

And the last folder is the normal CT-Scan images




About code
-----------

Load images and visualization
------------------------------

All the modules needed for this project were imported. Then, all the images used to train, validate, and test the model were uploaded to Google Drive. A connection was then established between Google Colab and Google Drive. The image files were then accessed and assigned to variables. The images were then called and displayed using imread in the matplotlib module. The images were displayed in multiple medical image-specific display types.


![Screenshot_31](https://github.com/SMohamed002/Chest_CT-Scan-VGG16--Classification/assets/130864211/60fdbd1b-58f6-4dc5-96eb-dc77f5f5c0b4)
![Screenshot_3](https://github.com/SMohamed002/Chest_CT-Scan-VGG16--Classification/assets/130864211/7e769d6e-3de4-4c5a-9b69-b34f6dfc82dd)
![Screenshot_2](https://github.com/SMohamed002/Chest_CT-Scan-VGG16--Classification/assets/130864211/5441bcfb-5e29-4dbc-aaad-5b4b536432d8)
![Screenshot_1](https://github.com/SMohamed002/Chest_CT-Scan-VGG16--Classification/assets/130864211/ab8aaad1-8b47-4141-9af2-c8587c09b8d0)

A function was created to count and display the number of images for each category in the training, validation, and test sets.the data augmentation process was used to increase the number of training images and try to avoid overfitting.

Build the Model
----------------

The VGG16 model was loaded for use in classification and trained on our data.

VGG16 is a convolutional neural network (CNN) that was developed by Karen Simonyan and Andrew Zisserman at the University of Oxford in 2014. It is a powerful model that has been shown to be effective for image classification.

Architectur
------------

VGG16 consists of 16 layers, 13 of which are convolutional layers. Each convolutional layer consists of a neural network with 3x3 filters. Multiple convolutional layers are connected to create a deep path.

Function
---------

VGG16 is used for image classification. The model is trained on a dataset of images that are labeled with their corresponding categories. After training, the model can classify new images into appropriate categories.

Training Data
--------------

VGG16 was trained on the ImageNet dataset, which contains over 1.2 million images from 1,000 different categories. These images were split into a training set and a test set. The training set was used to train the model, while the test set was used to evaluate the model's performance.

Image Types
------------

VGG16 was trained on the ImageNet dataset, which contains a wide variety of images, including natural images and synthetic images. The dataset includes images of people, animals, and objects.

Additional Information
-----------------------

VGG16 is one of the most popular deep learning models. It is a powerful model that can be used for image classification in a variety of tasks.

VGG16 achieves high accuracy rates in many classification tasks.
VGG16 can be used to create image features that can be used in other tasks, such as face recognition and object detection.
VGG16 has been used in a variety of applications, including the development of smartphone cameras and industrial vision systems.

Conclusion
-----------

VGG16 is a powerful deep learning model that can be used for image classification. It is a popular model that is used in a variety of applications.

![Screenshot_4](https://github.com/SMohamed002/Chest_CT-Scan-VGG16--Classification/assets/130864211/f3ad0ec4-5e58-4481-93fe-3acc478c2c2b)


Optimizer
----------

Adam optimizer is a popular optimization algorithm used in machine learning. It was developed by Diederik P. Kingma and Jimmy Lei Ba in 2014.

Name

The name Adam is derived from "adaptive moment estimation". "Adaptive" refers to the fact that Adam adjusts the learning rates for each parameter individually. "Moment estimation" refers to the fact that Adam uses estimates of the first and second moments of the gradients to update the parameters.

Basic idea

The basic idea of Adam is to combine the advantages of two other optimization algorithms: Momentum and RMSProp.

Momentum is an algorithm that relies on the idea that the current direction of improvement is important. This algorithm maintains the current learning rate and uses it to update the parameters.

RMSProp is an algorithm that relies on the idea that the learning rates should be small when the gradients are small. This algorithm maintains estimates of the second moments of the gradients and uses them to adjust the learning rates.

How Adam works

Adam works by computing estimates of the first and second moments of the gradients. The estimates of the first moment are used to determine the current direction of improvement. The estimates of the second moment are used to determine how much change should be made to the parameters.

Adam computes the estimates of the first and second moments using the following equations:

m_t = β1 * m_(t-1) + (1 - β1) * g_t
v_t = β2 * v_(t-1) + (1 - β2) * g_t^2
where:

mt
​is the estimate of the first moment of the gradients at time t.
vt
is the estimate of the second moment of the gradients at time t.
gt
​is the gradients at time t.
β1 and β2
​are hyperparameters that control the amount of momentum and RMSProp used.
The parameters are updated using the following equation:

w_t = w_(t-1) - α * m_t / (sqrt(v_t) + ε)
where:

wt
​is the value of the parameter at time t.
w(t−1)
is the value of the parameter at time t−1.
α 
is the learning rate.
ε 
is a small constant to stabilize the computation.

Advantages of Adam
-------------------

Adam is effective in a variety of tasks.
Adam converges quickly.
Adam requires little tuning.

Disadvantages of Adam
---------------------

Adam may be unstable in some cases.
Adam may be slow in some cases.
Applications of Adam

Adam is used in a variety of tasks in machine learning, including:

Classification
Regression
Image recognition
Speech recognition
Supervised learning
Unsupervised learning

Conclusion

Adam is a powerful and effective optimization algorithm. It is one of the most popular algorithms used in machine learning.


A learning rate of 0.001 was used and 100 epochs were used to train the model.
An accuracy of 92.7% and a loss of 0.37 were obtained.


Model evaluation
-----------------

![2](https://github.com/SMohamed002/Chest_CT-Scan-VGG16--Classification/assets/130864211/e5283996-f17c-460c-8df8-2f9653aa7f1d)
![1](https://github.com/SMohamed002/Chest_CT-Scan-VGG16--Classification/assets/130864211/45deff38-04d6-41ec-958f-c1d6c6ebc4d5)


![4](https://github.com/SMohamed002/Chest_CT-Scan-VGG16--Classification/assets/130864211/38c83777-8e6f-4c8c-929f-5708bd8ebae7)
![3](https://github.com/SMohamed002/Chest_CT-Scan-VGG16--Classification/assets/130864211/518a61ac-82ec-4eb3-96dc-4e344a56e08d)




