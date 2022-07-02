![Music](https://data.whicdn.com/images/135357007/original.gif)

# _**Identifying_Music_Genre**_
How many times we have wondered, what might have been a songs genre, sometimes it might be easy and sometimes it might make our brains go wild, so this is it, this program allows us to predict the genre of the music using Transfer learning. The flow is as follows first we load the music file, split the songs into chunks, then load the model using load model (), using the loaded model, predict on the music which you ahve loaded earlier and since the output will be in the form of array, we use argmax() to extract the highest probability of the music genre the model has predicted.

# _**Base Paper**_
+ https://www.researchgate.net/publication/324218667_Music_Genre_Classification_using_Machine_Learning_Techniques
+ https://www.researchgate.net/publication/329396097_Music_Genre_Classification_and_Recommendation_by_Using_Machine_Learning_Techniques

# _**Algorithm Description**_
So, the approach we have taken to run/execute this project is by using Transfer Learning, so basically transfer learning is a method where we train our dataset with a model which was already trained on such type of problem that we are working on i.e. Since we are dealing with a classification problem i.e., classifying which genre the music belongs to so we use a specific transfer learning algorithm which is made of classification problem. So, the model we are using in this project is VGG16. VGG16 is basically a 16-layer convolutional neural network which was trained on image net datset which consists of 14 million images, with around 1000 classes. VGG16 is most preferred classification model if we wanted to use transfer learning in our project. This model along with such good accuracy and performance, it also comes with some disadvantages such as exploding gradient descent problem due to more than 100 million parameters for training and due to the size of the model i.e., 528 MB, although it might be a very small size but this will take a lot of time to run and even download in many systems.

![VGG16](https://neurohive.io/wp-content/uploads/2018/11/vgg16-neural-network.jpg)

_**Reference**_
+ http://aishelf.org/vgg-transfer-learning/

# _**How to Execute?**_
So, before execution we have some pre-requisites that we need to download or install i.e., anaconda environment, python and a code editor.
**Anaconda**: Anaconda is like a package of libraries and offers a great deal of information which allows a data engineer to create multiple environments and install required libraries easy and neat.

**Download link:**

![Anaconda](http://img1.wikia.nocookie.net/__cb20100310215806/mafiawars/images/e/e0/Huge_item_anaconda_02.gif)

https://www.anaconda.com/

**Python**: Python is a most popular interpreter programming language, which is used in almost every field. Its syntax is very similar to English language and even children and learning it nowadays, due to its readability and easy syntax and large community of users to help you whenever you face any issues.

**Download link:**

![Python](http://animated.name/uploads/posts/2016-08/1470308776_240.gif)

https://www.python.org/downloads/

**Code editor**: Code editor is like a notepad for a programming language which allows user to write, run and execute program which we have written. Along with these some code editors also allows us to debug, which usually allows users to execute the code line by line and allows them to see where and how to solve the errors. But I personally feel visual code is very good to work with any programming language and makes a great deal of attachment with user.

**Download links:**

![Vs code](https://schwabencode.com/contents/logos/VS2019-Badge.png) ![Pycharm](https://i0.wp.com/scracked.com/wp-content/uploads/2020/01/PyCharm-2019.3.4-Crack.png?fit=200%2C200&ssl=1)

+ https://code.visualstudio.com/Download, 
+ https://www.jetbrains.com/pycharm/download/#section=windows

# _**How to create a new environment and configure jupyter notebook with it.**_
Let us define an environment and why we need different environments. An environment is a collection of libraries that are required to run our project. When we already have an environment with the necessary libraries, why do we need a new environment?
To avoid version mismatches, we create a new environment for each project. For example, in your previous project, you used "tf env" with tensorflow 2.4 and keras 2.4, but in your current project, you must use tensorflow 2.6 and keras 2.6. If you continue your project in the "tf env" environment, there will be a version mismatch and you will need to update tensorflow and keras, but this will cause problems with the previous project's execution. To avoid this, we create a new environment with tensorflow 2.6 and keras 2.6 and resume our project.

Let us now see how to create an environment in anaconda.
+ Type “conda create –n <<name_of_your_env>>”
example: conda create -n env
+ It will ask to proceed with the environment location, type ‘y’ and press enter.
+ When you press ‘y’, the environment will be created. To activate your environment type conda activate <<your_env_name>> . E.g., conda activate myenv.
+ You can see that the environment got changed after conda activate myenv line. It changed from “base” to “myenv” which means you are now working in “myenv” environment.
+ To install a library in your virtual environment type pip install <library_name>.
e.g., pip install pandas
+ Instead of installing libraries one by one you can even install by bunch, i.e., we have a txt file called requirements.tx which consists of all the libraries required to proceed with the project, so we can use it.
+ so, before installing requirements.txt, make sure you are in the specific path where your requirements.txt is located, basically this file is located in the folder where our executable files are located, so we need to move to that directory by following command.
**cd A:\folder_name**
+ Here A -> drive, folder name -> path where your executable file is saved
+ I go to that file path in anaconda using cd command 
1.	Go to drive where your project file is.
2.	Go to the path of your project using cd <path>
3.	Type pip install –r requirements.txt 
+ And all your required libraries will be downloaded and you can start your project.
+ But if you want to use jupyter notebook on the new environment you have to set it up for the new environment.
+ After you have installed all the libraries and created an environment, you need an editor to run the code, that is starting jupyter notebook, as soon as you enter jupyter notebook in the terminal you will definitely get this error. “Jupiter” is not recognized as an internal or external command.
So, to solve it it we have 2 commands.
1.	conda install –c conda-forge jupyterlab
2.	conda install –c anaconda python
Now you are ready to use jupyter on this environment and start with your project!

![thanks](https://media.giphy.com/media/3oEjHTsP47u1BTgY24/giphy.gif)
  
### _**Credits to my friend who gave detailed explanation of installation procedure.**_
+ https://github.com/PaVaNTrIpAtHi
+ https://www.linkedin.com/in/pavan-tripathi-3993641a1/

# _**Steps to execute**_
**Note:** Make sure you have added path while installing the software’s.
1. Install the prerequisites mentioned above.
2. open anaconda prompt and create a new environment.
  - conda create -n "env_name"
  - conda activate "env_name"
**If you face any issue while setting up, please feel free to click on the below link given in the issues section at the bottom to get more detailed explanation.
3. Install necessary libraries from requirements.txt file provided.
4. Run pip install -r requirements.txt or conda install requirements.txt (Requirements.txt is a text file consisting of all the necessary libraries required for executing this python file. If it gives any error while installing libraries, you might need to install them individually.)
5. Run classification_cnn_vgg.ipynb final code, and make sure to change the path of the model and dataset loading folders.

# _**Data Description**_
The particular dataset was downloaded from kaggle data repository, which consists of 10 classes and each class consists of around more than 100+ audio files. There are 10 music genres which are included in this dataset such as Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae and Rock. Each audio file has around 30 second timestamp. Below given link can be accessed to download the dataset from the kaggle data repository.

https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

 **Credits to the owners for making the dataset public.**
 
 # _**Issues Faced.**_
1. We might face an issue while installing specific libraries.
2. Make sure you have the latest version of python, since sometimes it might cause version mismatch.
3. Adding path to environment variables in order to run python files and anaconda environment in code editor, specifically in visual studio code.
4. Refer to the Below link to get more details on installing python and anaconda and how to configure it.
+ https://techieyantechnologies.com/2022/06/get-started-with-creating-new-environment-in-anaconda-configuring-jupyter-notebook-and-installing-libraries-using-requirements-txt-2/

# _**Note:**_
**All the required data hasn't been provided over here. Please feel free to contact me for any issues.**

### _**Let’s Connect**_
https://www.linkedin.com/in/abhinay-lingala-5a3ab7205/

![Connect](https://media3.giphy.com/media/3o6fJ0dKsQEGXytsOs/giphy.gif)

# _**Yes, you now have more knowledge than yesterday, Keep Going.**_
![Happy](https://www.qualitylogoproducts.com/blog/wp-content/uploads/2013/04/the-office-michael-scott-entrance.gif)
