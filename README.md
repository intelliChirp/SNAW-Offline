# Soundscape Noise Analysis Workbench
# Offline Standalone Version

[Team Website](https://www.cefns.nau.edu/capstone/projects/CS/2020/IntelliChirp-S20/)

##### Workflow
1. Run python script with -i input-folder of audio files and -o output-folder
2. Results save to .csv files for each uploaded file

## Description

### Problem
Various environmental changes affect a range of species around the world and as more species are being affected, proper management and observation are required to understand their response. Traditional field methods require trained observers to determine species presence/absence and are thus expensive and challenging to employ at large scales. Using sound to monitor biodiversity across landscapes is a fairly recent development.

Our clients are working with [Soundscapes2Landscapes](https://soundscapes2landscapes.org/). They are having a problem with an un-user friendly application that requires manual identification in terabytes of sound files. This manual approach is incredibly time consuming and needs to be automated. We feel confident that we can provide a solution that is user friendly and automates that identification process with machine learning.

### Solution

Our envisioned solution is a user friendly web application for use by any researcher or citizen scientist. This application is called the Soundscape Noise Analysis Workbench (S.N.A.W.), and will allow users to analyze sound files with the power of machine learning. The results given to the users include a summary of the audio components in the file, acoustic indices, and an export of the sound file with background noise masked out. Users will gain a better understanding of how various sources of noise in soundscape recordings diminish the ability to detect individual bird species and quantify avian diversity. Using machine learning, instead of the current manual identification process, will drastically speed up the identification of terabytes of acoustic data. This solution will allow users anywhere, anytime, to upload their soundscapes for noise analysis, quickly.

## Installation

Install Python (version 3.7.4) from: 
https://www.python.org/downloads/release/python-374/2.

Make sure Pip is up-to-date. Run command: 
~~~~
py -m pip install --upgrade pip
~~~~

Navigate to the directory where you would like to install the virtual environment. Run command:
~~~~
py -m venv snaw 
~~~~
(change ‘snaw’ to what you would like to name the virtual environment).

Activate the environment by navigating to the file directory of your environment and running command 
~~~~
.\Scripts\activate
~~~~

Verify that the virtual environment is running by looking on the left hand side of the bottom line. You should see the name of your virtual environment in parenthesis (Ex:(snaw) C:\......)7. Visit SNAW’s ‘Installation_Requirements’ repository at: https://github.com/intelliChirp/Installation_Requirements . 

Click the green ‘Clone or download’ button. 

Click on ‘Download ZIP’. 

Copy requirements.txt file to your virtual environment directory. 

Navigate to your virtual environment’s directory. Type the command: 
~~~~
pip install -r requirements.txt
~~~~
Wait for the packages to install on your local machine

## Running

Click on the green ‘Clone or download’ button

Activate the virtual environment by running 
~~~~
.\Scripts\activate
~~~~
Navigate to the standalone application’s repository on your local machine.

Navigate to /SNAW-Offline/snaw-standalone

There are already 2 directories created for you to use. Place your WAV files in the ‘audio’ directory. The output folder is where the CSV results will be placed after aclassification. Type command:
~~~~
py snaw.py -i audio -o output
~~~~
After the application completes, your results will be located in the ‘output’ directory. 

Note: The directories do not need to be named ‘audio’ and ‘output’. That is only there by default for convenience. You can create folders with any name, and then use the command format 
~~~~
py snaw.py -i [AUDIO-DIRECTORY] -o [OUTPUT-DIRECTORY]
~~~~

[Windows Command Prompt Users] If you would like to export the output of the application in a text file, add ‘| tee file-name.txt’ to the end of your command. Once the application finishes running, the text file will appear.
Note: Save this file elsewhere once the classification completes. If you run the application again with the same text file name, the file will be overwritten by the new classification.
Example: 
~~~~
py snaw.py -i audio -o output | tee output.txt
~~~~

## Authors

* **Steven Enriquez** - *Team Lead, Front-End Lead* - [GitHub](https://github.com/stevenenriquez)
* **Michael Ewers** - *Back-End Lead* - [GitHub](https://github.com/MEwers96)
* **Joshua Kruse** - *Machine Learning Lead* - [GitHub](https://github.com/JoshEKruse)
* **Zhenyu Liei** - *Testing Lead* - [GitHub](https://github.com/lei37927)

## Acknowledgments

* **Colin Quinn.** - *Client*
* **Patrick Burns.** - *Client*

School of Informatics, Computing and Cyber Systems
Global Earth Observation and Dynamics of Ecosystems Lab

* **Soundscapes 2 Landscapes** - [Website](https://soundscapes2landscapes.org/)
