# KSSEM-HIO25-065-athlentiq

# Athlentiq - Where Talent Meets Opportunity
<img width="1862" height="1914" alt="WhatsApp Image 2025-11-07 at 14_imgupscaler ai_Enhancer_2K" src="https://github.com/user-attachments/assets/897541b5-ced0-4e19-8b7d-43c716081d00" />

## BRIEF INTRO:
**Athlentiq** is an android application developed with Kotlin that leverages the capabilities of MediaPipe Pose Landmark Detection. It accurately counts repetitions, enhancing your workout experience and also its an all-in-one AI fitness companion that turns your cam into a personal trainer: it watches you work out, counts your reps in real time using pose estimation, and tracks your progress over time — no wearables, no manual logging, just you and your browser. 

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Exercises](#supported-exercises)
- [AI Model Training Process](#ai-model-training-process)
- [Getting Started](#getting-started)
- [What was found](#what-was-found)
- [What Worked](#what-worked)
- [What Didn't Work and Future Improvements](#what-didnt-work-and-future-improvements)
- [Pros and Cons of the App](#pros-and-cons-of-the-app)
- [Demo Video](#demo-video)
- [Screenshots](#screenshots)
- [Libraries](#libraries)
- [References](#references)
- [Contributors](#contributors)


## Overview
To transform how athletes are evaluated, connected, and developed. It brings together event hosting, verified assessments, performance analytics, nutrition tracking, and a professional sports network into one unified system.
Athletes can upload performance data and videos, complete standardized screening tests, and receive objective, tamper-proof analytics. Coaches, scouts and sports institutions get access to transparent dashboards. Organisers can host events and manage talent at scale. The platform integrates diet and nutrition guidance, enabling athletes to optimise both performance and health.

## Features

- *Pose Detection AI Model:* Accurate detection of user poses to evaluate exercise form.
- *Pose Classification:* Identify and classify different exercise poses for precise feedback.
- *Repetition Counting:* The AI model accurately counts repetitions for each exercise, helping users track their progress
- *Real-Time Feedback:* The app uses MediaPipe Pose Landmark Detection to analyze the user's exercise form in real-time. Currently it checks the confidence level of yoga and informs the user about the confidence percentage on it.
- *Exercise Plan:* Users can create customized exercise plans, specifying the number of repetitions for each exercise. Plans can be based on specific exercises, allowing for a personalized workout routine.
- *Room Database:* Athlentiq utilizes the Room database to store and manage exercise plans and workout results.
- *Voice Notifications:* Receive voice notifications at the start and completion of exercises for a seamless workout experience.
- *Exercise History:* The app maintains a detailed exercise history, providing users with insights into their performance over time. Users can view weekly and daily summaries, helping them stay motivated and on track.
- *Camera Flipping:* Easily switch between front and back cameras for varied workout perspectives.


## Supported Exercises

1. *Pushup*
2. *Squat*
3. *Situp*
4. *Deadlift*
5. *Chest Press*
6. *Shoulder Press*
7. *Lunges*

## Assessment Test

1. *Vetical Jump*
2. *Broad Jump*
3. *Squats*
4. *Medecine Ball*
5. *Sit and Reach*
6. *Shuttloe Run*

## AI Model Training Process

The training model is based on transfer learning and the training is done in [Google Colab ](https://colab.research.google.com/drive/19txHpN8exWhstO6WVkfmYYVC6uug_oVR):

### Step 1

- Collect approximately 200 images for each exercise pose from various sources.
- Compress images into a folder ("fitness_poses_images_in.zip") and upload to the specified section.

### Step 2

- Upload a sample video of a particular posture exercise for testing and model validation.

### Step 3

- Download the generated CSV file.
- Divide the output file into smaller CSV files based on posture names (e.g., "squats.csv").
- Integrate all files into the assets/pose directory of the Android application.


## Getting Started

### Prerequisites

- Android Studio: Make sure you have the latest version of Android Studio installed.
- Kotlin: The project is written in Kotlin, so familiarity with Kotlin is recommended.

### Installation

1. Clone the repository
    https://github.com/vinothmp21102005/KSSEM-HIO25-065-athlentiq
    
2. Open the project in Android Studio.

3. Build and run the app on your Android device or emulator.

OR--
1. Download the  [[app-release.apk]] to Android phone
2. Remember to "Allowing app installs from Unknown Sources in Android"
   
### Usage

1. Open the app on your Android device.

2. Choose the exercise you want to perform (pushup, squat, situp, lunges etc.).

3. Set up your exercise plan, specifying the number of repetitions for each exercise.

4. Start your workout, and the app will count your repetitions.

5. View your exercise history to track your progress over time.

6. Choose your assessment test (e.g., Sit and Reach, Vertical Jump, Shuttle Run, etc.).

7. Record your performance video for AI evaluation.

8.View AI-generated scores and progress reports on your dashboard.

9.Access Food AI for personalized nutrition plans and connect with coaches or event organizers for verified sports opportunities.

## What was found

- *Challenges in Model Training*:

Training the pose detection model posed significant challenges due to the requirement for a substantial number of images with perfect exercise form for each workout. The process demands meticulous curation of diverse training data to ensure the model's accuracy.

- *Successful Pose Detection and Repetition Count*:

With proper training data, accurate pose detection and repetition counting for various exercises is possible.

- *Conflicts in Similar Exercises*:

Similar forms of exercises can present conflicts for the pose detection model. Distinguishing between exercises with comparable poses requires careful consideration during both data collection and model training.

- *Pose Landmarker Model Details*:

The pose landmarker model tracks 33 body landmark locations, representing the approximate position of various body parts.The arrangement of landmark points provides comprehensive information about body orientation, limb positions, and spatial relationships.

- *Real-Time Operation and Continuous Analysis*:

MediaPipe Pose Landmark Detection operates in real-time, allowing continuous tracking and analysis of body movements during exercises.

- *Confidence Scores and Reliability*:

The pose detection model typically generates a confidence score for each landmark. This score serves as a valuable metric to assess the reliability of detected landmarks.

- *Application of Detected Landmark Data*:

The detected landmark data plays a crucial role in training machine learning models for broader analysis beyond simple pose detection.


## What Worked

- Efficiently utilized MediaPipe Pose Landmark Detection for accurate pose identification.
- Segregated training data into individual CSV files, ensuring future scalability.
- Successfully integrated trained models into the app for real-time feedback.
- Processed data extracted from the detection result and presented meaningful information to the user.

## What Didn't Work and Future Improvements

- Challenges in capturing a wide range of exercise variations for diverse training data.
- Pose landmarker model tracking 33 body landmarks affected classification for exercises with similar poses.
- Yoga pose detection accuracy may need improvement, especially in providing feedback for pose correction.
- Future improvement involves expanding the variety of exercises, requiring additional training and pose detection adjustments.

## Pros and Cons of the App

### Pros:
- Accurate pose detection and repetition count
- Customized exercise plans
- Exercise history and performance tracking
- Diverse supported exercises
- Voice notifications and seamless workout experience
- Gif images for learning how to do the consecutive exercise

### Cons:
- Challenge in model training
- There might be conflicts in similar Exercises
- The accuracy of yoga pose detection may need improvement


## Screenshots


## Demo Video



## Libraries

#### AndroidX Libraries:

- AppCompat Library - androidx.appcompat:appcompat:1.6.1
- Material Design Library - com.google.android.material:material:1.10.0
- ConstraintLayout Library - androidx.constraintlayout:constraintlayout:2.1.4
- Play Services Vision Common - com.google.android.gms:play-services-vision-common:19.1.3
- Camera Core Library - androidx.camera:camera-core:1.3.0
- Google ML Kit Common - com.google.mlkit:common:18.9.0
- Pose Detection Common - com.google.mlkit:pose-detection-common:17.0.0
- Pose Detection Accurate - com.google.mlkit:pose-detection-accurate:17.0.0
- Pose Detection - com.google.mlkit:pose-detection:17.0.0
- Navigation Runtime - androidx.navigation:navigation-runtime-ktx:2.7.5
- UI Automator - androidx.test.uiautomator:uiautomator:2.2.0
- Room Database - androidx.room:room-runtime:2.6.0
- Room KTX - androidx.room:room-ktx:2.6.0
- Fragment Testing - androidx.fragment:fragment-testing:1.6.2
- JUnit - junit:junit:4.13.2
- Espresso Core - androidx.test.espresso:espresso-core:3.5.1
- RecyclerView - androidx.recyclerview:recyclerview:1.3.2
- RecyclerView Selection - androidx.recyclerview:recyclerview-selection:1.1.0
- Kotlin Standard Library - org.jetbrains.kotlin:kotlin-stdlib-jdk7:1.5.31
- Multidex Library - androidx.multidex:multidex:2.0.1
- ViewPager2 - androidx.viewpager2:viewpager2:1.0.0
- Dots Indicator - com.tbuonomo:dotsindicator:4.3
- Curved Bottom Navigation - np.com.susanthapa:curved_bottom_navigation:0.6.5
- Navigation Fragment KTX - androidx.navigation:navigation-fragment-ktx:2.7.5
- Navigation UI KTX - androidx.navigation:navigation-ui-ktx:2.7.5
- Fragment KTX - androidx.fragment:fragment-ktx:1.3.6

#### Pose Detection Libraries:

- Pose Detection with Default Models - com.google.mlkit:pose-detection:18.0.0-beta3
- Pose Detection with Accurate Models - com.google.mlkit:pose-detection-accurate:18.0.0-beta3
- Google ML Kit Camera - com.google.mlkit:camera:16.0.0-beta3

#### Other Android Libraries:

- Google Gson - com.google.code.gson:gson:2.8.6
- Google Guava for Android - com.google.guava:guava:27.1-android
- CameraX - androidx.camera:camera-camera2:1.0.0-SNAPSHOT, androidx.camera:camera-lifecycle:1.0.0-SNAPSHOT, androidx.camera:camera-view:1.0.0-SNAPSHOT
- On Device Machine Learnings - com.google.android.odml:image:1.0.0-beta1
- MPAndroidChart - com.github.PhilJay:MPAndroidChart:v3.1.0
- GIF Drawable - pl.droidsonroids.gif:android-gif-drawable:1.2.23
- Glide - com.github.bumptech.glide:glide:4.16.0 (compiler: com.github.bumptech.glide:compiler:4.16.0)

#### Testing Libraries:

- AndroidX Core Library - androidx.test:core:1.4.0
- AndroidJUnitRunner - androidx.test:runner:1.4.0
- JUnit Rules - androidx.test:rules:1.4.0
- Assertions - androidx.test.ext:junit:1.1.3

#### ViewModel and LiveData:

- Lifecycle LiveData - androidx.lifecycle:lifecycle-livedata:2.3.1
- Lifecycle ViewModel - androidx.lifecycle:lifecycle-viewmodel:2.3.1


## References

[Pose landmark detection guide](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index)

[Train the model ](https://colab.research.google.com/drive/19txHpN8exWhstO6WVkfmYYVC6uug_oVR)


## Contributors
[Sanjit P] (https://github.com/sanjitp-23)
[Vinoth M P] (https://github.com/vinothmp21102005)
[Karthick Raja K] (https://github.com/Karthickraja23006120)
[Srivatsan V] (https://github.com/Srivatsan0405)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

