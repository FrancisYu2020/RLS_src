from pynput import keyboard
import pyautogui
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import time
import threading
from utils import *
import argparse

parser = argparse.ArgumentParser(description="EMG data labeling script")
parser.add_argument('start_frame', type=int, help="The index of the start frame")
parser.add_argument('start_timestamp', type=str, help="The start timestamp, format hh:mm:ss")
parser.add_argument('snapshots_path', type=str, help="Path to which the snapshots during the labeling are stored")
parser.add_argument('--csv', type=str, help="csv file path to store the labeled data, if .csv is not add, it will be automatically processed", default="EMG_label.csv")

args = parser.parse_args()

#TODO: debug the whole pipeline

#NOTE: left leg y axis: 373, right leg y axis: 393, left margin: 107, right margin: 10

# Define global variables here
running = True

def listen_for_exit():
    '''
    function for capturing the exit keyboard signal, default use 'esc' to exit
    '''
    global running
    print('Press "Esc" to exit...')
    keyboard.wait('esc')
    running = False

# Specify the file path for saving the CSV
global_start_timestamp = args.start_timestamp.split(':')
global_start_timestamp = [float(x) for x in global_start_timestamp]
print(global_start_timestamp)

annotator = Annotator(global_start_timestamp, args.start_frame, csv_file_path=args.csv, snapshots_path='debug')

# Set up and start the listener thread for early exit
listener_thread = threading.Thread(target=listen_for_exit)
listener_thread.start()

# Capture the initial screen and process the first screen
prev_screen = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
epoch_count = 0
annotator.process_screenshot(prev_screen, epoch_count)

while running:

    # Capture a new screen after some time
    # new_screen = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_BGR2GRAY)
    new_screen = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
    
    # Check if the screen has changed significantly
    if annotator.screen_changed(prev_screen, new_screen):
        epoch_count += 1
        annotator.process_screenshot(new_screen, epoch_count)

    # Update the previous screen
    prev_screen = new_screen

    # prevent busy loop
    time.sleep(0.5)

    # Add a delay or a method to exit the loop if necessary
print(f'total epoch counted: {epoch_count}')
running = False

listener_thread.join()
print('Program terminated')