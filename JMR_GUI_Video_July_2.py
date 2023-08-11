#JMR July 2 text to vidio - changed to float32, removed cpu_offload.WORKS!! Added my own player.
#Must open with VLC player or use built in player. Added dialoge box to pick location of model, and locatino of 
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import datetime
import cv2
import tkinter as tk
from tkinter import Tk
from tkinter import filedialog, Scale
from tkinter import messagebox
import os
import shutil

default_path = "/Users/jonathanrothberg"
model = "text-to-video-ms-1.7b"

root = tk.Tk()
root.title("JMRs Text to Video")
root.geometry("800x400")

default_path = "/Users/jonathanrothberg"
model = "text-to-video-ms-1.7b"
video_directory = "/Users/jonathanrothberg/VideoOut"

# Check if the default path with the model exists
if not os.path.exists(os.path.join (default_path, model)):
    messagebox.showinfo("Information","Select model location for: " + model)
    path = filedialog.askdirectory(title="Please select the directory containing the model: " + model)
     #path = path + "/" + model
    path = os.path.join(path, model)
    
else:
    path = os.path.join(default_path, model)
print (path)

pipe = DiffusionPipeline.from_pretrained(path, torch_dtype=torch.float32, variant="fp32")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Check if the video directory exists
if not os.path.exists(video_directory):
    messagebox.showinfo("Information","Please select Directory to Save Videos to")
    video_directory = filedialog.askdirectory(title="Please select Directory to Save Videos to")
print (video_directory)

# optimize for GPU memory
#pipe.enable_model_cpu_offload()
#pipe.enable_vae_slicing()

#Display the video
def displayvideo(path):
    if not path:  # Check if the path is empty
        print("No video file selected.")
        return
    cap = cv2.VideoCapture(path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
        else:
            break
    cap.release()
    

def displayvideoframe(path):  # adding a slider to view just frames we want.
    if not path:  # Check if the path is empty
        print("No video file selected.")
        return
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames
    # print(total_frames)
    # Set the current frame to display based on the slider value
    current_frame = int(total_frames * (frame_slider.get() / 100))
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    ret, frame = cap.read()
    if ret:
        cv2.imshow('Video', frame)
        
    cap.release()


def replay_video():
    displayvideo(video_path.get())
    
def show_frame(frame):
    displayvideoframe(video_path.get())
    
def load_video():
    video_path.set(filedialog.askopenfilename())  # Show an "Open" dialog box and return the path to the selected file
    displayvideo(video_path.get())

def save_as():
    save_path = filedialog.asksaveasfilename(defaultextension=".mp4")  # Show a "Save As" dialog box and return the path to the selected file
    if save_path:
        shutil.copy(video_path.get(), save_path)  # Copy the video to the new location

def generate_video():
    prompt = video_prompt.get()
    print (inference_steps.get(), num_frames.get()) #These get the correct values
    video_frames = pipe(prompt, num_inference_steps=inference_steps.get(), num_frames= num_frames.get()).frames #when number of frames changes bad generateion
    #video_frames = pipe(prompt, num_inference_steps=inference_steps.get()).frames # This works with scheduler
    #video_frames = pipe(prompt, num_inference_steps=25).frames  # workswell with set frames.
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    video_name = prompt[:10]
    #video_path.set(video_directory + f"{video_name}_{current_date}.mp4")
    video_path.set(os.path.join(video_directory, f"{video_name}_{current_date}.mp4"))
    export_to_video(video_frames, video_path.get())
    displayvideo(video_path.get())

current_frame = tk.IntVar()

video_prompt = tk.StringVar()
video_path = tk.StringVar()
inference_steps = tk.IntVar(value=25)
num_frames = tk.IntVar(value = 32)

frame_show = tk.IntVar(value = 50)

prompt_entry = tk.Entry(root, textvariable=video_prompt)
prompt_entry.pack(fill=tk.X)  # Make the entry widget fill the entire width of the window

frames_slider = Scale(root, from_=1, to=500, orient=tk.HORIZONTAL, label="Number of Frames", variable=num_frames,length=300)
frames_slider.pack()

inference_slider = Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, label="Inference Steps", variable=inference_steps,length=300)
inference_slider.pack()

generate_button = tk.Button(root, text="Generate", command=generate_video)
generate_button.pack()

replay_button = tk.Button(root, text="Replay", command=replay_video)
replay_button.pack()

frame_slider = Scale(root, from_=1, to=100, orient=tk.HORIZONTAL, label="Frame to View %", variable=frame_show,command=show_frame, length=300)
frame_slider.pack()

load_button = tk.Button(root, text="Load", command=load_video)
load_button.pack()

save_as_button = tk.Button(root, text="Save As", command=save_as)  # Add a "Save As" button
save_as_button.pack()

quit_button = tk.Button(root, text="Quit", command=root.quit)
quit_button.pack()

root.mainloop()
cv2.destroyAllWindows()