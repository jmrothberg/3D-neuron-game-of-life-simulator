#JMR SimpleChat buttons for models slider for temp  Streaming July 3 2023 - llama_cpp directly! But no GPU action
#Dialog boxes for loading directories
#Use ask directory so do not need to know the director in advance
#CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python
#the llamacpp directly runs gpu with my models when k_M .bin new format
#Added n_gqa=8 for llama2 70b models hack has n_gqa = 8 ONLY if model has "70" in it. 
import os
import tkinter as tk
from tkinter import scrolledtext
from llama_cpp import Llama
import time
import platform
from tkinter import Tk
from tkinter import filedialog, Scale
from tkinter import messagebox
from tkinter import Menu
from tkinter import END
import chromadb
from chromadb.config import Settings

print(platform.machine())

# Specify the path to the WorkingModels directory
directory = "/Users/jonathanrothberg/WorkingModels"
db = False

root = tk.Tk()
new_width = 800
new_height = 600
root.geometry(f"{new_width}x{new_height}")
root.title(("JMR's Little ggml LlamaCPP Chat")) # Set the title for the window

# Check if the default path with the model exists
if not os.path.exists(directory):
    messagebox.showinfo("Information","Select Directory for ggml llamacpp type models")
    directory = filedialog.askdirectory(title="Select Directory for ggml LlamaCPP type models")

print (directory)

# Get a list of all the .bin files in the directory
bin_files = [file for file in os.listdir(directory) if file.endswith(".bin")]

model_name = ""

# Print the numbered list of .bin files
print("Available models:")
for i, file in enumerate(bin_files, 1):
    print(f"{i}. {file}")


def clear_text(widget):
    if widget.winfo_class() == 'Text':
        widget.delete('1.0', END)
    elif widget.winfo_class() == 'Entry':
        widget.delete(0, END)


def show_context_menu(event):
    context_menu = Menu(root, tearoff=0)
    context_menu.add_command(label="Cut", command=lambda: root.focus_get().event_generate("<<Cut>>"))
    context_menu.add_command(label="Copy", command=lambda: root.focus_get().event_generate("<<Copy>>"))
    context_menu.add_command(label="Paste", command=lambda: root.focus_get().event_generate("<<Paste>>"))
    context_menu.add_command(label="Clear", command=lambda: clear_text(root.focus_get()))
    context_menu.tk_popup(event.x_root, event.y_root)

# Function to change the model
def change_model(model_number):
    global model, root, model_name
    model_name = bin_files[int(model_number) - 1]
    model_path = os.path.join(directory, model_name)
    if "70" in model_name:
        gqa = 8
    else:
        gqa = None
    print ("qga: ", gqa)
    temp = tempLLM.get()
    #model = Llama(model_path, use_mlock=True, n_gpu_layers=1)
    model = Llama(model_path, use_mlock=True, n_gpu_layers=1, n_ctx=4096, n_gqa=gqa)
    root.title(("JMR's Little " + model_name + " Chat. Temp: " + str(temp))) # Set the title for the window

# Create buttons for each available model
def create_model_buttons():
    for i, file in enumerate(bin_files, 1):
        button_text = file[:8]  # Limit the button text to the first 5 letters of the model name
        button = tk.Button(root, text=button_text, command=lambda i=i: change_model(i))
        button.grid(row=1, column=i +4, sticky='w')
    return (i) # number of buttons so we can space correctly

# Select persist directory
def select_directories():
    print("Select the Vectorstore/Persist directory for the vector search:")
    persist_directory = filedialog.askdirectory(initialdir=os.path.expanduser("~"))
    # Create the persist directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    print("VectorStore/Persist directory:", persist_directory)
    return persist_directory


def set_up_chromastore():
    global collection, db, vector_folder_name
    print ("setting up db & collection")
    #Select the Folder for the VectorStore/Database you want to use   
    persist_directory = select_directories()  
    vector_folder_name = os.path.basename(persist_directory)
    # Set the persist_directory and CHROMA_SETTINGS to match the previous configuration
    CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory = persist_directory,
        anonymized_telemetry=False
    )
    # Load the Chroma database
    db = chromadb.Client(settings=CHROMA_SETTINGS)
    
    collections = db.list_collections() # Debugging to get name in collectons
    for collection in collections:
        print(collection.name)
    mycollection = (collections[0].name)
    
    print (mycollection) # it is "langchain" when I use the langchain based ingest, but in future it may have different names.
    collection = db.get_collection(mycollection)

# Perform a search in the loaded database
def vector_search(query):
    docs = collection.query(query_texts =[query], n_results=3)
    vector_text = ""
    sources = ""
    print(docs)
    for i in range(len(docs['documents'][0])):
        print("Document:", docs['documents'][0][i])
        vector_text = vector_text + "\n Vector Response "+ str(i) +": " + docs['documents'][0][i]
        source_i = docs['metadatas'][0][i]['source']
        
        if '/' in source_i:
            source_i = source_i.split('/')[-2:] #removing path from source.
        else:
            source_i = source_i
    
        source_i = ' '.join(source_i)  # convert list to string
        print ("Source: ",source_i)
        sources = sources + "\n Sources: " + source_i
        print("\n") # Just to create a new line between entries
    return vector_text, sources


def talk_to_vector_search():
    if not db:
        set_up_chromastore()
    prompt = entry.get("1.0", "end-1c") # get text from Text widget    
    #prompt = text_area.get("1.0", "end-1c") # get text from Text widget
    vector_answers, sources = vector_search(prompt)
    #response_with_vector = model("### USER: " + prompt + "Answer based on this information: " + vector_answer)
    text_area.insert("end", f"\n\n----- {vector_folder_name} Database Search -----\n")
    text_area.see(tk.END) 
    text_area.insert(tk.END, f"{vector_answers}\n")
    
    text_area.insert(tk.END, f"{sources}\n")
    text_area.see(tk.END)  # Make the last line visible
    text_area.update() 


def save_text():
    prompt = entry.get("1.0", "end-1c")  # Get text from Text widget
    generated_text = text_area.get("1.0", "end-1c")  # Get text from Text widget
    # Create the filename using the first 10 characters of the prompt and a 4-digit timestamp
    filename = prompt[:10] + "_" + model_name[:5] + "_" + time.strftime("%m%d-%H%M") + ".txt"
    # Create a directory to save the files if it doesn't exist
    directory = "saved_texts"
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Save the prompt and generated text to the file
    with open(os.path.join(directory, filename), "w") as file:
        file.write("Prompt:\n" + prompt + "\n\nGenerated Text:\n" + generated_text)
    print("Text saved successfully in: ",filename)


def talk_to_LLM():
    if not model_name:
        print ("No Model Loaded")
        return
    print ('''	### System:
    {System}
    ### User:
    {User}
    ### Assistant:
    {Assistant}''')
    print ("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER:  ASSISTANT:")

    temp = tempLLM.get()
    root.title(("JMR's Little " + model_name + " Chat. Temp: " + str(temp))) # Set the title for the window
    prompt = entry.get("1.0", "end-1c") # get text from Text widget
  
    fullprompt = "### User: \n" + prompt + " \n### Assistant: " + model_name + " Chat. Temp: " + str(temp) + "\n"
    text_area.insert(tk.END, f"{fullprompt}")
    
    #response = model (fullprompt, max_tokens = max_new_tokens.get(), temperature = temp, stream = True)
    response = model (prompt, max_tokens = max_new_tokens.get(),temperature = temp, stream = True )
    print (response) # a function when streaming.
    #text_area.insert(tk.END, f"Result:\n")
    for stream in response:
        #text = stream["choices"][0]["delta"].get("content", "")
        text = stream['choices'][0]['text']
        text_area.insert(tk.END, text)
        text_area.see(tk.END)  # Make the last line visible
        text_area.update()
    text_area.insert(tk.END, "\n\n")
    
new_width = 800
new_height = 800
root.geometry(f"{new_width}x{new_height}")
root.title(("JMR's Little " + model_name + " Chat")) # Set the title for the window
root.bind("<Button-2>", show_context_menu)

root.grid_rowconfigure(0, weight=1)  # Entry field takes 1 part
root.grid_rowconfigure(1, weight=0)  # "Send" button takes no extra space
root.grid_rowconfigure(2, weight=0)  # Model buttons take no extra space
root.grid_rowconfigure(3, weight=3)  # Output field takes 3 parts
root.grid_columnconfigure(0, weight=1)  # Column 0 takes all available horizontal space

# Add the create_model_buttons function call
number_of_models = create_model_buttons()  # Create buttons for each available model
print (number_of_models)

if number_of_models > 3:
    new_width += 180 * (number_of_models - 3)
    root.geometry(f"{new_width}x{new_height}")

entry = scrolledtext.ScrolledText(root, height=5,wrap="word") # change Entry to ScrolledText, set height
entry.grid(row=0, column=0, columnspan=number_of_models+8, sticky='nsew') # make it as wide as the root window and expand with window resize

button_V = tk.Button(root, text="Search", command=talk_to_vector_search)
button_V.grid(row=1, column=number_of_models+6, sticky='e')

button = tk.Button(root, text="Send", command=talk_to_LLM)
button.grid(row=1, column=number_of_models+7, sticky='e')  # place Send button in row 1, column 1, align to right

save_button = tk.Button(root, text="Save", command=save_text)
save_button.grid(row=1, column=0, sticky='w')  # Place the Save button in row 1, column 0, align to left

temp_label = tk.Label(root, text="Temperature:")
temp_label.grid(row=1, column=1, sticky='w')

tempLLM = tk.DoubleVar(value = 0.0)
slider = tk.Scale(root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, variable=tempLLM)
slider.grid(row=1, column=2, sticky='w')
temp = tempLLM.get()

max_label = tk.Label(root, text="Max New Tokens:")
max_label.grid(row=1, column=3, sticky='w')

max_new_tokens = tk.DoubleVar(value = 256)
slider_token = tk.Scale(root, from_=0, to=4000, resolution=1, orient=tk.HORIZONTAL, variable=max_new_tokens)
slider_token.grid(row=1, column=4, sticky='w')

text_area = tk.Text(root, wrap="word")
text_area.grid(row=3, column=0, columnspan=number_of_models+8, sticky='nsew')  # make text area fill the rest of the window and expand with window resize, span 3 columns

# Adding a scrollbar to the text area
scrollbar = tk.Scrollbar(text_area)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
text_area.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=text_area.yview)

root.mainloop()

