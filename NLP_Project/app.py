import tkinter as tk
from tkinter import scrolledtext
from yake import KeywordExtractor
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    def get_keywords():
        text = input_text.get("1.0", tk.END)
        extractor = KeywordExtractor(lan="en", top=10)
        keywords = extractor.extract_keywords(text)
        result_text.insert(tk.END, "Keywords from YAKE model:\n")
        for score, keyword in keywords:
            result_text.insert(tk.END, f"{keyword} - Score: {score}\n")
        result_text.insert(tk.END, "\n")

    def get_bow_result():
        def search():
            key_input = search_entry_2.get()
            vectorizer = CountVectorizer()
            key_size = len(nltk.word_tokenize(key_input))
            document = input_text.get("1.0", tk.END)
            tokens = nltk.word_tokenize(document)
            new_text, accepted, idx = "", [], 0
            while idx < (len(tokens) - key_size):
                doc = ''.join(text + " " for text in tokens[idx: idx + key_size])
                vector = vectorizer.fit_transform([doc, key_input])
                cosine = cosine_similarity(vector[0], vector[1])
                if cosine == 1.0: 
                    accepted.append(idx)
                    idx += key_size - 1
                idx += 1
            ind, cnt, idx = 0, 0, 0
            while idx < len(tokens):
                if ind < len(accepted) and idx == accepted[ind]:
                    while cnt < key_size:
                        if idx < len(tokens) - 1 and (tokens[idx + 1] == "n't" or tokens[idx + 1] in string.punctuation): 
                            new_text += (tokens[idx] + tokens[idx + 1] + " ").upper()
                            idx += 1
                        else: new_text += (tokens[idx] + " ").upper()
                        cnt += 1
                        idx += 1
                    ind += 1
                    idx -= 1
                    cnt = 0
                elif idx < len(tokens) - 1 and (tokens[idx + 1] == "n't" or tokens[idx + 1] in string.punctuation): 
                    new_text += tokens[idx] + tokens[idx + 1] + " "
                    idx += 1
                else: new_text += tokens[idx] + " "
                idx += 1
            for text in new_text: result_text.insert(tk.END, text)
            input_window.destroy()
                
                
        # Create a new window for input
        input_window = tk.Toplevel(background = "#ff7f50")
        input_window.title("Input-BOW")
        input_window.geometry("400x200")  # Set the size of the window

        # Label and Entry widget for user input
        search_label = tk.Label(input_window, text="Enter your word:", font="Arial 12 bold", background = "Orange")
        search_label.pack(pady=10)
        
        search_entry_2 = tk.Entry(input_window, width=30, font="Arial 12")
        search_entry_2.pack(pady=5)

        # Button to trigger search
        search_button = tk.Button(input_window, text="Search", font="Arial 12 bold", background="Orange", command=search)
        search_button.pack(pady=10)

    def clean_text():
        result_text.delete('1.0', 'end')

    # Create the main window
    window = tk.Tk()
    window.title("Keyword Extraction")
    window.configure(background='gold')
    window.geometry("1350x750+0+0")

    tk.Label(window, text = "Input text", font = "Arial 14 bold", background = "Orange").pack(padx = 0, pady = 5)
    # Create a scrolled text widget
    input_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=100, height=20, background="#40e0d0")
    input_text.pack(padx=10, pady=10)

    # Create a frame for buttons
    button_frame = tk.Frame(window, bg="gold")  # Set background color
    button_frame.pack(padx=10, pady=10)

    # Create the 'Get Keywords' button
    keywords_button = tk.Button(button_frame, text="Get Keywords", font="Arial 14 bold", background="Orange", command=get_keywords)
    keywords_button.grid(row=0, column=0, padx=5)

    # Create the 'Get BOW Result' button
    bow_button = tk.Button(button_frame, text="Search Information", font="Arial 14 bold", background="Orange", command=get_bow_result)
    bow_button.grid(row=0, column=2, padx=5)

    clear = tk.Button(button_frame, text = "Clean", font = "Arial 14 bold", background = "Orange", command = clean_text)
    clear.grid(row = 0, column = 3, padx = 5)

    # Create a scrolled text widget for displaying results
    tk.Label(window, text = "Ouput text", font = "Arial 14 bold", background = "Orange").pack(padx = 10, pady = 10)

    result_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=100, height=20, background="#40e0d0")
    result_text.pack(padx=10, pady=10)
    window.mainloop()

if __name__ == "__main__":
    main()
