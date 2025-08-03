import pandas as pd
import tkinter as tk
from tkinter import messagebox, Scrollbar, Toplevel, scrolledtext
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt


df = pd.read_csv('shuffled_combined_news.csv')
X, y = df['title'], df['label']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Model
model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {100*accuracy:.2f}")


def predict_news():
    user_input = text_entry.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showwarning("⚠ Input Missing", "Please enter a news headline.")
        return
    prob = model.predict_proba([user_input])[0]
    pred_class = model.classes_[prob.argmax()]
    confidence = prob.max()
    if pred_class == 1:
        result_label.config(text=f"✅ REAL NEWS | Confidence: {confidence:.2f}", fg="lightgreen")
    else:
        result_label.config(text=f"🚨 FAKE NEWS | Confidence: {confidence:.2f}", fg="red")

def clear_text():
    text_entry.delete("1.0", tk.END)
    result_label.config(text="")

# Theme Toggle
dark_mode = True
def toggle_dark_mode():
    global dark_mode
    dark_mode = not dark_mode
    bg = "#121212" if dark_mode else "#f0f0f0"
    fg = "white" if dark_mode else "black"
    txt_bg = "#2C2C2C" if dark_mode else "white"
    txt_fg = "white" if dark_mode else "black"
    root.config(bg=bg)
    header.config(bg=bg, fg=fg)
    input_frame.config(bg=bg)
    label.config(bg=bg, fg=fg)
    button_frame.config(bg=bg)
    result_label.config(bg=bg, fg="#F1C40F")
    text_entry.config(bg=txt_bg, fg=txt_fg, insertbackground=txt_fg)

# WordCloud
def show_wordcloud(label_type):
    text_data = " ".join(df[df['label'] == label_type]['title'])
    color_map = "Blues" if label_type == 1 else "Reds"
    wc = WordCloud(width=800, height=400, background_color="white", colormap=color_map).generate(text_data)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud for {'REAL' if label_type==1 else 'FAKE'} News")
    plt.show()

#  Show Dataset Stats
def show_dataset_stats():
    total = len(df)
    real_count = sum(df['label'] == 1)
    fake_count = sum(df['label'] == 0)

    stats_window = Toplevel(root)
    stats_window.title("📊 Dataset Statistics")
    stats_window.geometry("400x250")
    stats_window.config(bg="#1e1e1e" if dark_mode else "white")

    label_stats = tk.Label(stats_window, text=f"📊 DATASET STATISTICS\n\nTotal Samples: {total}\nReal News: {real_count}\nFake News: {fake_count}",font=("Arial", 14), bg="#1e1e1e" if dark_mode else "white", fg="white" if dark_mode else "black")
    label_stats.pack(pady=20)

    def on_close():
        stats_window.destroy()
        labels = ['Real', 'Fake']
        sizes = [real_count, fake_count]
        colors = ['#4CAF50', '#E74C3C']
        explode = (0.05, 0.05)
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode, shadow=True)
        ax.axis('equal')
        plt.title("Real vs Fake News Distribution")
        plt.show()

    stats_window.protocol("WM_DELETE_WINDOW", on_close)

root = tk.Tk()
root.title("📰 Fake News Detector")
root.state('zoomed')
root.config(bg="#121212")

header = tk.Label(root, text="📰 News Headline Detection System", font=("Helvetica", 22, "bold"), fg="white", bg="#121212", pady=15)
header.pack(fill="x")

input_frame = tk.Frame(root, bg="#121212", padx=20, pady=10)
input_frame.pack(pady=10, fill="x")

label = tk.Label(input_frame, text="Enter News Text Below:", font=("Arial", 16), bg="#121212", fg="white")
label.pack(anchor="w")

# Textbox with Scrollbar
text_frame = tk.Frame(input_frame, bg="#121212")
text_frame.pack(pady=5, fill="x")
scrollbar = Scrollbar(text_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
text_entry = tk.Text(text_frame, height=18, width=120, font=("Arial", 13), wrap="word",bg="#2C2C2C", fg="white", insertbackground="white", relief="flat", padx=10, pady=10, yscrollcommand=scrollbar.set)
text_entry.pack(side=tk.LEFT, fill="both", expand=True)
scrollbar.config(command=text_entry.yview)

# Button Style
def style_button(btn, color):
    btn.config(font=("Arial", 14, "bold"), fg="white", bg=color, relief="flat", padx=12, pady=6, activebackground="#333333", cursor="hand2")

button_frame = tk.Frame(root, bg="#121212")
button_frame.pack(pady=10)

btn_stats = tk.Button(button_frame, text="📈 Dataset Stats", command=show_dataset_stats)
style_button(btn_stats, "#16A085")
btn_stats.grid(row=0, column=0, padx=10)

btn_clear = tk.Button(button_frame, text="🗑️ Clear", command=clear_text)
style_button(btn_clear, "#E74C3C")
btn_clear.grid(row=0, column=1, padx=10)

btn_dark = tk.Button(button_frame, text="🌗 Toggle Theme", command=toggle_dark_mode)
style_button(btn_dark, "#8E44AD")
btn_dark.grid(row=0, column=2, padx=10)

btn_wc_real = tk.Button(button_frame, text="☁️ Real WordCloud", command=lambda: show_wordcloud(1))
style_button(btn_wc_real, "#3498DB")
btn_wc_real.grid(row=0, column=3, padx=10)

btn_wc_fake = tk.Button(button_frame, text="🔥 Fake WordCloud", command=lambda: show_wordcloud(0))
style_button(btn_wc_fake, "#E67E22")
btn_wc_fake.grid(row=0, column=4, padx=10)

btn_check = tk.Button(button_frame, text="✔️ Check News", command=predict_news)
style_button(btn_check, "#27AE60")
btn_check.grid(row=0, column=5, padx=10)

result_label = tk.Label(root, text="", font=("Arial", 18, "bold"), bg="#121212", fg="#F1C40F", pady=15)
result_label.pack()

root.mainloop()
