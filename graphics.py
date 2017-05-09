from tkinter import *

class Example(Frame):
    def __init__(self, *args, **kwargs):
        Frame.__init__(self, *args, **kwargs)
        self.l1 = Label(self, text="Hover over me")
        self.l2 = Label(self, text="", width=40)
        self.l1.pack(side="top")
        self.l2.pack(side="top", fill="x")

        self.l1.bind("<Enter>", self.on_enter)
        self.l1.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        self.l2.configure(text="Hello world")

    def on_leave(self, enter):
        self.l2.configure(text="")

if __name__ == "__main__":
    root = Tk()
    Example(root).pack(side="top", fill="both", expand="true")
    root.mainloop()