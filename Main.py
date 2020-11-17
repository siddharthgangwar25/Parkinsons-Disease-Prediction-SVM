import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pylab as pl
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from tkinter import *
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import ttk
import tkinter
import csv

#Preprocessing dataset

df=pd.read_csv("parkinsons.csv")
X=df.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,12,14,15,16,18,19,20,21,22,23]].values
y=df.iloc[:,17].values
# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

def pca_button(top1):
    top1.destroy()
    plot_pca()

def plot_pca():
    top2=Toplevel()
    fig = Figure(figsize=(5, 5))
    a = fig.add_subplot(111)
    for i in range(0, X_train.shape[0]):
        if y_train[i] == 0:
            c1 = a.scatter(X_train[i, 0], X_train[i, 1], c='r', marker='+')
        elif y_train[i] == 1:
            c2 = a.scatter(X_train[i, 0], X_train[i, 1], c='g', marker='o')

    a.set_title("After PCA",pad=20)
    a.set_ylabel("2nd Principal Component")
    a.set_xlabel("1st Principal Component")
    a.legend([c1, c2], ['Healthy', "Parkinson's"])
    canvas = FigureCanvasTkAgg(fig, master=top2)
    canvas.get_tk_widget().pack()
    canvas.draw()
    btn1=Button(top2,text="Exit", width=10,command=top2.destroy).pack(pady=5)


def apply_pca():
    button2 = Button(root, text='Apply PCA', width=10, state=DISABLED).place(x=200,y=360)
    global X_train
    global X_test
    pca = PCA(n_components=None)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    variance = pca.explained_variance_ratio_
    variance.tolist()
    top1=Toplevel()
    top1.title("PCA Variance")
    frame_pca=LabelFrame(top1,text="Variance Explained by each dimension:",padx=10,pady=10)
    frame_pca.pack(padx=50,pady=15)
    for i in variance:
        v_label=Label(frame_pca,text=i).pack()

    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    btn=Button(top1,text="Choose n_components = 2", command=lambda: pca_button(top1)).pack(pady=5)

def apply_svm2(c_value,g_value):
    top4=Toplevel()
    global y_train
    global y_test
    classifier = SVC(kernel='rbf', C=c_value, gamma=g_value, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    frame_accuracy = LabelFrame(top4, text="Accuracy:", padx=2, pady=2)
    frame_accuracy.pack(padx=15, pady=15)
    accuracy_label=Label(frame_accuracy,text=round(accuracy_score(y_test, y_pred),2)*100).pack()
    frame_samples = LabelFrame(top4, text="Misclassified Samples:", padx=2, pady=2)
    frame_samples.pack(padx=15, pady=15)
    sample_label=Label(frame_samples,text=(y_test!=y_pred).sum()).pack()

    frame_graph = LabelFrame(top4, text="Boundary", padx=2, pady=2)
    frame_graph.pack(padx=15, pady=15)

    Xp = X_train
    yp = y_train

    def make_meshgrid(x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    model = svm.SVC(kernel='rbf', C=c_value, gamma=g_value)
    clf = model.fit(Xp, yp)
    fig, ax = plt.subplots()

    # Set-up grid for plotting.
    X0, X1 = Xp[:, 0], Xp[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=yp, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title("Decision Boundaries of SVM RBF kernel",pad=10)
    ax.set_ylabel("2nd Principal Component")
    ax.set_xlabel("1st Principal Component")

    canvas = FigureCanvasTkAgg(fig, master=frame_graph)
    canvas.get_tk_widget().pack()
    canvas.draw()

def apply_svm_button(c_value,g_value,top3):
    top3.destroy()
    apply_svm2(c_value,g_value)

def apply_svm1():
    top3=Toplevel()
    top3.geometry("400x300")
    frame_svm3 = LabelFrame(top3, padx=2, pady=2)
    frame_svm3.pack(padx=10, pady=10)
    label_ker=Label(frame_svm3,text="Kernel : RBF").pack()
    frame_svm1 = LabelFrame(top3, text="Choose value for C (Hard Margin/Soft Margin)", padx=2, pady=2)
    frame_svm1.pack(padx=10, pady=10)
    horizontal1=Scale(frame_svm1,orient = HORIZONTAL,length=200,tickinterval=50,from_=0.1,to=100)
    horizontal1.pack(pady=5)
    frame_svm2 = LabelFrame(top3, text="Choose value for Gamma", padx=2, pady=2)
    frame_svm2.pack(padx=10, pady=10)
    horizontal2 = Scale(frame_svm2, orient=HORIZONTAL, length=200, tickinterval=50, from_=0.1, to=100)
    horizontal2.pack(pady=5)
    btn2=Button(top3,text="Submit",command=lambda : apply_svm_button(horizontal1.get(),horizontal2.get(),top3))
    btn2.pack()

def opt_plot(max_c,max_g,max_acc):
    top6 = Toplevel()
    frame_opt_graph=LabelFrame(top6)
    frame_opt_graph.pack()
    svm_model = SVC(kernel='rbf', C=max_c, gamma=max_g, random_state=0)

    classify = svm_model.fit(X_train, y_train)

    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    def make_meshgrid(x, y, h=.1):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy

    X0, X1 = X_train[:, 0], X_train[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    fig, ax = plt.subplots(figsize=(6,6))
    fig.patch.set_facecolor('white')
    cdict1 = {0: 'lime', 1: 'deeppink'}

    Y_tar_list = y_train.tolist()
    yl1 = [int(target1) for target1 in Y_tar_list]
    labels1 = yl1

    labl1 = {0: "Healthy", 1: "Parkinson's"}
    marker1 = {0: '*', 1: 'd'}
    alpha1 = {0: .8, 1: 0.5}

    for l1 in np.unique(labels1):
        ix1 = np.where(labels1 == l1)
        ax.scatter(X0[ix1], X1[ix1], c=cdict1[l1], label=labl1[l1], s=70, marker=marker1[l1], alpha=alpha1[l1])

    ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=40, facecolors='none',
               edgecolors='navy', label='Support Vectors')

    plot_contours(ax, classify, xx, yy, cmap='seismic', alpha=0.4)
    ax.legend(fontsize=15)
    ax.set_title("SVM Solution with Support Vectors",pad=20)
    ax.set_ylabel("2nd Principal Component")
    ax.set_xlabel("1st Principal Component")

    canvas = FigureCanvasTkAgg(fig, master=frame_opt_graph)
    canvas.get_tk_widget().pack()
    canvas.draw()
    ker='RBF'
    s=("Maximum Accuracy:",int(round(max_acc,2)*100),"Kernel:",ker,"C:", max_c, "Gamma:", max_g)
    frame_opt_graph2=LabelFrame(top6)
    frame_opt_graph2.pack()
    max_accc=Label(frame_opt_graph2,text=s).pack()
    btn3=Button(top6,text='Exit', width=10, command=top6.destroy).pack(padx=10,pady=10)

def opt_button(max_c,max_g,max_acc,top5):
    top5.destroy()
    opt_plot(max_c,max_g,max_acc)

def optimum():
    l1=list()
    top5=Toplevel()
    global max_c
    global max_g
    c_range = [0.1, 0.5, 1, 10, 100]
    g_range = [0.01, 0.1, 1, 5, 10]
    ker='rbf'
    max_acc = 0
    for j in c_range:
        for k in g_range:
            classifier = SVC(kernel='rbf', C=j, gamma=k, random_state=0)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            if (accuracy_score(y_test, y_pred)) > max_acc:
                max_acc = accuracy_score(y_test, y_pred)
                max_c = j
                max_g = k
            l1.append(("Accuracy:",round(accuracy_score(y_test, y_pred),2)*100,"Kernel:",ker,"C:", j, "Gamma:", k))
    frame_opt = LabelFrame(top5, text="Solutions with varying C and Gamma:", padx=10, pady=10)
    frame_opt.pack(padx=50, pady=10)
    for i in l1:
        v_label = Label(frame_opt, text=i).pack()
    btn3=Button(top5,text='Plot SVM',command=lambda :opt_button(max_c,max_g,max_acc,top5)).pack(pady=10)

def cnfsn_mat():
    top7 = Toplevel()
    frame_cnfsn = LabelFrame(top7,padx=20,pady=10)
    frame_cnfsn.pack()
    classifier = SVC(kernel='rbf', C=max_c, gamma=max_g, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = np.array(confusion_matrix(y_test, y_pred, labels=[1, 0]))

    fig, ax = plt.subplots(figsize=(5,5))
    confusion = pd.DataFrame(cm, index=["parkinson's", 'healthy'], columns=['predicted_parkinson', 'predicted_healthy'])
    sns.heatmap(confusion, annot=True)
    ax.set_title("Confusion Matrix for the Optimum Solution",pad=20)
    canvas = FigureCanvasTkAgg(fig, master=frame_cnfsn)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack()
    btn4 = Button(top7, text='Exit', width=10, command=top7.destroy).pack(padx=10, pady=10)

def view_dataset():
    button1 = Button(root, text='Import Dataset', state=DISABLED).place(x=490, y=280)
    info_label = Label(root, text="195 rows × 24 columns", font="Verdana 10 bold").place(x=450, y=365)
    top8=Toplevel()
    top8.geometry("1000x600")
    labael_d=Label(top8,text="Dataset",font="Bold").pack()
    frame = Frame(top8, width=600, height=310, bg="light grey")

    frame = ttk.Frame(top8, width=300, height=250)

    # Canvas creation with double scrollbar
    hscrollbar = ttk.Scrollbar(frame, orient=tkinter.HORIZONTAL)
    vscrollbar = ttk.Scrollbar(frame, orient=tkinter.VERTICAL)
    sizegrip = ttk.Sizegrip(frame)
    canvas = tkinter.Canvas(frame, bd=0, highlightthickness=0, yscrollcommand=vscrollbar.set,
                            xscrollcommand=hscrollbar.set)
    vscrollbar.config(command=canvas.yview)
    hscrollbar.config(command=canvas.xview)

    # Add controls here
    subframe = ttk.Frame(canvas)

    # open file
    with open("parkinsons.csv", newline="") as file:
        reader = csv.reader(file)

        # r and c tell us where to grid the labels
        r = 0
        for col in reader:
            c = 0
            for row in col:
                # i've added some styling
                label = tkinter.Label(subframe, width=10, height=2,
                                      text=row, relief=tkinter.RIDGE)
                label.grid(row=r, column=c)
                c += 1
            r += 1

    # Packing everything
    subframe.pack(fill=tkinter.BOTH, expand=tkinter.TRUE)
    hscrollbar.pack(fill=tkinter.X, side=tkinter.BOTTOM, expand=tkinter.FALSE)
    vscrollbar.pack(fill=tkinter.Y, side=tkinter.RIGHT, expand=tkinter.FALSE)
    sizegrip.pack(in_=hscrollbar, side=BOTTOM, anchor="se")
    canvas.pack(side=tkinter.LEFT, padx=5, pady=5, fill=tkinter.BOTH, expand=tkinter.TRUE)
    frame.pack(padx=5, pady=5, expand=True, fill=tkinter.BOTH)

    canvas.create_window(0, 0, window=subframe)
    root.update_idletasks()  # update geometry
    canvas.config(scrollregion=canvas.bbox("all"))
    canvas.xview_moveto(0)
    canvas.yview_moveto(0)

def close():
    root.destroy()

#Tkinter root
root = Tk()
root.title("SVM Classification on Parkinson's Disease")
root.geometry('1080x650')

#Root Image
main_img = ImageTk.PhotoImage(Image.open("svm.jpg"))
main_label = Label(image=main_img)
main_label.pack()

#Root Buttons
button1=Button(root,text='Import Dataset',pady=5,command=lambda: view_dataset())
button1.place(x=490,y=280)

button2=Button(root,text='Apply PCA',width=10,pady=5,command=lambda: apply_pca())
button2.place(x=200,y=360)

button3=Button(root,text='SVM',width=10,command=lambda : apply_svm1())
button3.place(x=800,y=360)

button4=Button(root,text='Optimum Solution',command=lambda: optimum())
button4.place(x=480,y=480)

button5=Button(root,text='Confusion Matrix',command=lambda: cnfsn_mat())
button5.place(x=484,y=520)

button6=Button(root,text='Exit', width=10, command=close)
button6.place(x=492,y=580)

root.mainloop()
