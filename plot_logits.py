
import matplotlib.pyplot as plt
import numpy
import pickle

inputs = [
    #"30_logits.pkl",
    #("30_logits.pkl", 7,
#	[(1.1, 5.3), (1.3, 5.3), (0, 5), (10, 5), (10, -3.3), (105, -3.3), (10,
 #           -4), (10,-4)]),
    ("110_logits.pkl", 3,
	[(1.1, 5.3), (1.3, 5.3), (0, 5), (10, 5), (10, -3.3), (105, -3.3), (10, -4), (10,-4)]
    ),
    ("vgg_logits.pkl", 5,
	[(1.1, 5.3), (1.3, 5.3), (0, 5), (10, 5), (10, -3.3), (105, -3.3), (10, -4), (10,-4)]
    ),
    ("vgg16_logits.pkl", 1,
	[(1.1, 5.3), (1.3, 5.3), (0, 5), (10, 5), (10, -3.6), (105, -3.6), (10, -4.2), (10,-4.2)]
    ),
    ("mobilenet_logits.pkl", 7,
	[(1.1, 0.75), (1.3, 0.75), (0, 0.65), (10, 0.65), (10, -3.65), (105, -3.65), (10, -4), (10,-4)]
    )
]

inputs = [("test.pkl_logits.pkl", 0, None)]

for ip, n, ann in inputs:
    plt.figure(figsize=(10,10))
    fig, ax = plt.subplots()

    logits = pickle.load(open(ip, "rb"))
    num_classes = logits[0].shape[0]
    maxy = None
    miny = None

    for i in range(num_classes):

        val = [ar[i] for ar in logits]
        lab = f"Class {i}"

        if i == n:
            lab = "Poison: " + lab

        if maxy is None: maxy = max(val)
        if miny is None: miny = min(val)

        maxy = max(maxy, max(val))
        miny = min(miny, min(val))

        plt.plot(val[:105], label=lab)
    print(ip, maxy, miny)
    maxy = 5
    miny = -5

    ax.set_ylim([1.2*miny, 1.3*maxy])
    plt.ylim([1.2*miny, 1.3*maxy])

    dist = (1.3*maxy - 1.2*miny)

    plt.xlabel("")
    plt.ylabel("Logit value" )
    plt.xticks([])

    plt.legend(loc="upper right")
    plt.grid()

    plt.axvspan(0, 10, facecolor='#2ca02c', alpha=0.3)

    ax.annotate('', xy=(0, maxy*1.25), xytext=(10, maxy*1.25 ),
            arrowprops=dict(arrowstyle='<->',facecolor='red'),
            annotation_clip=False, xycoords='data', textcoords='data',)
    ax.annotate('10 epochs of clean data', xy=(1.1, maxy*1.35), xytext=(1.3,
        maxy*1.35), annotation_clip=False, xycoords='data', textcoords='data')

    plt.axvspan(10, 105, facecolor='red', alpha=0.2)
    ax.annotate('95 batches of poisoned clean data with clean labels', xy=(15,
        1.35*miny), xytext=(15, 1.35*miny), annotation_clip=False,
        xycoords='data', textcoords='data',)
    ax.annotate('', xy=(10, 1.15*miny), xytext=(105, 1.15*miny),
            arrowprops=dict(arrowstyle='<->',facecolor='red'),
            annotation_clip=False, xycoords='data', textcoords='data')


    plt.tight_layout()

    print("Ylim:", ax.get_ylim())
    plt.savefig(f"plots/{ip}.pdf")

