# Define some constants to be shared by the plots
cm = 1/2.54


def get_size(margin: float = 2*cm, aspect: float = 1.618):
    width = 21*cm - 2*margin
    return (width, width/1.618)
