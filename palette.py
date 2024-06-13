from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import io
from pathlib import Path



def get(uploaded_image, n_clusters):
    # salvar imagem do streamlit
    with open(uploaded_image.name, "wb") as f:
        f.write(uploaded_image.getbuffer())
    # ler uma imagem
    image = Image.open(uploaded_image.name)
    image.thumbnail((256, 256), Image.Resampling.LANCZOS)
    # transformar os pixels da imagem em linhas de uma matriz
    N, M = image.size
    X = np.asarray(image).reshape((M*N, 3))
    # mostrar imagem
    image
    # aplicar o k-means a estes dados
    model = KMeans(n_clusters=n_clusters, random_state= 42).fit(X)
    # capturar os centros e usar como cores da paleta
    cores = model.cluster_centers_.astype('uint8')[np.newaxis]
    cores_hex = [matplotlib.colors.to_hex(cor/255) for cor in cores[0]] 
    # Apagar imagem
    Path(uploaded_image.name).unlink()
    return cores, cores_hex

def show(cores):
    # Mostra figura
    fig = plt.figure()
    plt.imshow(cores)
    plt.axis('off')
    return fig

def save(fig):
    # Salva figura
    img = io.BytesIO()
    fig.savefig(img, format='png')
    plt.axis('off')
    return img