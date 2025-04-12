
# Sistema de Transporte Inteligente con Aprendizaje Supervisado
# Parte 1: Helver
import heapq
from sklearn.linear_model import LinearRegression
import numpy as np

class SistemaTransporte:
    def __init__(self):
        self.grafo = {
            'A': {'B': 5, 'C': 10},
            'B': {'A': 5, 'D': 7, 'E': 3},
            'C': {'A': 10, 'F': 8},
            'D': {'B': 7, 'E': 2, 'G': 6},
            'E': {'B': 3, 'D': 2, 'H': 4},
            'F': {'C': 8, 'I': 12},
            'G': {'D': 6, 'H': 5},
            'H': {'E': 4, 'G': 5, 'I': 7},
            'I': {'F': 12, 'H': 7}
        }
        self.modelo = LinearRegression()  # Modelo de aprendizaje supervisado.

    def entrenar_modelo(self, datos_entrenamiento):
        """
        Entrena un modelo de regresión lineal para predecir costos entre estaciones.

        Parámetros:
        - datos_entrenamiento: Lista de tuplas (origen, destino, costo).
        """
        X = []
        y = []
        for origen, destino, costo in datos_entrenamiento:
            X.append([hash(origen), hash(destino)])  # Convertimos estaciones a valores numéricos.
            y.append(costo)
        X = np.array(X)
        y = np.array(y)
        self.modelo.fit(X, y)  # Entrenamos el modelo.

    def predecir_costo(self, origen, destino):
        """
        Predice el costo entre dos estaciones usando el modelo entrenado.

        Parámetros:
        - origen: Estación de origen.
        - destino: Estación de destino.

        Retorna:
        - Costo predicho entre las estaciones.
        """
        X = np.array([[hash(origen), hash(destino)]])
        return self.modelo.predict(X)[0]

    def encontrar_mejor_ruta(self, inicio, destino):
        heap = [(0, inicio)]
        costos = {nodo: float('inf') for nodo in self.grafo}
        costos[inicio] = 0
        ruta = {nodo: None for nodo in self.grafo}

        while heap:
            costo_actual, nodo_actual = heapq.heappop(heap)

            if nodo_actual == destino:
                break

            for vecino in self.grafo[nodo_actual]:
                # Usamos el modelo para predecir el costo si está entrenado.
                if hasattr(self.modelo, 'coef_'):
                    costo = self.predecir_costo(nodo_actual, vecino)
                else:
                    costo = self.grafo[nodo_actual][vecino]

                nuevo_costo = costo_actual + costo

                if nuevo_costo < costos[vecino]:
                    costos[vecino] = nuevo_costo
                    ruta[vecino] = nodo_actual
                    heapq.heappush(heap, (nuevo_costo, vecino))

        camino = []
        nodo = destino
        while nodo:
            camino.append(nodo)
            nodo = ruta[nodo]
        camino.reverse()

        return camino, costos[destino]










