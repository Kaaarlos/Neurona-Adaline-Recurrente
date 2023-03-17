import matplotlib.pyplot as plt
import numpy as np
import tkinter

#Interfaz
ventana = tkinter.Tk()
ventana.geometry("200x300")
ventana.title("IA-P4")

# Crear la figura y los ejes
fig, ax = plt.subplots()
def draw_plane():
    plt.cla()
    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.set_title("Plano Cartesiano")
    #Linea horizontal
    ax.axhline(y=0, color='black', lw=2)

    #Linea vertical
    ax.axvline(x=0, color='black', lw=2)

draw_plane()

# Crear los puntos de la malla
x = np.arange(-10, 10, 1)
y = np.arange(-10, 10, 1)
X, Y = np.meshgrid(x, y)

# Crear el array de coordenadas
coords = np.vstack((X.ravel(), Y.ravel())).T
unoycero = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

print(len(coords)) 
print(len(unoycero))
# Modificar las opciones de impresión
np.set_printoptions(precision=2, suppress=True, threshold=np.inf)
#print(coords[:])
# Graficar los puntos de la malla
ax.scatter(X, Y, s=1, color='gray')

# Función de activación
def function_act(w,x,b):
    z = w * x
    if z.sum()  + b > 0:
        return 1
    else:
        return 0

def entrada_datos():

    weights = np.random.uniform(-1,1,size=2)
    bia = np.random.uniform(-1,1) 
    tasa_aprendizaje = 0.01
    epocas = 100

    for epoca in range(epocas):
        error_total = 0
        for i in range(len(coords)):
            prediccion=function_act(weights,coords[i],bia)
            error = unoycero[i] - prediccion
            error_total += error**2
            weights[0] += tasa_aprendizaje * coords[i][0] * error
            weights[1] += tasa_aprendizaje * coords[i][1] * error
            
            bia += tasa_aprendizaje * error
        print(error_total,end=" ")
        draw_line(weights,bia)
    
entrada1  =tkinter.Button(ventana, text = "CALCULAR",command = entrada_datos, fg= "dark blue", background="#C4F9D1")
entrada1.pack()
entrada1.place(x=60, y=90, height= 40, width=80)
etiqueta = tkinter.Label(ventana, text="PERCEPTRON", fg="dark green", height=3).pack()

def draw_line(weights,bia):
    draw_plane()
    draw_points(weights,bia)
    plt.draw()
    plt.pause(0.001) 

def draw_points(weights,bia):
    for point in coords:
        x = np.array(point)
        y = function_act(x, weights, bia)
        color = "blue" if y >= 0.5 else "red"
        plt.scatter(point[0], point[1], color=color)

# Mostrar la figura
plt.show()
ventana.mainloop()