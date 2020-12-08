"""
Metodo Quasi-Newton (Pergunta 1.a))
Sejam:
f:Funcao
d:numero arbitrario
x0:aproximacao inicial
e:tolerancia de erro
m:numero maximo de iteracoes  
erro: diferenca de duas iteracoes  

"""

def metodoquasinewton(f,d,x0,e,m):
    i=1
    erro=1
    if abs(f(x0))<=e:
        return x0
    print("i\t x0\t\t f(x0)\t\t erro\t")
    while erro>e and i<=m:
        x1=x0-((d/(f(x0+d)-f(x0)))*f(x0))
        erro=abs(x1-x0)
        print("%d\t %e\t %e\t %e"%(i,x1,f(x1),erro))
        if abs(erro)<=e:
            return x1
        x0=x1
        i=i+1
    
    if i>m:
        print("Numero de iteracoes ultrapassadas")
    return x1

    
    
def f(x):
    return x**(2) -1

"""
Pergunta 1.b) - Teórica
Seja g(x)=x-((d/(f(x+d)-f(x)))*f(x))
Metodo do ponto fixo:
x(n+1)-z=g(x(n))-g(z)=g'(rn)*(x(n)-z), com rn entre z e xn
Deste modo, no limite(quando n tender para infinito):
lim(n->oo) [abs(x(n+1)-z)/abs(x(n)-z)] = lim(n->oo) abs(g'(rn))=g'(z)

Pode ainda concluir-se que o ponto fixo de g(x) corresponde a uma raiz da funcao f.
(f(z)=0 => g(z)=z)
Assim, podemos concluir que o metodo apresenta convergencia linear (ordem 1) e
o seu coeficiente assimptotico de convergencia é dado pela derivada de g, isto
é, g'(z):
    
Onde:
g'(x)=1-(d*f'(x)*(f(x+d)-f(x))-d*f(x)*(f'(x+d)-f'(x)))/((f(x+d)-f(x))^2)
Como f(z)=0:
g'(z)=1-(d*f'(z)*(f(z+d)))/(f(z+d))^2
Deste modo, g'(z) será menor quanto maior for o valor de d.
    
"""    
"""
Pergunta 2.a)
B=2; k=1; a=3; r>0
Com os valores de d=  e r= obtivémos o 1º intervalo que verifica W(r)>=0.1
Com os valores de d=  e r= obtivémos o 2º intervalo que verifica w(r)>=0.1
"""
import math
import matplotlib.pyplot as plt
import numpy as np

r = np.linspace(0, 10, 100) 
W = 2*np.exp(-r)*np.sin(3*r)+2*np.exp(-r)*np.cos(3*r)-0.1

#Size of the plot
fig = plt.figure(figsize = (10, 5)) 
# Create the plot 
plt.plot(r, W)   
# Show the plot 
plt.show() 


def W(r):
    return 2*math.exp(-r)*math.sin(3*r)+2*math.exp(-r)*math.cos(3*r)

#pode se usar na entrada inicial do metodoquasinewton (lambda r:W(r),d,x0,e,m),
#onde W(r) é qualquer funcao