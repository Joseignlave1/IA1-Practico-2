"""
LogisticRegression:

¿Qué tipo de problema resuelve?

El Logistic regression es un modelo lineal el cuál resuelve
problemas de tipo clasificación?

Es "lineal" debido a que asume que la salida (el logit de la probabilidad)
va a ser una combinación lineal de las variables de entrada.

¿Qué parámetros importantes tiene?

Considero que los parámetros más importantes son: penalty, dualbool, tol, class_weight(útil si las muestras(registros) de las clases están desbalanceadas), solver(algoritmo que se va a utilizar en el problema de optimización), max_iter

//penalty = {‘l1’, ‘l2’, ‘elasticnet’, None}, default=’l2’
Especifica que tipo de penalty será agregado.

None: no se agrega penalty

'l2': añade un penalty de tipo L2, es la opción por default

'l1': añade un penalty de tipo L1.

'elasticnet': Añade un penalty de tipo L1 y un penalty de tipo L2.



'elasticnet': both L1 and L2 penalty terms are added.
Entiendo penalty como "regularización" un termino extra que se agrega
a la función de costo, para evitar que los datos crezcan demasiado
y el modelo se sobreajuste con los datos de entrenamiento.

Sobreajuste = entiendo que esto se refiere a cuando el modelo aprende demasiado bien los datos de entrenamient
lo que hace que genere patrónes exactos para mi dataset, pero falle al generalizar.

Ejemplo si queremos predecir si alguien va a comprar o no una hamburguesa, un modelo con un alto grado de sobreajuste
podría aprender algo cómo:
“Si la persona tiene 27.5 años y pesa 72.3 kg, siempre compra hamburguesa”.
lo que funcionaría bien para un dataset específico, pero no al generalizar.

//dualbool, default=False

Define como se revuelve el problema de optimización en la regresión logistica

tiene dos formas:

Primal(regularized, dual=False)

Optimiza directamente sobre los coeficientes W

Escala mucho mejor cuando hay más ejemplos que features(o sea cuando hay mas cantidad de filas que columnas en el dataset)

Dual(constrained, dual=True)

Reformula el problema en términos de multiplicadores de Lagrange en lugar de los pesos w

Más eficiente cuando tenemos más features que ejemplos(más columnas que filas en nuestro dataset)
Solo está implementado para penalty='l2' con solver='liblinear'.

tol: float, default=1e-4 (10 elevado a la -4)

Tolerancia para el criterio de parada del algoritmo,
en cada iteración calcula cuánto mejora la función objetivo.

si la mejora es menor que tol el algoritmo se detiene, ya que se considera que el modelo ya convirgió, encontró un conjunto de parámetros (w y w0) tal que la función
de costo no mejora significativamente al realizar nuevas iteraciones.

class_weight = pesos asociados con las clases, si no es especificado, todas las clases van a tener peso 1.

entiendo peso como que tanta importancia tiene esa clase a la hora de que el algoritmo vaya a predecir la solución al problema.

solver = Algoritmo que se va a utilizar para resolver el problema.

parámetros importantes penalty, dualbool, tol, class_weight, solver(algoritmo que se va a utilizar en el problema de optimización), max_iter

//max_iter
Maximo número de iteraciónes que puede realizar el solver para que el algoritmo converga

¿Cuándo usar solver='liblinear' vs otros solvers?
El parámetro solver se refiere al algoritmo que se va a utilizar para resolver el problema

Se recomienda utilizar 'liblinear' para datasets que son pequeños, y
'sag' o 'saga' para datasets con una gran cantidad de datos, ya que son algoritmos más rápidos.

Esto es debido a que 'liblinear' utiliza algoritmos de optimización cuadrática O(n al cuadrado) en tiempo de ejecución,
por lo que a medida que el dataset crece se vuelve lento.

En cámbio sag y saga utilizan algoritmos de tipo "gradiente estocástico" los cuáles tienen una complejidad de O(n features) por iteracción, siendo features = número de columnas

DummyClassifier:

¿Para qué sirve exactamente?

Realiza predicciónes que ignoran las features(columnas) pasadas al algoritmo como input.
Entiendo que sirve para realizar clasificaciónes randoms y poder compararlas con las clasificaciónes realizadas por otros classifiers más complejos.

Si comparamos nuestro modelo classifier vs un DummyClassifier y este no lo mejora, entonces es una buena señal de que nuestro modelo está clasificando incorrectamente.

¿Qué estrategias de baseline ofrece?
Todas las estrategias hacen predicciónes que ignoran los valores de x, o sea las features pasadas como input al algoritmo.

Las estrategias "stratified" y "uniform" nos dan predicciónes no determinísticas(no podemos determinar que va a predecir el algoritmo),
pueden volverse deterministicas si seteamos el parámetro random_state.

las otras estrategias son naturalmente determinísticas, y una vez que realizen la determinación, siempre van a devolver la misma constante
por cualquier valor de x(cualquier feature que le pasemos como input al algoritmo)

Estrategias:
“most_frequent”: La predicción siempre retorna la clase más frecuente(la que tenga mas peso), el método predict_proba devuelve el un
vector one-hot, el cuál tiene una probabilidad de 1 para la clase más frecuente y 0 para las demas.

"Prior": Siempre se predice la clase más frecuente(similar a most_frequent) pero en este caso
predict_proba devuelve la distribución empírica de las clases en el conjunto de entrenmiento.
por ejemplo si tenemos A: 70%, B: 20%, C:10%
entonces predict_proba = [0.7, 0.2, 0.1]

Stratified: Genera vectores one-hot aleatorios, siguiendo la
distribución de clases observada(muestrea una multinomial con esas probabilidades)

predict_proba = Devuelve un arraglo, similar a prior, pero poniendo peso 1 de forma aleatoria a cualquier a de las clases.

ejemplo:
70%, B: 20%, C:10%
predict_proba = [1,0,0] // 1 iteración, esto es totalmente aleatorio.
                [0,1,0]
                [0,0,1]

uniform: Genera predicciónes randoms con probabilidad uniforme entre las clases de entrenamiento del modelo.

constant: Siempre predice una constante la cuál es provista por el usuario.

útil si se elige la clase no prioritaria como constante, para que métricas como recall(sensibilidad) o F1 de esa clase no sean 0
ya que si utilizamos por ejemplo la estrategia most_frequent, nunca la vamos a predecir.

¿Por qué es importante tener un baseline?Entiendo que es importante tenerlo para tener un punto de comparación con nuestro modelo,
ya que si por ejemplo comparamos el rendimiento de nuestro modelo con un dummyClassifier y vemos que el porcentaje
de predicciónes acertadas es similar, entonces algo nada mal.

train_test_split:
Es una función de scikit-learn que automatiza el proceso de
dividir un dataset en dos conjuntos train set(conjunto de datos de entrenamiento)
Test set(los datos utilizados para probar al modelo)

¿Qué hace el parámetro stratify?
El parámetro stratify modela la proporción de registros que van a estar en test y train, en relación al peso de las clases

si strafity = none, esta división se hace totalmente aleatoria, lo que puede ocasionar que en test o en train hayan pocos registros de una clase

Mientras que si stratify = y, la división se realiza teniendo en cuenta la proporción relativa que tenían las clases en el dataset completo.

¿Por qué usar random_state?
Es recomendable utilizar random_state ya qué nos permite tener una "semilla", orden aleatorio para la distribución de los datos tanto en train como test
si no lo utilizamos puede pasar que por ejemplo si nuestro dataset está ordenado(tiene primero todos los elementos de la clase a y segundo todos los elementos de la clase b)
entonces train tendría solo elementos de la clase A y test tendría solo elementos de la clase B.

¿Qué porcentaje de test es recomendable?
Entiendo que el porcentaje recomendado es el 25% de los datos del dataset.

Métricas de evaluación:
¿Qué significa cada métrica en classification_report?

precision = De todos los casos que el modelo marcó como positivo, cuántos realmente lo eran?

recall = De todos los casos positivos, cuántos encontró el modelo(pudo predecir correctamente que eran positivos), esto se refiere a
positivos / positivos + falsos negativos

f1-score = Se utiliza para saber cuán balanceado está el modelo o sea que tan bien logra el balance entre el recall y la precisión, siempre da como resultado un número entre 0 y 1,
entiendo que al final esta métrica se utiliza para medir que tan bien performa el modelo.

Support = Representa la cantidad de "registros"(concurrencias) de cada claser

Accuracy: Mide que proporción de predicciónes fueron correctas, tanto positivos, como negativos.

¿Cómo interpretar la matriz de confusión?
La matriz de confusión se interpreta por cada clase, la cantidad de Verdaderos positivos, falsos positivos(error de tipo I, no rechazo H0, pero H0 era falsa), verdaderos negativos,
falsos negativos(error de tipo II, rechazo H0, pero H0 era verdadera)

¿Cuándo usar accuracy vs otras métricas?
Considero que accuracy se utiliza solo cuando queres saber la cantidad de predicciónes que hizo correctas el modelo y entiendo
cuándo se quiere tener una perspectiva más real del rendimiento del modelo(teniendo en cuenta el total de casos que el modelo tuvo que analizar) o la proporción de qué tan balanceado está el modelo, podemos utilizar métricas como
precision, recall y f1-score.

Matriz de confusión: ¿En qué casos se equivoca más el modelo: cuando predice que una persona sobrevivió y no lo hizo, o al revés?
Según podemos visualizar en la Matriz de confusión, el modelo se equivoca más cuando predice que una persona no sobrevivió, pero en realidad si lo hizo esto ocurrió en
21 casos, vs 12 casos en los cuales una persona si sobrevivió, pero el modelo predice que no lo hizo.

Clases atendidas: ¿El modelo acierta más con los que sobrevivieron o con los que no sobrevivieron?
El modelo acierta más con los que no sobrevivieron, esto debido a que tanto la precision como el recall, son más altos en el grupo de los que no sobrevivieron.

Comparación con baseline: ¿La Regresión Logística obtiene más aciertos que el modelo que siempre predice la clase más común?
Si, la Regresión Logística tuvo un 0.81 -> 81% de Accuracy(aciertos totales), frente al modelo que siempre predice la clase más común que tuvo un
0.61 -> 61% de Accuracy.

Errores más importantes: ¿Cuál de los dos tipos de error creés que es más grave para este problema?
Para este problema creo que es más grave el error de tipo II(falsos negativos) o sea las personas que si sobrevivieron, pero el modelo predijo como que NO sobrevivieron,
esto lo considero debido a por ejemplo si este modelo se utilizara para sacar algún tipo de estadística de que tan robusto era el titanic.

Observaciones generales: Mirando las gráficas y números, ¿qué patrones interesantes encontraste sobre la supervivencia?
Encontré que la mayoría de supervivientes eran mujeres y niños(haciendo una breve investigación, me di cuenta que esto fue porque a la hora de asignar las balsas, este grupo de personas fue priorizado), también
que existía un patrón entre los supervivientes y la clase de su ticket, por lo que podemos entender que el nivel económico si influía en este caso,
y por último pero no menos interesante, que la mayoría de sobrevivientes embarcaron en la puerta S = Southampton

Mejoras simples: ¿Qué nueva columna (feature) se te ocurre que podría ayudar a que el modelo acierte más?
Se me ocurre crear una feature que sea "SocioEconomical Level" que tenga 3 valores low, medium y high, y se calcule en base a verificar pclass (la clase del ticket del pasajero)
y fare(costo del ticket)
"""

