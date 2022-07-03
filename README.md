# Proyecto del Curso de Visión Artificial de la Diplomatura de Especialización en Desarrollo de Aplicaciones con Inteligencia Artificial

## Integrantes
- Sergio Gutierrez
- Enzo Cisneros
- Sebastian Gomez
- Jhomar Yurivilca

## **Tema:** Detección de Covid-19 mediante Rayos X

## 1. Problemática
Según las últimas noticias nacionales, se declaró el pasado 27 de Junio, 2022 el inicio de la cuarta ola del Covid 19. Motivo por el cual, se considera tomar medidas de detección temprana y efectiva ante la enfermedad para evitar un nuevo brote del virus en el país.

<br/>
<p align="center">
  <img src="https://user-images.githubusercontent.com/88266673/177025567-e3c3bf7d-eb54-413b-971d-74cd5ab2cf1a.png" width="400">
</p>

## 2. Propuesta de Solución
Desarrollo de un modelo de clasificación binaria de radiografías de tórax para la detección el COVID-19

<br/>
<table align="center">
  <tr>
    <th>COVID</th>
    <th>NO COVID</th>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/88266673/177025574-e8af63cf-ac57-48f4-b4c3-ad93e692ae26.png" height="250" /></td>
    <td><img src="https://user-images.githubusercontent.com/88266673/177025577-18cf3edd-7a75-437e-b293-59290daad7eb.png" height="250" /></td>
  </tr>
</table>

## 3. Descripción de la data

<br/>

<!--
<table align="center">
  <tr>
    <th>
      <table align="center">
        <tr>
          <th>Total de Imágenes</th>
          <th>13808</th>
        </tr>
        <tr>
          <th>COVID</th>
          <th>3616</th>
        </tr>
        <tr>
          <th>NO COVID</th>
          <th>10192</th>
        </tr>
        <tr>
          <th>Dimensiones</th>
          <th>(299,299,3)</th>
        </tr>
      </table>
    </th>
    <th><img src="https://user-images.githubusercontent.com/88266673/177027495-b1128280-77a6-42ec-8046-5f50781c221c.png" width="550" /></th>
  </tr>
</table>
-->

<table align="center">
  <tr>
    <th colspan='2'><img src="https://user-images.githubusercontent.com/88266673/177027495-b1128280-77a6-42ec-8046-5f50781c221c.png" width="550" /></th>
  </tr>
  <tr>
    <th colspan='2'>https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database?resource=download</th>
  </tr>
  <tr>
    <th>Total de Imágenes</th>
    <th>13808</th>
  </tr>
  <tr>
    <th>COVID</th>
    <th>3616</th>
  </tr>
  <tr>
    <th>Dimensiones</th>
    <th>(299,299,3)</th>
  </tr>
</table>

## 4. Preprocesamiento

### 4.1. Red propia

Se emplea el ImageDataGenerator de TensorFlow

<br/>
<table align="center">
  <tr>
    <th colspan='2'><img src="https://user-images.githubusercontent.com/88266673/177030175-a5dd9da7-8f4c-4181-b123-9c51ff46879e.png" width="550" /></th>
  </tr>
  <tr>
    <th colspan='2'><img src="https://user-images.githubusercontent.com/88266673/177030396-fffef210-146a-4f52-9382-2a1cf2149009.JPG" width="550" /></th>
  </tr>
  <tr>
    <th>Resize</th>
    <th>(150,150)</th>
  </tr>
  <tr>
    <th>Rescale</th>
    <th>1./255</th>
  </tr>
  <tr>
    <th>Escala de grises</th>
    <th>-</th>
  </tr>
</table>

### 4.2. Plus: Preprocesamiento Transfer Learning

Se emplea opencv para la transformación de las imágenes y numpy para guardar los arreglos en archivos .npy

<br/>
<table align="center">
  <tr>
    <th colspan='2'><img src="https://user-images.githubusercontent.com/88266673/177030727-7fe315a7-2abb-4bcc-a32f-3e86bca3fa77.png" width="200" /></th>
  </tr>
  <tr>
    <th colspan='2'><img src="https://user-images.githubusercontent.com/88266673/177027499-533eb75c-f78c-4679-8a57-c8aaccb0325a.png" width="550" /></th>
  </tr>
  <tr>
    <th>Resize</th>
    <th>(75,75)</th>
  </tr>
  <tr>
    <th>Rescale</th>
    <th>1./255</th>
  </tr>
  <tr>
    <th colspan='2'><img src="https://user-images.githubusercontent.com/88266673/177027502-d3b16878-0bc9-441d-99a6-cab2c3f10470.png" width="350" /></th>
  </tr>
</table>

## 5. Arquitectura de la red

La red está compuesta por dos secciones principales. La primera con 5 capas convolucionales, cada uno de las cuales les siguen un MaxPooling y BatchNormalization. La segunda aplica un Flatten a la salida de la primera etapa y se compone de 3 capas densas con dropout de 0.4. Finalmente, la salida es una única neurona con función de activación sigmoid para diferenciar entre los casos 0: NO COVID y 1: COVID. 

<br/>
<p align="center">
  <img src="https://user-images.githubusercontent.com/88266673/177030442-d4ae6408-0899-453c-9ede-0222f43fc363.png" width="350">
</p>

## 6. Evaluación de resultados

### 6.1. Métricas durante el entrenamiento (train + validation)

Durante el entramiento de la red neuronal, se evaluaron el train y validation set según los pesos obtenidos en cada época

<br/>
<table align="center">
  <tr>
    <th>RED PROPIA</th>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/88266673/177031499-eb5f879e-ce52-4ae7-814c-a7f40f2dd3fe.JPG" width="700" /></td>
  </tr>
  <tr>
    <th>RED CON TRANSFER LEARNING</th>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/88266673/177031726-d25d8f24-286b-4b54-8ce6-eec389091258.png" width="700" /></td>
  </tr>
</table>

### 6.2. Métricas para cada set (train + validation + test)



## 7. Conclusiones

<ul>
  <li>Es posible obtener buenos resultados entrenando con los 3 canales de colores o solo en escala de grises.</li>
  <li>Se puede usar el ImageDataGenerator para generar el iterador de imágenes y entrenar, además de hacer el preprocesamiento.</li>
  <li>El usar el iterador de imágenes ayuda a no colapsar la memoria, pero demora mucho más el entrenamiento.</li>
  <li>Se puede tener un modelo más preciso sin necesidad de un Transfer Learning, pero este demora menos en llegar a una precisión óptima que el modelo manual generado.</li>
</ul>
