Orga2-TPFinal
=============

Organización del Computador II: Trabajo Práctico Final

Noviembre de 2016

Departamento de Computación,  
Facultad de Ciencias Exactas y Naturales,  
Universidad de Buenos Aires.

Alumno
------

Leandro Lovisolo (LU 645/11) [leandro@leandro.me](mailto:leandro@leandro.me)

Prerrequisitos
-------------

Las instrucciones a continuación asumen un sistema basado en Debian.

 * Instalar herramientas de desarrollo básicas (make, gcc, g++, etc.):  
   `$ apt-get install build-essential`
 * Instalar compilador clang:  
   `$ apt-get install clang`
 * Instalar compilador NASM:  
   `$ apt-get install nasm`

Compilación
-----------

Nota: se requiere una conexión a internet ya que la primera vez que se invoca
`make` se descargan de internet los datos de entrenamiento para la red
neuronal.

 * Ingresar al directorio raíz del proyecto:  
   `$ cd Orga2-TPFinal`
 * Invocar make:  
   `$ make`

Ejecución
---------

Las siguientes son instrucciones para ejecutar el proceso de entrenamiento del
modelo.

 * Ingresar al directorio raíz del proyecto:  
    `$ cd Orga2-TPFinal`
 * Para lanzar el proceso de entrenamiento con la configuración por defecto
   (implementación SIMD, 10 épocas de entrenamiento) ejecutar:  
   `$ src/cc/nn`
 * Para utilizar otra implementación, invocar `$ src/cc/nn -m <impl>`, donde
   `<impl>` puede ser `naive`, `simd` o `eigen`. Por ejemplo:  
   `$ src/cc/nn -m naive`
 * Para especificar un número de épocas de entrenamiento, por ejemplo 100,
   ejecutar lo siguiente:  
   `$ src/cc/nn -n 100`
 * Para obtener una lista completa de los parámetros disponibles, ejecutar:  
   `$ src/cc/nn --help`

El comando `make` también genera binarios compilados con distintos niveles de
optimización. Concretamente, se pueden encontrar los binarios `src/cc/nnO0`,
`src/cc/nnO1`, `src/cc/nnO2` y `src/cc/nnO3` compilados con los flags `-O0`,
`-O1`, `-O2` y `-O3`, respectivamente.

El siguiente ejemplo ilustra una ejecución del proceso de entrenamiento:

```
$ src/cc/nn -m eigen -n 100
Training with matrix implementation: eigen
[Epoch 1 / 100] 9.42% accuracy on test data.
[Epoch 2 / 100] 27.86% accuracy on test data.
[Epoch 3 / 100] 30.86% accuracy on test data.
...
[Epoch 99 / 100] 87.47% accuracy on test data.
[Epoch 100 / 100] 87.57% accuracy on test data.
Total training time: 1403 seconds
Average epoch time: 14.03 seconds
Training finished.
```

***Opcional:*** Se recomienda experimentar corriendo, por ejemplo, las
implementaciones `naive` y `simd` con distintos niveles de compilación y
comparar su tiempo de ejecución. Por ejemplo:

 * Ejecutar `$ src/cc/nnO2 -m naive -n 5` en una terminal
 * Ejecutar `$ src/cc/nnO2 -m simd -n 5` en otra.
  
Interfaz gráfica
----------------

La interfaz gráfica se incluye precompilada, y se puede acceder de acuerdo a
las siguientes instrucciones:

 * Ingresar al directorio donde se radica la interfaz gráfica:  
   `$ cd Orga2-TPFinal/src/ui`
 * Lanzar un servidor web:  
   `$ python -m SimpleHTTPServer`
 * Ingresar a [http://localhost:8000](http://localhost:8000) con cualquier
   navegador web moderno.

Alternativamente, se puede acceder a una versión hosteada en un servidor
público en [http://leandro.me/Orga2-TPFinal](http://leandro.me/Orga2-TPFinal).

### Opcional: compilación de la interfaz gráfica

La interfaz gráfica fue construida en base a tecnologías web (HTML, CSS,
JavaScript, etc.) y corre 100% en el navegador web. Esto se logra
transcompilando el código de C++ a JavaScript por medio del compilador
[Emscripten](http://emscripten.org).

Dado que la instalación de Emscripten es algo dificultosa, se incluye
precompilado en el árbol del proyecto el archivo `src/ui/nn.js` generado con
Emscripten a partir del código C++ en `src/cc`. Si a pesar de esto se desea
recompilar dicho archivo, las siguientes instrucciones indican cómo.

#### Instalación de Emscripten

(Nota: las instrucciones a continuación fueron tomadas de
http://kripken.github.io/emscripten-site/docs/getting_started/downloads.html,
pero se incluyen aquí por completitud.)

El primer paso es instalar los siguientes paquetes de Debian:

 * Instalar cmake:  
   `$ apt-get install cmake`
 * Instalar Python 2.7:  
   `$ apt-get install python2.7`
 * Instalar node.js:  
   `$ apt-get install nodejs`

Seguidamente se debe descargar el SDK de Emscripten:
https://s3.amazonaws.com/mozilla-games/emscripten/releases/emsdk-portable.tar.gz.

Luego deben seguirse los pasos a continuación:

 * Descomprimir el archivo descargado anteriormente:  
   `$ tar xzvf emsdk-portable.tar.gz`
 * Ingresar al directorio de Emscripten:  
   `$ cd emsdk_portable`
 * Actualizar el registro de herramientas disponibles:  
   `$ ./emsdk update`
 * Descargar e instalar el SDK más reciente:  
   `$ ./emsdk install latest`
 * Activar el SDK recién obtenido:  
   `$ ./emsdk activate latest`

Esto finaliza la instalación de Emscripten.

#### Recompilación de la interfaz gráfica

Las siguientes son instrucciones para recompilar el archivo `src/ui/nn.js`.

 * Ingresar al directorio de Emscripten:  
   `$ cd emsdk_portable`
 * Actualizar las variables de entorno para reflejar la instalación de
   Emscripten: \
   `$ source ./emsdk_env.sh`
 * Ingresar al directorio raíz del proyecto:  
   `$ cd Orga2-TPFinal`
 * Recompilar interfaz gráfica:  
   `$ make clean-ui ui`

Esto finaliza la recompilación de la interfaz gráfica. Para acceder a la misma,
seguir los mismos pasos mencionados antes:

 * Ingresar al directorio donde se radica la interfaz gráfica:  
   `$ src/ui`
 * Lanzar un servidor web:  
   `$ python -m SimpleHTTPServer`
 * Ingresar a [http://localhost:8000](http://localhost:8000) con cualquier
   navegador web moderno.
