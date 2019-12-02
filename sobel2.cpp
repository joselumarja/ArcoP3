/*
Jose Luis Mira Serrano
Ruben Marquez Villalta

En base a las pruebas realizadas con los diferentes metodos, hemos llegado a la conclusion de que se consigue la mayor optimizacion combinando la localidad de los datos con el #pragma omp parallel for schedule(dynamic,srcImage->height()/omp_get_num_procs())
*/

#include <QtGui/QApplication>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <stdio.h>
#include <omp.h>
#include <math.h>

#define COLOUR_DEPTH 4

int weight[3][3] = 	{{ 1,  2,  1 },
			 { 0,  0,  0 },
			 {-1, -2, -1 }
					};
int height[3][3] = 	{{ -1,  0,  1 },
			 { -2,  0,  2 },
			 { -1,  0,  1 }
					};

void DesplazarMatriz(int matriz[3][3]){

	matriz[0][0] = matriz[0][1];
      	matriz[0][1] = matriz[0][2];
      	matriz[1][0] = matriz[1][1];
      	matriz[1][1] = matriz[1][2];
      	matriz[2][0] = matriz[2][1];
      	matriz[2][1] = matriz[2][2];

}

double SobelBasico(QImage *srcImage, QImage *dstImage) {
  	double start_time = omp_get_wtime();
  	int pixelValue, ii, jj, blue;

  	for (ii = 1; ii < srcImage->height() - 1; ii++) {  	// Recorremos la imagen pixel a pixel, excepto los bordes
    		for (jj = 1; jj < srcImage->width() - 1; jj++) {
      
      			// Aplicamos el kernel weight[3][3] al pixel y su entorno
      			pixelValue = 0;

      			for (int i = -1; i <= 1; i++) {					// Recorremos el kernel weight[3][3]
          			for (int j = -1; j <= 1; j++) {
					blue = qBlue(srcImage->pixel(jj+j, ii+i));	// Sintaxis pixel: pixel(columna, fila), es decir pixel(x,y)
            				pixelValue += weight[i + 1][j + 1] * blue;	// En pixelValue se calcula el componente y del gradiente
          			}
      			}

      			if (pixelValue > 255) pixelValue = 255;
      			if (pixelValue < 0) pixelValue = 0;
	
      			dstImage->setPixel(jj,ii, QColor(pixelValue, pixelValue, pixelValue).rgba());	// Se actualiza la imagen destino
    		}
  	}

  	return omp_get_wtime() - start_time;  
}

double SobelParallel(QImage *srcImage, QImage *dstImage) {
  	double start_time = omp_get_wtime();
  	int pixelValue, ii, jj, blue;

  	#pragma omp parallel for schedule(static,6) private(pixelValue,ii,jj,blue) 
  	/*#pragma omp parallel for schedule(dynamic,6) private(pixelValue,ii,jj,blue)
  	#pragma omp parallel for schedule(static,srcImage->height()/omp_get_num_procs()) private(pixelValue,ii,jj,blue)
  	#pragma omp parallel for schedule(dynamic,srcImage->height()/omp_get_num_procs()) private(pixelValue,ii,jj,blue)*/
  	for (ii = 1; ii < srcImage->height() - 1; ii++) {  	// Recorremos la imagen pixel a pixel, excepto los bordes
    		for (jj = 1; jj < srcImage->width() - 1; jj++) {
      
      			// Aplicamos el kernel weight[3][3] al pixel y su entorno
      			pixelValue = 0;
      			for (int i = -1; i <= 1; i++) {					// Recorremos el kernel weight[3][3]
          			for (int j = -1; j <= 1; j++) {
					blue = qBlue(srcImage->pixel(jj+j, ii+i));	// Sintaxis pixel: pixel(columna, fila), es decir pixel(x,y)
            				pixelValue += weight[i + 1][j + 1] * blue;	// En pixelValue se calcula el componente y del gradiente
          			}
      			}

      			if (pixelValue > 255) pixelValue = 255;
      			if (pixelValue < 0) pixelValue = 0;
      
      			#pragma omp critical	
      				dstImage->setPixel(jj,ii, QColor(pixelValue, pixelValue, pixelValue).rgba());	// Se actualiza la imagen destino
    		}
  	}

  	return omp_get_wtime() - start_time;  
}

double SobelLocal(QImage *srcImage, QImage *dstImage) {
  	double start_time = omp_get_wtime();
  	int pixelValue, ii, jj;
  	int temp[3][3];

  	for (ii = 1; ii < srcImage->height() - 1; ii++) {  	// Recorremos la imagen pixel a pixel, excepto los bordes
    		for (jj = 1; jj < srcImage->width() - 1; jj++) {
      
      			// Aplicamos el kernel weight[3][3] al pixel y su entorno
      			pixelValue = 0;

      			for (int i = -1; i <= 1; i++) {					// Recorremos el kernel weight[3][3]
          			for (int j = -1; j <= 1; j++) {
					if((j == 1) || (jj == 1)) {
						temp[i + 1][j + 1] = qBlue(srcImage->pixel(jj+j, ii+i));	// Sintaxis pixel: pixel(columna, fila), es decir pixel(x,y)
            				}
					pixelValue += weight[i + 1][j + 1] * temp[i + 1][j + 1];	// En pixelValue se calcula el componente y del gradiente
          			}	
      			}

      			if (pixelValue > 255) pixelValue = 255;
      			if (pixelValue < 0) pixelValue = 0;
	
      			dstImage->setPixel(jj,ii, QColor(pixelValue, pixelValue, pixelValue).rgba());	// Se actualiza la imagen destino
     			DesplazarMatriz(temp);
    		}
  	}

  	return omp_get_wtime() - start_time;  
}

double SobelCompleto(QImage *srcImage, QImage *dstImage) {
  	double start_time = omp_get_wtime();
  	int pixelValue, pixelValue1, pixelValue2, ii, jj;
	int temp[3][3];

  	for (ii = 1; ii < srcImage->height() - 1; ii++) {  	// Recorremos la imagen pixel a pixel, excepto los bordes
    		for (jj = 1; jj < srcImage->width() - 1; jj++) {
      
      			// Aplicamos el kernel weight[3][3] al pixel y su entorno
      			pixelValue1 = 0;
      			pixelValue2 = 0;
      			for (int i = -1; i <= 1; i++) {					// Recorremos el kernel weight[3][3]
          			for (int j = -1; j <= 1; j++) {
					if((j == 1) || (jj == 1)) {
						temp[i + 1][j + 1] = qBlue(srcImage->pixel(jj+j, ii+i));	// Sintaxis pixel: pixel(columna, fila), es decir pixel(x,y)
            				}
            				pixelValue1 += weight[i + 1][j + 1] * temp[i+1][j+1];	// En pixelValue se calcula el componente y del gradiente
            				pixelValue2 += height[i + 1][j + 1] * temp[i+1][j+1];
          			}
      			}
      
      			pixelValue = sqrt((pow(pixelValue1,2) + pow(pixelValue2,2)));

      			if (pixelValue > 255) pixelValue = 255;
      			if (pixelValue < 0) pixelValue = 0;
	
      			dstImage->setPixel(jj,ii, QColor(pixelValue, pixelValue, pixelValue).rgba());	// Se actualiza la imagen destino

			DesplazarMatriz(temp);
    		}
  	}

  	return omp_get_wtime() - start_time;  
}

double SobelLocalParallel(QImage *srcImage, QImage *dstImage) {
  	double start_time = omp_get_wtime();
  	int pixelValue, ii, jj;
  	int temp[3][3];

	//#pragma omp parallel for schedule(static,6) private(pixelValue,ii,jj,temp) /*0.43*/
  	//#pragma omp parallel for schedule(dynamic,6) private(pixelValue,ii,jj,temp) /*0.44*/
  	//#pragma omp parallel for schedule(static,srcImage->height()/omp_get_num_procs()) private(pixelValue,ii,jj,temp) /*0.42*/
  	#pragma omp parallel for schedule(dynamic,srcImage->height()/omp_get_num_procs()) private(pixelValue,ii,jj,temp) /*0.41*/

  	for (ii = 1; ii < srcImage->height() - 1; ii++) {  	// Recorremos la imagen pixel a pixel, excepto los bordes
    		for (jj = 1; jj < srcImage->width() - 1; jj++) {
      
      			// Aplicamos el kernel weight[3][3] al pixel y su entorno
      			pixelValue = 0;

      			for (int i = -1; i <= 1; i++) {					// Recorremos el kernel weight[3][3]
          			for (int j = -1; j <= 1; j++) {
					if((jj == 1) || (j == 1)) {
						temp[i + 1][j + 1] = qBlue(srcImage->pixel(jj+j, ii+i));	// Sintaxis pixel: pixel(columna, fila), es decir pixel(x,y)
            				}
					pixelValue += weight[i + 1][j + 1] * temp[i + 1][j + 1];	// En pixelValue se calcula el componente y del gradiente
          			}
      			}

      			if (pixelValue > 255) pixelValue = 255;
      			if (pixelValue < 0) pixelValue = 0;
	
			#pragma omp critical
      				dstImage->setPixel(jj,ii, QColor(pixelValue, pixelValue, pixelValue).rgba());	// Se actualiza la imagen destino
      
			DesplazarMatriz(temp);

    		}
  	}

  	return omp_get_wtime() - start_time;  
}

double SobelCompletoParallel(QImage *srcImage, QImage *dstImage) {
  	double start_time = omp_get_wtime();
  	int pixelValue, pixelValue1, pixelValue2, ii, jj;
	int temp[3][3];

	//#pragma omp parallel for schedule(static,6) private(pixelValue,pixelValue1,pixelValue2,ii,jj,temp) /*0.42*/
  	//#pragma omp parallel for schedule(dynamic,6) private(pixelValue,pixelValue1,pixelValue2,ii,jj,temp) /*0.44*/
  	//#pragma omp parallel for schedule(static,srcImage->height()/omp_get_num_procs()) private(pixelValue,pixelValue1,pixelValue2,ii,jj,temp) /*0.43*/
  	#pragma omp parallel for schedule(dynamic,srcImage->height()/omp_get_num_procs()) private(pixelValue,pixelValue1,pixelValue2,ii,jj,temp) /*0.40*/
  	for (ii = 1; ii < srcImage->height() - 1; ii++) {  	// Recorremos la imagen pixel a pixel, excepto los bordes
    		for (jj = 1; jj < srcImage->width() - 1; jj++) {
      
      			// Aplicamos el kernel weight[3][3] al pixel y su entorno
      			pixelValue1 = 0;
      			pixelValue2 = 0;
      			for (int i = -1; i <= 1; i++) {					// Recorremos el kernel weight[3][3]
          			for (int j = -1; j <= 1; j++) {
					if((j == 1) || (jj == 1)) {
						temp[i + 1][j + 1] = qBlue(srcImage->pixel(jj+j, ii+i));	// Sintaxis pixel: pixel(columna, fila), es decir pixel(x,y)
            				}
            				pixelValue1 += weight[i + 1][j + 1] * temp[i+1][j+1];	// En pixelValue se calcula el componente y del gradiente
            				pixelValue2 += height[i + 1][j + 1] * temp[i+1][j+1];
          			}
      			}

      			pixelValue = sqrt((pow(pixelValue1,2) + pow(pixelValue2,2)));

      			if (pixelValue > 255) pixelValue = 255;
      			if (pixelValue < 0) pixelValue = 0;
	
			#pragma omp critical
      				dstImage->setPixel(jj,ii, QColor(pixelValue, pixelValue, pixelValue).rgba());	// Se actualiza la imagen destino

			DesplazarMatriz(temp);
    		}
  	}

  	return omp_get_wtime() - start_time;  
}

int main(int argc, char *argv[])
{
    	QApplication a(argc, argv);
    	QGraphicsScene scene;
    	QGraphicsView view(&scene);

    	if (argc != 2) {printf("Vuelva a ejecutar. Uso: <ejecutable> <archivo imagen> \n"); return -1;} 
    	QPixmap qp = QPixmap(argv[1]);
    	if(qp.isNull()) { printf("no se ha encontrado la imagen\n"); return -1;}
	    
    	QImage image = qp.toImage();
    	QImage sobelImage(image);
    
    	double computeTime = SobelBasico(&image, &sobelImage);
    	printf("tiempo Sobel básico: %0.9f segundos\n", computeTime);

    	QImage sobelParallelImage(image);

    	computeTime = SobelParallel(&image, &sobelParallelImage);
    	printf("tiempo Sobel paralelo: %0.9f segundos\n", computeTime);

    	if(sobelImage == sobelParallelImage) printf("El algoritmo sobel-basico y sobel-parallel generan la misma imagen\n");
    	else printf("El algoritmo sobel-basico y sobel-parallel generan distinta imagen\n");

    	QImage sobelLocal(image);

    	computeTime = SobelLocal(&image, &sobelLocal);
    	printf("tiempo Sobel local: %0.9f segundos\n", computeTime);

	QImage sobelLocalParallel(image);
	computeTime = SobelLocalParallel(&image, &sobelLocalParallel);
    	printf("tiempo Sobel local paralelo: %0.9f segundos\n", computeTime);

    	if(sobelImage == sobelLocal) printf("El algoritmo sobel-basico y sobel-local generan la misma imagen\n");
    	else printf("El algoritmo sobel-basico y sobel-local generan distinta imagen\n");

    	if(sobelLocal == sobelLocalParallel) printf("El algoritmo sobel-local y sobel-local paralelo generan la misma imagen\n");
    	else printf("El algoritmo sobel-basico y sobel-local generan distinta imagen\n");

    	QImage sobelCompleto(image);

    	computeTime = SobelCompleto(&image, &sobelCompleto);
    	printf("tiempo Sobel completo (local): %0.9f segundos\n", computeTime);

    	QImage sobelCompletoParallel(image);

    	computeTime = SobelCompletoParallel(&image, &sobelCompletoParallel);
    	printf("tiempo Sobel completo paralelo: %0.9f segundos\n", computeTime);

	if(sobelCompleto == sobelCompletoParallel) printf("El algoritmo sobel-completo y sobel-completo paralelo generan la misma imagen\n");
    	else printf("El algoritmo sobel-completo y sobel-completo paralelo generan distinta imagen\n");

    	QPixmap pixmap = pixmap.fromImage(sobelLocalParallel);
    	QGraphicsPixmapItem *item = new QGraphicsPixmapItem(pixmap);
    	scene.addItem(item);
    	view.show();
	a.exec();	

	pixmap = pixmap.fromImage(sobelCompletoParallel);
    	item = new QGraphicsPixmapItem(pixmap);
    	scene.addItem(item);
    	view.show();
    	return a.exec();
}
