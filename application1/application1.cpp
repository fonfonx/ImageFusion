#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>

#include "../maxflow/graph.h"

using namespace std;
using namespace cv;

//fonction produisant le gradient de I, le gradient selon x et le gradient selon y
void gradient(Mat& I, Mat& out, Mat& outx, Mat& outy)
{
       	Mat NB;
        cvtColor(I, NB, CV_BGR2GRAY);
        int m = NB.rows, n = NB.cols;
        Mat Ix(m, n, CV_32F), Iy(m, n, CV_32F);
        for (int i = 0; i<m; i++) {
            for (int j = 0; j<n; j++){
               	float ix, iy;
                if (i == 0 || i == m - 1)
                   	iy = 0;
                else
                   	iy = (float(NB.at<uchar>(i + 1, j)) - float(NB.at<uchar>(i-1,j)))/2;
               	if (j == 0 || j == n - 1)
                   	ix = 0;
               	else
			ix=(float(NB.at<uchar>(i,j+1))-float(NB.at<uchar>(i,j-1)))/2;
		Ix.at<float>(i,j)=ix;
		Iy.at<float>(i,j)=iy;
     		out.at<float>(i, j) = sqrt(ix*ix + iy*iy);
     		}
  	}
	outx=Ix;
	outy=Iy;
}


//norme d'un vecteur (x,y)
double norme(float x, float y)
{
	return sqrt(x*x+y*y);
}

//distance entre deux Vec3b
double norme_dist(Vec3b src, Vec3b comp)
{
	int v0=src[0]-comp[0];
	int v1=src[1]-comp[1];
	int v2=src[2]-comp[2];
	int car=v0*v0+v1*v1+v2*v2;
	return (int)sqrt(car);
}

//distance entre A(x,y) et B(x,y)
double dist(Mat& A, Mat& B, int x, int y)
{
	return norme_dist(A.at<Vec3b>(x,y),B.at<Vec3b>(x,y));
}

//1re fonction de cout
double cost(Mat& A, Mat& B, int x1, int y1, int x2, int y2)
{
	return dist(A,B,x1,y1)+dist(A,B,x2,y2);
}

//2e fonction de cout, tenant compte du gradient
double cost2(Mat&A, Mat&B, int x1, int y1, int x2, int y2, Mat& GxA, Mat& GyA, Mat& GxB, Mat& GyB)
{
	Mat GA;
	Mat GB;
	if (x1==x2)
	{
		GA=GxA;
		GB=GxB;
	}
	else
	{
		GA=GyA;
		GB=GyB;
	}
	double grad=norme(GA.at<float>(x1,y1),0);
	grad=grad+norme(GA.at<float>(x2,y2),0);
	grad=grad+norme(GB.at<float>(x1,y1),0);
	grad=grad+norme(GB.at<float>(x2,y2),0);
	grad=pow(grad,0.26);
	return cost(A,B,x1,y1,x2,y2)/grad;
}

//fonction remplissant la matrice seam indiquant la position de la coupure
//in et out ont la même dimension
void find_couture(Mat& in, Mat& out, Mat& seam)
{
	Mat aux(in.rows, in.cols, CV_8U);
	int lf=out.cols;
	int hf=out.rows;

	for (int i=0; i<in.rows; i++)
	{
		for (int j=0; j<in.cols; j++)
		{
			if (out.at<Vec3b>(i,j)==in.at<Vec3b>(i,j))
			{
				aux.at<uchar>(i,j)=1;
			}
			else
			{
				aux.at<uchar>(i,j)=0;
			}
		}
	}
	
	int a;
	int b;
	double c;
	//détection verticale de la couture
	for(int j=1; j<in.cols; j++)
	{
		a=aux.at<uchar>(1,j);
		for (int i=1; i<in.rows; i++)
		{
			b=aux.at<uchar>(i,j);
			if (a!=b)
			{
				a=b;
				seam.at<uchar>(i,j)=1;
			}
		}
	}
}

//fonction de lissage (au niveau de la couture)

//lissage gaussien
//p1, p2 et p3 sont les poids affectés respectivement à la case centrale, les 4 cases adjacentes et les 4 coins d'un carré de 9 cases
Vec3b gaussien(int i, int j, Mat& in, int p1, int p2, int p3)
{
	Vec3b rep;
	Vec3b a,b,c,d,e,f,g,h,ii;
	b=in.at<Vec3b>(i-1,j);
	e=in.at<Vec3b>(i,j);
	h=in.at<Vec3b>(i+1,j);
	if (j!=0)
	{
		a=in.at<Vec3b>(i-1,j-1);
		d=in.at<Vec3b>(i,j-1);
		g=in.at<Vec3b>(i+1,j-1);
	}
	if (j!=in.cols-1)
	{
		c=in.at<Vec3b>(i-1,j+1);
		f=in.at<Vec3b>(i,j+1);
		ii=in.at<Vec3b>(i+1,j+1);
	}
	if (j==0)
	{
		for (int k=0; k<3; k++)
		{
			rep[k]=(p2*b[k]+p3*c[k]+p1*e[k]+p2*f[k]+p2*h[k]+p3*ii[k])/(p1+3*p2+2*p3);
		}	
	}
	else if (j==in.cols-1)
	{
		for (int k=0; k<3; k++)
		{
			rep[k]=(p2*b[k]+p3*a[k]+p1*e[k]+p2*d[k]+p2*h[k]+p3*g[k])/(p1+3*p2+2*p3);
		}	
	}
	else
	{
		for (int k=0; k<3; k++)
		{
			rep[k]=(p3*a[k]+p2*b[k]+p3*c[k]+p2*d[k]+p1*e[k]+p2*f[k]+p3*g[k]+p2*h[k]+p3*ii[k])/(p1+4*p2+4*p3);
		}
	}	
	return rep;

}

//fonction de lissage
//les trois matrices ont les mêmes dimensions
void lissage(Mat& in, Mat& out, Mat& seam)
{
	int hf=in.rows;
	int lf=out.cols;
	for (int i=0; i<hf; i++)
	{
		for(int j=0; j<lf; j++)
		{
			out.at<Vec3b>(i,j)=in.at<Vec3b>(i,j);
		}
	}
	for (int i=0; i<hf; i++)
	{
		for (int j=0; j<lf; j++)
		{
			if (seam.at<uchar>(i,j)==1)
			{
				out.at<Vec3b>(i,j)=gaussien(i,j,in,3,2,1);
				out.at<Vec3b>(i+1,j)=gaussien(i+1,j,in,4,2,1);
				out.at<Vec3b>(i-1,j)=gaussien(i-1,j,in,4,2,1);
				out.at<Vec3b>(i+2,j)=gaussien(i+2,j,in,5,2,1);
				out.at<Vec3b>(i-2,j)=gaussien(i-2,j,in,5,2,1);
				//décommenter la ligne suivante pour obtenir en rouge la ligne de coupure
				//out.at<Vec3b>(i,j)=Vec3b(0,0,255);
			}
		}
	}
}



int main(){

	//lecture des images d'origine
	Mat I1=imread("../../hut.jpg");
	Mat I2=imread("../../mountain.jpg");

	imshow("I1",I1);
	imshow("I2",I2);

	waitKey();
	
	//si les images n'ont pas la même hauteur
	const int DIFF=I1.rows-I2.rows;

	int largeur=min(I1.cols,I2.cols);
	int hauteur=max(I1.rows,I2.rows);
	
	//Creation d'images de meme taille
	Mat I1bis(hauteur,largeur,CV_8UC3);
	Mat I2bis(hauteur,largeur,CV_8UC3);

	for (int i=0; i<hauteur; i++)
	{
		for(int j=0; j<largeur; j++)
		{
			if (i<I1.rows && j<I1.cols)
			{
				I1bis.at<Vec3b>(i-DIFF,j)=I1.at<Vec3b>(i,j);
			}
			else
			{
				I1bis.at<Vec3b>(i-I1.rows,j)=Vec3b(0,0,0);
			}
			if (i<I2.rows && j<I2.cols)
			{
				I2bis.at<Vec3b>(i,j)=I2.at<Vec3b>(i,j);
			}
			else
			{
				I2bis.at<Vec3b>(i,j)=Vec3b(0,0,0);
			}

		}
	}
	I1=I1bis;
	I2=I2bis;
	Mat Gx1(hauteur,largeur,CV_32F);
	Mat Gx2(hauteur,largeur,CV_32F);
	Mat Gy1(hauteur,largeur,CV_32F);
	Mat Gy2(hauteur,largeur,CV_32F);
	Mat G1(hauteur,largeur,CV_32F);
	Mat G2(hauteur,largeur,CV_32F);
	gradient(I1,G1,Gx1,Gy1);
	gradient(I2,G2,Gx2,Gy2);

	//matrice contenant la couture
	Mat seam(hauteur,largeur, CV_8U);
	for (int i=0; i<hauteur; i++)
	{
		for (int j=0; j<largeur; j++)
		{
			seam.at<uchar>(i,j)=0;
		}
	}

	//bandes haute et basse de l'image qui restent figées dans l'image finale
	int hbas=60;
	int hhaut=30;
	hhaut=hhaut-DIFF;
	imshow("I1",I1);
	imshow("I2",I2);
	waitKey();

	int taille=hauteur-1-hhaut-hbas+2;
	
	//constante représentant l'infini
	const int MAX_INT=10000;

	//creation du graphe de flot
	Graph<double,double,double> g(largeur*taille,4*taille*largeur);
	g.add_node(largeur*taille);
	int ii;
	for (int i=0; i<taille; i++)
	{
		ii=i+hhaut-1;
		for (int j=0; j<largeur; j++)
		{
			int pix = i*largeur+j;
			double c;
			if (i==0)
			{
				g.add_tweights(pix,MAX_INT,0);
			}
			if (i==taille-1)
			{
				g.add_tweights(pix,0,MAX_INT);
			}
			if (i!=taille-1)
			{	
				c=cost2(I1,I2,ii,j,ii+1,j,Gx1,Gy1,Gx2,Gy2);
				g.add_edge(pix, (i+1)*largeur+j,c,c);
			}
			if (j!=largeur-1)
			{
				c=cost2(I1,I2,ii,j,ii,j+1,Gx1,Gy1,Gx2,Gy2);
				g.add_edge(pix,i*largeur+j+1,c,c);
			}
		}
	}
	
	double flow=g.maxflow();
	//cout<<flow<<endl;

	//image resultat
	Mat res(hauteur,largeur, CV_8UC3);
	Mat reslisse(hauteur, largeur, CV_8UC3);
	for (int i=0; i<hhaut; i++)
	{
		for (int j=0; j<largeur; j++)
		{
			res.at<Vec3b>(i,j)=I2.at<Vec3b>(i,j);
		}
	}

	for (int i=0; i<hbas; i++)
	{
		for (int j=0; j<largeur; j++)
		{
			res.at<Vec3b>(i-1+hhaut+taille,j)=I1.at<Vec3b>(i-1+hhaut+taille,j);
		}
	}

	//reconstitution de l'image
	for (int i=0; i<taille; i++)
	{
		for (int j=0; j<largeur; j++)
		{
			int pix=i*largeur+j;
			if (g.what_segment(pix)==Graph<double,double,double>::SINK)
			{
				res.at<Vec3b>((i+hhaut-1),j)=I1.at<Vec3b>((i+hhaut-1),j);
			}
			else
			{
				res.at<Vec3b>((i+hhaut-1),j)=I2.at<Vec3b>((i+hhaut-1),j);
			}
		}
	}
	
	imshow("Resultat",res);
	find_couture(res,I2,seam);
	lissage(res,reslisse,seam);
	imshow("Resultat avec lissage", reslisse);
	waitKey();
	return 0;

}
