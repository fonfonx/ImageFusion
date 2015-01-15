#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "../maxflow/graph.h"

using namespace std;
using namespace cv;

//fonction random dans un intervalle entier
int random(int a,int b)
{
	return rand()%(b-a)+a;
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

//fonction produisant le gradient de I, le gradient selon x et le gradient selon y, à partir de l'image en couleurs
void gradient(Mat& I, Mat& out, Mat& outx, Mat& outy)
{
        int m = I.rows, n = I.cols;
        Mat Ix(m, n, CV_32F), Iy(m, n, CV_32F);
	for (int i=0; i<m;i++)
	{
		for (int j=0; j<n; j++)
		{
			Ix.at<float>(i,j)=0;
			Iy.at<float>(i,j)=0;
			out.at<float>(i,j)=0;
		}
	}
        for (int i = 0; i<m; i++) {
            for (int j = 0; j<n; j++){
               	float ix, iy;
                if (i == 0 || i == m - 1)
                   	iy = 0;
                else
                   	iy = (float(norme_dist(I.at<Vec3b>(i + 1, j),I.at<Vec3b>(i-1,j))))/2;
               	if (j == 0 || j == n - 1)
                   	ix = 0;
               	else
			ix=(float(norme_dist(I.at<Vec3b>(i,j+1),I.at<Vec3b>(i,j-1))))/2;
		Ix.at<float>(i,j)=ix;
		Iy.at<float>(i,j)=iy;
     		out.at<float>(i, j) = sqrt(ix*ix + iy*iy);
     		}
  	}
	outx=Ix;
	outy=Iy;
}

//fonction produisant le gradient de I, le gradient selon x et le gradient selon y, à partir de l'image en Noir et Blanc
void gradientNB(Mat& I, Mat& out, Mat& outx, Mat& outy)
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



//1re fonction de cout
double cost(Mat& A, Mat& B, int x1, int y1, int x2, int y2)
{
	return dist(A,B,x1,y1)+dist(A,B,x2,y2)/3;
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
	grad=pow(grad,0.2);
	return cost(A,B,x1,y1,x2,y2)/grad;
}

//parties entière et décimale
int ent(double d)
{
	return floor(d);
}

double dec(double d)
{
	return d-floor(d);
}

//////////////////////////
//Fonctions de placement//
//////////////////////////

//fonction remplissant la matrice des anciennes coutures seam
//in et out ont la même dimension
void find_couture(int k, int l, Mat& in, Mat& out, Mat& seam, Mat& seam_pix,  Mat& Gxin, Mat& Gyin, Mat& Gxout, Mat& Gyout)
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
	for(int i=1; i<in.rows; i++)
	{
		a=aux.at<uchar>(i,0);
		for (int j=1; j<in.cols; j++)
		{
			b=aux.at<uchar>(i,j);
			if (a!=b)
			{
				a=b;
				c=cost2(in,out,i,j,i,j+1,Gxin,Gyin,Gxout,Gyout);
				seam.at<float>(i*lf+j,0)=c;
				seam_pix.at<Vec3b>(i,j)=out.at<Vec3b>(i,j);
				seam_pix.at<Vec3b>(i,j-1)=out.at<Vec3b>(i,j-1);
			}
		}
	}
	for (int j=0; j<in.cols; j++)
	{
		a=aux.at<uchar>(0,j);
		for (int i=0; i<in.rows; i++)
		{
			b=aux.at<uchar>(i,j);
			if (a!=b)
			{
				a=b;
				c=cost2(in,out,i-1,j,i,j,Gxin,Gyin,Gxout,Gyout);
				seam.at<float>(i*lf+j,1)=c;
				seam_pix.at<Vec3b>(i,j)=out.at<Vec3b>(i,j);
				seam_pix.at<Vec3b>(i-1,j)=out.at<Vec3b>(i-1,j);

			}
		}
	}
}


//Fonction de placement initial, quand le cadre n'est pas encore rempli
void placement_initial(int k, int l, Mat& in, Mat& out, Mat& seam, Mat& seam_pix, int over)
{
	const int MAX_INT=230;
	int larg=in.cols;
	int haut=in.rows;
	int lf=out.cols;
	int hf=out.rows;

	int deb_i=k*(haut-over);
	int deb_j=l*(larg-over);
	int fin_i=deb_i+haut;
	int fin_j=deb_j+larg;
	
	//cette matrice est une image de même taille que l'image de sortie out,
	//elle contient l'image d'entrée in dans un fond noir
	//c'est plus pratique pour les indices dans les calculs de travailler avec des images de même taille
	Mat dans_noir(hf,lf,CV_8UC3);
		
	//creation du graphe de flot
	Graph<double,double,double> g(lf*hf,5*lf*hf);
	g.add_node(lf*hf);

	for(int i=0; i<hf; i++)
	{
		for(int j=0; j<lf; j++)
		{
			if (i>=deb_i&&i<=fin_i&&j>=deb_j&&j<fin_j)
			{
				dans_noir.at<Vec3b>(i,j)=in.at<Vec3b>(i-deb_i,j-deb_j);
			}
		}
	}

	//gradient
	Mat Gxin(hf,lf,CV_32F);
	Mat Gxout(hf,lf,CV_32F);
	Mat Gyin(hf,lf,CV_32F);
	Mat Gyout(hf,lf,CV_32F);
	Mat Gin(hf,lf,CV_32F);
	Mat Gout(hf,lf,CV_32F);
	gradient(dans_noir,Gin,Gxin,Gyin);
	gradient(out,Gout,Gxout,Gyout);
	
	for (int i=0; i<hf; i++)
	{
		for(int j=0; j<lf; j++)
		{
			int pix = i*lf+j;
			if (out.at<Vec3b>(i,j)==Vec3b(0,0,0))
			{
				if (i>=deb_i&&i<fin_i&&j>=deb_j&&j<fin_j)
				{
					g.add_tweights(pix,0,MAX_INT);
				}
				else
				{
					g.add_tweights(pix,MAX_INT,0);
				}
			}
			else
			{
				if (i>=deb_i&&i<fin_i&&j>=deb_j&&j<fin_j)
				{
					//zone d'overlap
					//on doit mettre des poids infinis sur les bords de cette zone
					if (abs(i-(haut-over))<=1)
					{
						g.add_tweights(pix,MAX_INT*(k==1),MAX_INT*(k==0));
					}
					if (abs(i-haut)<=1)
					{
						g.add_tweights(pix,MAX_INT*(k==0),MAX_INT*(k==1));
					}
					if (abs(i-(2*haut-2*over))<=1)
					{
						g.add_tweights(pix,MAX_INT*(k==2),0);
					}
					if (abs(i-(2*haut-over))<=1)
					{
						g.add_tweights(pix,0,MAX_INT*(k==2));
					}


					if (abs(j-(larg-over))<=1)
					{
						g.add_tweights(pix,MAX_INT*(l==1),MAX_INT*(l==0));
					}
					if (abs(j-larg)<=1)
					{
						g.add_tweights(pix,MAX_INT*(l==0),MAX_INT*(l==1));
					}
					if (abs(j-(2*larg-2*over))<=1)
					{
						g.add_tweights(pix,MAX_INT*(l==2),0);
					}
					if (abs(j-(2*larg-over))<=1)
					{
						g.add_tweights(pix,0,MAX_INT*(l==2));
					}

					//définition des poids des arêtes internes
					double c;
					if (i!=hf-1)
					{	
						c=cost2(dans_noir,out,i,j,i+1,j,Gxin,Gyin,Gxout,Gyout);
						g.add_edge(pix, (i+1)*lf+j,c,c);
					}
					if (j!=lf-1)
					{
						c=cost2(dans_noir,out,i,j,i,j+1,Gxin,Gyin,Gxout,Gyout);
						g.add_edge(pix,i*lf+j+1,c,c);
					}
				}
				else
				{
					g.add_tweights(pix,MAX_INT,0);
				}
 					
			}
		}
	}
	//on rajoute les "old seams" au graphe
	int node;
	double c;
	int i,j;
	for (int kk=0; kk<lf*hf;kk++)
	{
		for (int ll=0;ll<2;ll++)
		{
			if (seam.at<float>(kk,ll)!=0)
			{
				node=g.add_node();
				i=kk/lf;
				j=kk%lf;
				g.add_tweights(node,0,seam.at<float>(kk,ll));
				c=norme_dist(seam_pix.at<Vec3b>(i,j),dans_noir.at<Vec3b>(i,j));
				c=c+norme_dist(seam_pix.at<Vec3b>(i-(ll==0),j-(ll==1)),dans_noir.at<Vec3b>(i-(ll==0),j-(ll==1)));
				g.add_edge(node,i*lf+j,c,c);
				g.add_edge(node,((i-(ll==0))*lf+j-(ll==1)),c,c);
			}
		}
	}

	//le graphe est maintenant défini
	double flow=g.maxflow();
	//cout<<flow<<endl;
	for (int i=0; i<hf; i++)
	{
		for (int j=0; j<lf;j++)
		{
			int pix=i*lf+j;
			if (g.what_segment(pix)==Graph<double,double,double>::SINK)
			{
				out.at<Vec3b>(i,j)=dans_noir.at<Vec3b>(i,j);
			}
		}
	}
	find_couture(k,l,dans_noir,out,seam,seam_pix,Gxin,Gyin,Gxout,Gyout);
}

//fonction d'ajout d'un patch en position (k,l) (coordonnées en haut à gauche) quand l'image est déjà remplie
void ajout_patch(int k, int l, Mat& patch, Mat& out, Mat& seam, Mat& seam_pix, int over)
{
	const int MAX_INT=1230;
	int larg=patch.cols;
	int haut=patch.rows;
	int lf=out.cols;
	int hf=out.rows;
	
	Mat dans_noir(hf,lf,CV_8UC3);
	for (int i=0; i<hf; i++)
	{
		for (int j=0; j<lf; j++)
		{
			dans_noir.at<Vec3b>(i,j)=Vec3b(0,0,0);
		}
	}
		
	//creation du graphe de flots
	Graph<double,double,double> g(lf*hf,5*lf*hf);
	g.add_node(lf*hf);

	for(int i=0; i<hf; i++)
	{
		for(int j=0; j<lf; j++)
		{
			if (i>=k&&i<k+haut&&j>=l&&j<l+larg)
			{
				dans_noir.at<Vec3b>(i,j)=patch.at<Vec3b>(i-k,j-l);
			}
		}
	}
	Mat Gxin(hf,lf,CV_32F);
	Mat Gxout(hf,lf,CV_32F);
	Mat Gyin(hf,lf,CV_32F);
	Mat Gyout(hf,lf,CV_32F);
	Mat Gin(hf,lf,CV_32F);
	Mat Gout(hf,lf,CV_32F);
	gradient(dans_noir,Gin,Gxin,Gyin);
	gradient(out,Gout,Gxout,Gyout);

	for (int i=0; i<hf; i++)
	{
		for (int j=0; j<lf; j++)
		{
			int pix=i*lf+j;
			if (dans_noir.at<Vec3b>(i,j)==Vec3b(0,0,0))
			{
				g.add_tweights(pix,MAX_INT,0);
			}
			else 
			{
				//inutile (cf article)
				/*if (abs(i-(k+haut/2))<=over/2 && abs(j-(l+larg/2))<=over/2)
				{
					g.add_tweights(pix,0,MAX_INT);
				}*/
				bool b_x=(abs(i-k)<=2);
				bool b_xx=(abs(i-(k+haut))<=2);
				bool b_y=(abs(j-l)<=2);
				bool b_yy=(abs(j-(l+larg))<=2);
				if ((b_x && b_y) || (b_x && b_yy) || (b_xx && b_y) || (b_xx && b_yy))
				{
					g.add_tweights(pix,MAX_INT,0);
				}

				//définition des poids des arêtes internes
				double c;
				if (i!=hf-1)
				{	
					c=cost2(dans_noir,out,i,j,i+1,j,Gxin,Gyin,Gxout,Gyout);
					g.add_edge(pix, (i+1)*lf+j,c,c);
				}
				if (j!=lf-1)
				{
					c=cost2(dans_noir,out,i,j,i,j+1,Gxin,Gyin,Gxout,Gyout);
					g.add_edge(pix,i*lf+j+1,c,c);
				}


			}
		}
	}
	//Prise en compte de la matrice seam
	int node;
	double c;
	int i,j;
	for (int kk=0; kk<lf*hf;kk++)
	{
		for (int ll=0;ll<2;ll++)
		{
			if (seam.at<float>(kk,ll)!=0)
			{
				node=g.add_node();
				i=kk/lf;
				j=kk%lf;
				g.add_tweights(node,0,seam.at<float>(kk,ll));
				c=norme_dist(seam_pix.at<Vec3b>(i,j),patch.at<Vec3b>(i,j));
				c=c+norme_dist(seam_pix.at<Vec3b>(i-(ll==0),j-(ll==1)),patch.at<Vec3b>(i-(ll==0),j-(ll==1)));
				g.add_edge(node,i*lf+j,c,c);
				g.add_edge(node,((i-(ll==0))*lf+j-(ll==1)),c,c);
			}
		}
	}
	//le graphe est maintenant défini
	double flow=g.maxflow();
	for (int i=0; i<hf; i++)
	{
		for (int j=0; j<lf;j++)
		{
			int pix=i*lf+j;
			if (g.what_segment(pix)==Graph<double,double,double>::SINK)
			{
				out.at<Vec3b>(i,j)=dans_noir.at<Vec3b>(i,j);
			}
		}
	}
	find_couture(k,l,dans_noir,out,seam,seam_pix,Gxin,Gyin,Gxout,Gyout);
}


//fonction déterminant un patch aléatoire
void random_patch(Mat& in, Mat& patch)
{
	int x=in.rows;
	int y=in.cols;
	int taille=patch.cols;
	int debx=random(0,x-taille);
	int deby=random(0,y-taille);
	for (int i=0; i<taille; i++)
	{
		for (int j=0; j<taille; j++)
		{
			patch.at<Vec3b>(i,j)=in.at<Vec3b>(i+debx,j+deby);
		}
	}
}


int main(){
	
	//pour la fonction random
	srand(time(NULL));
	Mat Img=imread("../../pavage.jpg");
//	Mat Img=imread("../../argent.jpg");
	imshow("Image",Img);

	int largeur=Img.cols;
	int hauteur=Img.rows;
	//Creation de l'image finale
	//taille de l'overlap entre deux images
	int over=ent(min(largeur,hauteur)/5);
	//largeur et hauteur finales de l'image output
	int lf=3*largeur-2*over;
	int hf=3*hauteur-2*over;
	Mat output(hf,lf,CV_8UC3);

	//matrice indiquant la position des précédentes coûtures et la valeur du coût de la coupe
	//On remplit la case d'ordonnée 0 si la coupure est à i constant, et 1 si c'est à j constant (cf la fonction find_couture)
	Mat seam(hf*lf,2,CV_32F);
	//matrice contenant les valeurs des pixels au bord desquels a eu lieu une coupe
	Mat seam_pix(hf,hf,CV_8UC3);
	for (int i=0; i<hf*lf; i++)
	{
		for (int j=0; j<2; j++)
		{
			seam.at<float>(i,j)=0.0f;
		}
	}
	
	//premier remplissage de output, avec l'image d'origine au milieu
	for (int i=0; i<hf; i++)
	{
		for(int j=0; j<lf; j++)
		{
			if ((i>=hauteur-over&&i<2*hauteur-over)&&(j>=largeur-over&&j<2*largeur-over))
			{
				output.at<Vec3b>(i,j)=Img.at<Vec3b>(i-hauteur+over,j-largeur+over);
			}
			else
			{
				output.at<Vec3b>(i,j)=Vec3b(0,0,0);
			}
			seam_pix.at<Vec3b>(i,j)=Vec3b(0,0,0);
		}
	}
	imshow("Resultat",output);
	waitKey();
	
	//positionnement des 8 images du début
	for (int l=0; l<=2; l++)
	{
		for (int k=0; k<=2; k++)
		{
			if (k*l!=1)
			{
				placement_initial(k,l,Img,output,seam,seam_pix,over);
				imshow("Resultat",output);
				//waitKey();
			}
		}
	}

	waitKey();

	//Décommenter tout ce paragraphe pour voir le résultat obtenu (moins bon) en rajoutant des patches aléatoires
/*
	//Ajout de 4 patchs aléatoires au niveau des 4 coins de l'image de départ
	int tpatch=ent(over*4);
	Mat patch(tpatch,tpatch, CV_8UC3);
	for (int i=0; i<tpatch; i++)
	{
		for (int j=0; j<tpatch; j++)
		{
			patch.at<Vec3b>(i,j)=Vec3b(0,0,0);
		}
	}

	int x1=hf/3-tpatch/2;
	int y1=lf/3-tpatch/2;

	for (int aa=0; aa<2; aa++)
	{
		for (int bb=0; bb<2; bb++)
		{
			random_patch(Img, patch);
			ajout_patch(x1+aa*(hauteur+over/4),y1+bb*(largeur+over/4),patch,output,seam,seam_pix,over);
		}
	}
	imshow("Resultat",output);
	waitKey();

	//on rajoute de nombreux (nbr) patches aléatoires
	int nbr=50;
	int x;
	int y;
	int step=0;
	while (step<nbr)
	{
		x=random(0,hf-tpatch);
		y=random(0,lf-tpatch);
		
		random_patch(Img, patch);
		ajout_patch(x,y,patch,output,seam,seam_pix,over);
		step++;
	}
	imshow("Resultat",output);
	waitKey();

*/
	return 0;

}
