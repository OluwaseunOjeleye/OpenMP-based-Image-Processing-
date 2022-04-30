#pragma once

#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <math.h>


#define NO_CHANNELS 3
#define M_PI 3.14159265358979323846

/************************************Image******************************************************/
class Image {
	public:
		Image();
		Image(std::string filename);
		Image(cv::Mat_<cv::Vec3f> img);
		~Image();

		// Image Scaling
		Image scale_NN(int scale_factor)const;	// Nearest Neighbor Interpolation
		Image scale_Bilinear(int scale_factor)const;	// Bilinear Interpolation

		// Arithmetic Operators
		Image operator=(const Image &img);
		Image operator+(const Image &img)const;		// Addition
		Image operator-(const Image &img)const;		// Subtraction
		Image operator*(const Image &img)const;		// Multiplication
		Image operator/(const Image &img)const;		// Division
		Image operator+(float offset)const;			// offset pixel
		Image operator*(float scale_factor)const;	// Scaling Pixel


		//Filters - Convolution
		Image Blur(int kernel_size, float sigma);	// Gaussian Filter
		Image laplacian_Filter();					// Laplacian Filter

		Image gaussian_Separable(int kernel_size, float sigma);		// Separable Filter

		Image denoise(int kernel_size, float percentage) const;			// Denoise Filter

		// Color Transformation
		Image to_XYZ() const;
		Image to_Lab() const;
		Image shift_Hue(float hue) const;

		// Sapiro
		Image PDE(int iter, float tau, float alpha);

		// Display Image
		void display(std::string text);


	private:
		// Initialization Method
		void init(std::string filename);

		// Scaling
		float bilinear_interp(int x, int y, float x_diff, float y_diff, int color_index) const;

		// Convolution 2D
		cv::Mat_<cv::Vec3f> Convolution_2D(float **kernel, int kernel_size) const;
		float **get_2D_GaussianKernel(int kernel_size, float sigma) const;

		// Separable Filter
		cv::Mat_<cv::Vec3f> Convolution_1D_X(const cv::Mat_<cv::Vec3f> img, float *kernel, int kernel_size) const;
		cv::Mat_<cv::Vec3f> Convolution_1D_Y(const cv::Mat_<cv::Vec3f> img, float *kernel, int kernel_size) const;
		float *get_1D_GaussianKernel(int kernel_size, float sigma) const;

		// Denoising
		void sort(float *array, int window_size) const;

		// Color Transformation
		float linearize(float c) const;
		float linearize_inverse(float c) const;
		cv::Vec3f rgb_to_XYZ(cv::Vec3f pixel) const;

		float F(float v) const;
		float F_inverse(float v) const;
		cv::Vec3f rgb_to_Lab(cv::Vec3f pixel) const;
		cv::Vec3f rgb_to_LCh(cv::Vec3f pixel, float hue) const;
		cv::Vec3f LCh_to_rgb(cv::Vec3f pixel) const;

		// Sapiro
		cv::Mat_<cv::Vec3f> colorSapiro(float tau, float alpha) const;

		float dfX_0(int channel, int i, int j) const;
		float dfY_0(int channel, int i, int j) const;
		float dfX2(int channel, int i, int j) const;
		float dfY2(int channel, int i, int j) const;
		float dfXY(int channel, int i, int j) const;

		// Member Data
		cv::Mat_<cv::Vec3f> image;

		cv::Matx33f xyz_Matrix, rgb_Matrix;
		const float E = 0.2068965517241379;
		const cv::Vec3f	R = cv::Vec3f(95.04492182750991, 100, 108.89166484304715); //White Reference - D65
};