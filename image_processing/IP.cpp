#include "IP.h"

void Image::init(std::string filename) {
	if (filename != "") {
		cv::Mat_<cv::Vec3b> img = cv::imread(filename, 1);
		img.convertTo(this->image, CV_32FC3, 1 / 255.0);
	}	

	xyz_Matrix = { 0.4124564, 0.3575761, 0.1804375,
					0.2126729, 0.7151522, 0.0721750,
					0.0193339, 0.1191920, 0.9503041 };

	rgb_Matrix = { 3.2404542, -1.5371385, -0.4985314,
					-0.9692660, 1.8760108, 0.0415560,
					0.0556434, -0.2040259, 1.0572252 };
}

Image::Image() {
	init("grenouille.jpg");
}

Image::Image(std::string filename) {
	init(filename);
}

Image::Image(cv::Mat_<cv::Vec3f> img) {
	init("");
	img.copyTo(this->image);
}

Image::~Image() {

}

Image Image::operator=(const Image &img) {
	if (this != &img) this->image = img.image;
	return  *this;
}

// Scaling Algorithm - Nearest Neighbor Interpolation
Image Image::scale_NN(int scale_factor) const{
	cv::Mat_<cv::Vec3f> temp; 
	temp.create(this->image.rows * scale_factor, this->image.cols * scale_factor);

#pragma omp parallel for
	for (int i = 0; i < temp.rows; i++) {
		for (int j = 0; j < temp.cols; j++) {
			temp(i, j) = this->image(floor(i/scale_factor), floor(j / scale_factor));
		}
	}
	return(Image(temp));
}

float Image::bilinear_interp(int x, int y, float x_diff, float y_diff, int color_index) const {
	float A = this->image(y, x)[color_index];
	float B = this->image(y, x + 1)[color_index];
	float C = this->image(y + 1, x)[color_index];
	float D = this->image(y + 1, x + 1)[color_index];
	return A * (1 - x_diff) * (1 - y_diff) + B * x_diff*(1 - y_diff) + C * (1 - x_diff) * y_diff + D * x_diff * y_diff;
}

// Scaling Algorithm - Bilinear Interpolation
Image Image::scale_Bilinear(int scale_factor) const{
	cv::Mat_<cv::Vec3f> temp;
	temp.create(this->image.rows * scale_factor, this->image.cols * scale_factor);

	float x_ratio = float(this->image.cols - 1) / temp.cols;
	float y_ratio = float(this->image.rows - 1) / temp.rows;

#pragma omp parallel for
	for (int i = 0; i < temp.rows; i++) {
		for (int j = 0; j < temp.cols; j++) {
			int x = x_ratio * j;
			int y = y_ratio * i;

			float x_diff = (x_ratio * j) - x;
			float y_diff = (y_ratio * i) - y;

			temp(i, j)[0] = bilinear_interp(x, y, x_diff, y_diff, 0);
			temp(i, j)[1] = bilinear_interp(x, y, x_diff, y_diff, 1);
			temp(i, j)[2] = bilinear_interp(x, y, x_diff, y_diff, 2);

		}
	}
	return(Image(temp));
}
/********************************************************* Arithmetic Operator******************************************************/

// Addition Operator
Image Image::operator+(const Image &img)const{
	if (this->image.rows != img.image.rows || this->image.cols != img.image.cols) return this->image;

	cv::Mat_<cv::Vec3f> temp;
	this->image.copyTo(temp);

#pragma omp parallel for
	for (int i = 0; i < this->image.rows; i++) {
		for (int j = 0; j < this->image.cols; j++) {
			temp(i, j) = this->image(i, j) + img.image(i, j);
		}
	}
	return Image(temp);
}

// Subtraction Operator
Image Image::operator-(const Image &img)const{
	if (this->image.rows != img.image.rows || this->image.cols != img.image.cols) return this->image;

	cv::Mat_<cv::Vec3f> temp;
	this->image.copyTo(temp);

#pragma omp parallel for
	for (int i = 0; i < this->image.rows; i++) {
		for (int j = 0; j < this->image.cols; j++) {
			temp(i, j)= this->image(i, j) - img.image(i, j);
		}
	}
	return Image(temp);
}


// Multiplication
Image Image::operator*(const Image &img)const {
	if (this->image.rows != img.image.rows || this->image.cols != img.image.cols) return this->image;

	cv::Mat_<cv::Vec3f> temp;
	this->image.copyTo(temp);

#pragma omp parallel for
	for (int i = 0; i < this->image.rows; i++) {
		for (int j = 0; j < this->image.cols; j++) {
			temp(i, j) = this->image(i, j).mul(img.image(i, j));
		}
	}
	return Image(temp);
}

// Division
Image Image::operator/(const Image &img)const {
	if (this->image.rows != img.image.rows || this->image.cols != img.image.cols) return this->image;

	cv::Mat_<cv::Vec3f> temp;
	this->image.copyTo(temp);

#pragma omp parallel for
	for (int i = 0; i < this->image.rows; i++) {
		for (int j = 0; j < this->image.cols; j++) {
			for (int c = 0; c < NO_CHANNELS; c++)
				temp(i, j)[c] = (img.image(i, j)[c] == 0)? 0: this->image(i, j)[c] / img.image(i, j)[c];
		}
	}
	return Image(temp);
}

// Offset pixel
Image Image::operator+(float offset) const{
	cv::Mat_<cv::Vec3f> temp;
	this->image.copyTo(temp);
	 
#pragma omp parallel for
	for (int i = 0; i < this->image.rows; i++) {
		for (int j = 0; j < this->image.cols; j++) {
			temp(i, j) = this->image(i, j) + cv::Vec3f(offset);
		}
	}
	return Image(temp);
}
 
// Scaling Pixel
Image Image::operator*(float scale_factor) const{
	cv::Mat_<cv::Vec3f> temp;
	this->image.copyTo(temp);

#pragma omp parallel for
	for (int i = 0; i < this->image.rows; i++) {
		for (int j = 0; j < this->image.cols; j++) {
			temp(i, j) = this->image(i, j).mul(cv::Vec3f(scale_factor, scale_factor, scale_factor));
		}
	}
	return Image(temp);
}


/********************************************************* Convolution Operator: Gaussian and Laplacian******************************************************/
// 2D-Convolution Operator
cv::Mat_<cv::Vec3f> Image::Convolution_2D(float **kernel, int kernel_size) const{
	int size = floor(kernel_size / 2);

	cv::Mat_<cv::Vec3f> temp;
	this->image.copyTo(temp);
	//temp.image.setTo(cv::Scalar(0.0, 0.0, 0.0));

	// Convolving image with 2D-kernel
#pragma omp parallel for
	for (int i = size; i < this->image.rows - size; i++) {
		for (int j = size; j < this->image.cols - size; j++) {

			float pixel_sum[NO_CHANNELS] = { 0.0 };
			for (int k = -size; k <= size; k++) {
				for (int l = -size; l <= size; l++) {

					for (int c = 0; c < NO_CHANNELS; c++)
						pixel_sum[c] += this->image(i + k, j + l)[c] * kernel[k + size][l + size];
				}
			}
			for (int c = 0; c < NO_CHANNELS; c++)
				temp(i, j)[c] = pixel_sum[c];
		}
	}
	return temp;
}

// Creating gaussian filter
float** Image::get_2D_GaussianKernel(int kernel_size, float sigma) const{
	int k_size = (kernel_size % 2 == 0) ? kernel_size + 1 : kernel_size;

	float **filter = new float*[k_size];
	for (int i = 0; i < k_size; i++)
		filter[i] = new float[k_size];


	int size = floor(k_size / 2);
	float s = 2.0*sigma*sigma, sum = 0;
	for (int i = -size; i <= size; i++) {
		for (int j = -size; j <= size; j++) {
			float r		= -(i*i + j * j) / s;
			float Coeff	= exp(r) / (M_PI * s);
			filter[i+size][j+size] = Coeff;
			sum += Coeff;
		}
	}

	for (int i = 0; i < k_size; i++)
		for (int j = 0; j < k_size; j++)
			filter[i][j] /= sum;
			
	return filter;
}

// Gaussian Filter
Image Image::Blur(int kernel_size, float sigma) {
	float **gaussian_filter = get_2D_GaussianKernel(kernel_size, sigma);		// Creating Gaussian Filter
	return Image(Convolution_2D(gaussian_filter, kernel_size));			// Applying Gaussian Filter
}

// Laplacian Filter
Image Image::laplacian_Filter() {
	// Creating Laplacian Kernel
	int kernel_size = 3;
	float **filter = new float*[kernel_size];
	for (int i = 0; i < kernel_size; i++)
		filter[i] = new float[kernel_size];

	for (int i = 0; i < kernel_size; i++)
		for (int j = 0; j < kernel_size; j++)
			filter[i][j] = -1.0;
	int size = floor(kernel_size / 2);
	filter[size][size] = 8.0;

	return Image(Convolution_2D(filter, kernel_size));				// Applying Laplacian Filter
}


/********************************************************* Convolution Operator 1D: Separable Filter******************************************************/
// 1D-Convolution Operator - Along X-axis
cv::Mat_<cv::Vec3f> Image::Convolution_1D_X(const cv::Mat_<cv::Vec3f> img, float *kernel, int kernel_size) const {
	int size = floor(kernel_size / 2);

	cv::Mat_<cv::Vec3f> temp;
	img.copyTo(temp);

	// Convolving image with 1D-kernel
#pragma omp parallel for
	for (int i = size; i < img.rows - size; i++) {
		for (int j = size; j < img.cols - size; j++) {

			float pixel_sum[NO_CHANNELS] = { 0.0 };
			for (int k = -size; k <= size; k++) {
				for (int c = 0; c < NO_CHANNELS; c++){
					pixel_sum[c] += img(i + k, j)[c] * kernel[k + size];		// convolve along x-axis
				}
			}
			for (int c = 0; c < NO_CHANNELS; c++)
				temp(i, j)[c] = pixel_sum[c];
		}
	}
	return temp;
}

// 1D-Convolution Operator - Along Y-axis
cv::Mat_<cv::Vec3f> Image::Convolution_1D_Y(const cv::Mat_<cv::Vec3f> img, float *kernel, int kernel_size) const {
	int size = floor(kernel_size / 2);

	cv::Mat_<cv::Vec3f> temp;
	img.copyTo(temp);

	// Convolving image with 1D-kernel
#pragma omp parallel for
	for (int i = size; i < img.rows - size; i++) {
		for (int j = size; j < img.cols - size; j++) {

			float pixel_sum[NO_CHANNELS] = { 0.0 };
			for (int k = -size; k <= size; k++) {
				for (int c = 0; c < NO_CHANNELS; c++) {
					pixel_sum[c] += img(i, j + k)[c] * kernel[k + size];		// convolve along y-axis
				}
			}
			for (int c = 0; c < NO_CHANNELS; c++)
				temp(i, j)[c] = pixel_sum[c];
		}
	}
	return temp;
}

// Creating 1D-Gaussian Kernel
float* Image::get_1D_GaussianKernel(int kernel_size, float sigma) const {
	int k_size = (kernel_size % 2 == 0) ? kernel_size + 1 : kernel_size;

	float *filter = new float[k_size];

	int size = floor(k_size / 2);
	float s = 2.0*sigma*sigma, sum = 0;
	for (int i = -size; i <= size; i++) {
		float r = -(i*i) / s;
		float Coeff = exp(r) / (M_PI * s);
		filter[i + size] = Coeff;
		sum += Coeff;
	}

	for (int i = 0; i < k_size; i++)
		filter[i] /= sum;

	return filter;
}

Image Image::gaussian_Separable(int kernel_size, float sigma) {
	float *gaussian_filter = get_1D_GaussianKernel(kernel_size, sigma);		// Creating Gaussian Filter

	// Applying Gaussian Filter along x-axis
	cv::Mat_<cv::Vec3f> temp = Convolution_1D_Y(this->image, gaussian_filter, kernel_size);

	// Applying Gaussian Filter along y-axis
	return Image(Convolution_1D_X(temp, gaussian_filter, kernel_size));	
}

void Image::sort(float *array, int window_size) const {
	for (int i = 0; i < window_size - 1; i++) {
		for (int j = 0; j < window_size - i - 1; j++) {
			if (array[j] > array[j + 1]) {
				float temp = array[j];
				array[j] = array[j + 1];
				array[j + 1] = temp;
			}
		}
	}
}

// Denoising
Image Image::denoise(int kernel_size, float percentage) const {
	int k_size = (kernel_size % 2 == 0) ? kernel_size + 1 : kernel_size;
	int window_size = k_size * k_size;
	int size = floor(k_size / 2);

	cv::Mat_<cv::Vec3f> temp;
	this->image.copyTo(temp);

#pragma omp parallel for
	for (int i = size; i < this->image.rows - size; i++) {
		for (int j = size; j < this->image.cols - size; j++) {

			// Get array when window is convolved over image
			float *R_array = new float[window_size];
			float *G_array = new float[window_size];
			float *B_array = new float[window_size];
			int counter = 0;

			for (int k = -size; k <= size; k++) {
				for (int l = -size; l <= size; l++) {
					B_array[counter] = this->image(i + k, j + l)[0];
					G_array[counter] = this->image(i + k, j + l)[1];
					R_array[counter] = this->image(i + k, j + l)[2];
					counter++;
				}
			}

			// Sort neighbors (RGB arrays) in ascending order
			sort(B_array, window_size);
			sort(G_array, window_size);
			sort(R_array, window_size);

			// Get median - since window_size is always odd, index = window_size/2
			int index = int(window_size / 2);

			// Using either Median Filter or Average Filter
			int length = (window_size * percentage / 100) / 2.0;

			float r = 0.0, g = 0.0, b = 0.0;
			for (int i = index - length; i <= index + length; i++) {
				b += B_array[i];
				g += G_array[i];
				r += R_array[i];
			}

			temp(i, j)[0] = b / float(2 * length + 1);
			temp(i, j)[1] = g / float(2 * length + 1);
			temp(i, j)[2] = r / float(2 * length + 1);
		}
	}
	return temp;
}

float Image::linearize(float c) const{
	if (c <= 0.04045) return c / 12.92;
	return pow(((c + 0.055) / 1.055), 2.4);
}

float Image::linearize_inverse(float c) const{
	if (c <= 0.0031308) return 12.92*c;
	return (1.055 * pow(c, 1.0 / 2.4)) - 0.055;
}


cv::Vec3f Image::rgb_to_XYZ(cv::Vec3f pixel) const{
	cv::Vec3f linear_color = cv::Vec3f(linearize(pixel[0]), linearize(pixel[1]), linearize(pixel[2]));	
	return (xyz_Matrix * linear_color);
}

float Image::F(float v) const{
	if (v > pow(E, 3.0)) return pow(v, 1.0 / 3.0);
	return ((v / (3.0*pow(E, 2.0))) + 0.1379310344827586);
}

float Image::F_inverse(float v) const{
	if (v > E) return pow(v, 3.0);
	return (3.0*pow(E, 2.0)) * (v - 0.1379310344827586);
}

cv::Vec3f Image::rgb_to_Lab(cv::Vec3f pixel) const{
	cv::Vec3f XYZ = rgb_to_XYZ(pixel);

	float Fx = F(XYZ[0] / R[0]);
	float Fy = F(XYZ[1] / R[1]);
	float Fz = F(XYZ[2] / R[2]);

	return cv::Vec3f((116.0*Fy) - 16.0, 500.0*(Fx - Fy), 200.0*(Fy - Fz));
}

cv::Vec3f Image::rgb_to_LCh(cv::Vec3f pixel, float hue) const{
	cv::Vec3f lab = rgb_to_Lab(pixel);
	float c = sqrt((lab[1] * lab[1]) + (lab[2] * lab[2]));
	float h = (atan2(lab[2], lab[1]) * 180 / M_PI) + hue;
	h = h > 360.0 ? h - 360.0 : h;
	return cv::Vec3f(lab[0], c, h*M_PI / 180);
}

cv::Vec3f Image::LCh_to_rgb(cv::Vec3f pixel) const{
	cv::Vec3f Lab = cv::Vec3f(pixel[0], pixel[1] * cos(pixel[2]), pixel[1] * sin(pixel[2]));
	float L_star = (Lab[0] + 16.0) / 116.0;
	float a_star = Lab[1] / 500.0;
	float b_star = Lab[2] / 200.0;
	cv::Vec3f XYZ = cv::Vec3f(R[0] * F_inverse(L_star + a_star),
								R[1] * F_inverse(L_star),
								R[2] * F_inverse(L_star - b_star));
	cv::Vec3f color = rgb_Matrix * XYZ;
	return cv::Vec3f(linearize_inverse(color[0]), linearize_inverse(color[1]), linearize_inverse(color[2]));
}

Image Image::to_XYZ() const{
	cv::Mat_<cv::Vec3f> temp;
	this->image.copyTo(temp);

#pragma omp parallel for
	for (int i = 0; i < this->image.rows; i++) {
		for (int j = 0; j < this->image.cols; j++) {
			cv::Vec3f color = this->image(i, j);
			cv::Vec3f result = rgb_to_XYZ(cv::Vec3f(color[2], color[1], color[0]));		// bgr to rgb
			temp(i, j) = cv::Vec3f(result[2], result[1], result[0]);					// rgb to bgr
		}
	}

	return temp;
}

Image Image::to_Lab() const {
	cv::Mat_<cv::Vec3f> temp;
	this->image.copyTo(temp);

#pragma omp parallel for
	for (int i = 0; i < this->image.rows; i++) {
		for (int j = 0; j < this->image.cols; j++) {
			cv::Vec3f color = this->image(i, j);
			cv::Vec3f result = rgb_to_Lab(cv::Vec3f(color[2], color[1], color[0]));		// bgr to rgb
			temp(i, j) = cv::Vec3f(result[2], result[1], result[0]);					// rgb to bgr
		}
	}

	return temp;
}

Image Image::shift_Hue(float hue) const {
	cv::Mat_<cv::Vec3f> temp;
	this->image.copyTo(temp);

#pragma omp parallel for
	for (int i = 0; i < this->image.rows; i++) {
		for (int j = 0; j < this->image.cols; j++) {
			cv::Vec3f color = this->image(i, j);
			cv::Vec3f result = LCh_to_rgb(rgb_to_LCh(cv::Vec3f(color[2], color[1], color[0]), hue));		// bgr to rgb
			temp(i, j) = cv::Vec3f(result[2], result[1], result[0]);					// rgb to bgr
		}
	}

	return temp;
}

float Image::dfX_0(int channel, int i, int j) const {
	return (image(i + 1, j)[channel] - image(i - 1, j)[channel]) / 2.0;
}

float Image::dfY_0(int channel, int i, int j) const {
	return (image(i, j + 1)[channel] - image(i, j - 1)[channel]) / 2.0;
}

float Image::dfX2(int channel, int i, int j) const {
	return (image(i + 1, j)[channel] - 2.0* image(i, j)[channel] + image(i - 1, j)[channel]);
}

float Image::dfY2(int channel, int i, int j) const {
	return (image(i, j + 1)[channel] - 2.0* image(i, j)[channel] + image(i, j - 1)[channel]);
}

float Image::dfXY(int channel, int i, int j) const {
	return ((image(i + 1, j + 1)[channel] + image(i - 1, j - 1)[channel] - (image(i - 1, j + 1)[channel] + image(i + 1, j - 1)[channel])) / 4.0);
}

cv::Mat_<cv::Vec3f> Image::colorSapiro(float tau, float alpha) const{
	cv::Mat_<cv::Vec3f> temp;
	this->image.copyTo(temp);

#pragma omp parallel for
	for (int i = 1; i < this->image.rows - 1; i++){
		for (int j = 1; j < this->image.cols - 1; j++){
			float indicator, g11, g22, g12;
			float Ix[3];
			float Iy[3];
			float Ixx[3];
			float Iyy[3];
			float Ixy[3];
			float value;
			float alphag;

			float num;
			float denom;

			Ix[0] = dfX_0(0, i, j);
			Ix[1] = dfX_0(1, i, j);
			Ix[2] = dfX_0(2, i, j);

			Iy[0] = dfY_0(0, i, j);
			Iy[1] = dfY_0(1, i, j);
			Iy[2] = dfY_0(2, i, j);

			Ixx[0] = dfX2(0, i, j);
			Ixx[1] = dfX2(1, i, j);
			Ixx[2] = dfX2(2, i, j);

			Iyy[0] = dfY2(0, i, j);
			Iyy[1] = dfY2(1, i, j);
			Iyy[2] = dfY2(2, i, j);

			Ixy[0] = dfXY(0, i, j);
			Ixy[1] = dfXY(1, i, j);
			Ixy[2] = dfXY(2, i, j);

			g11 = 1.0 + Ix[0] * Ix[0] + Ix[1] * Ix[1] + Ix[2] * Ix[2];
			g12 = Ix[0] * Iy[0] + Ix[1] * Iy[1] + Ix[2] * Iy[2];
			g22 = 1.0 + Iy[0] * Iy[0] + Iy[1] * Iy[1] + Iy[2] * Iy[2];

			indicator = sqrt((g11 - g22)*(g11 - g22) + 4.0 *g12*g12);

			value = sqrt(indicator) / tau;
			value *= -value;
			alphag = alpha * exp(value);

			for (int c = 0; c < 3; c++)
			{
				num = Ixx[c] * Iy[c] * Iy[c] - 2.0*Ixy[c] * Ix[c] * Iy[c] + Iyy[c] * Ix[c] * Ix[c];
				denom = 1e-8 + Ix[c] * Ix[c] + Iy[c] * Iy[c];

				value = image(i, j)[c] + alphag * num / denom;

				if (value > 1) value = 1;
				else if (value < 0) value = 0;

				temp(i, j)[c] = value;
			}
		}
	}

	return temp;
}

Image Image::PDE(int iter, float tau, float alpha) {
	cv::Mat_<cv::Vec3f> temp, result, k;
	this->image.copyTo(temp);

	for (int i = 0; i < iter; i++) {
		if (i % 10 == 0) std::cout << "iteration " << i << std::endl;
		this->image = colorSapiro(tau, alpha);
	}
	this->image.copyTo(result);
	temp.copyTo(this->image);
	return result;
}

void Image::display(std::string text) {
	cv::imshow(text, this->image);
}
