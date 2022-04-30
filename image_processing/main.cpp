#include <chrono>  // for high_resolution_clock
#include "IP.h"		// for Image Processing

/*
scale -bilinear 3 resize.jpg
add 1.0 0.0 im1.png im2.jpg
subtract 1.0 0.0 im1.png im2.jpg
multiply 1.0 0.0 im1.png im2.jpg
divide 1.0 0.0 im1.png im2.jpg
blur 5 3.7 grenouille.jpg
laplacian grenouille.jpg
separable 5 3.7 grenouille.jpg
denoise 3 50 grenouille.jpg
shift_hue 100 grenouille.jpg
pde 50, 0.05, 0.1 grenouille.jpg
*/

int main(int argc, char** argv){
	
	Image result;
	if (argv[1] == std::string("scale")) {
		Image img(argv[4]); img.display("Image");
		if(argv[2] == std::string("-nearest")) result = img.scale_NN(atoi(argv[3]));
		else								   result = img.scale_Bilinear(atoi(argv[3]));
		result.display("Scale");
	}
	else if (argv[1] == std::string("add")) {
		Image img1(argv[4]);
		Image img2(argv[5]);

		result = ((img1 + img2) * atof(argv[2])) + atof(argv[3]);
		result.display("Combined Arithmetic Operation: Addition");
	}
	else if (argv[1] == std::string("subtract")) {
		Image img1(argv[4]);
		Image img2(argv[5]);

		result = ((img1 - img2) * atof(argv[2])) + atof(argv[3]);
		result.display("Combined Arithmetic Operation: Subtraction");
	}
	else if (argv[1] == std::string("multiply")) {
		Image img1(argv[4]);
		Image img2(argv[5]);

		result = ((img1 * img2) * atof(argv[2])) + atof(argv[3]);
		result.display("Combined Arithmetic Operation: Multiply");
	}
	else if (argv[1] == std::string("divide")) {
		Image img1(argv[4]);
		Image img2(argv[5]);

		result = ((img1 / img2) * atof(argv[2])) + atof(argv[3]);
		result.display("Combined Arithmetic Operation: Division");
	}
	else if (argv[1] == std::string("blur")){
		Image img(argv[4]); img.display("Image");
		result = img.Blur(atoi(argv[2]), atof(argv[3]));
		result.display("Gaussian - Blur");
	}
	else if (argv[1] == std::string("laplacian")) {
		Image img(argv[2]); img.display("Image");
		result = img.laplacian_Filter();
		result.display("Laplacian");
	}
	else if (argv[1] == std::string("separable")) {
		Image img(argv[4]); img.display("Image");
		result = img.gaussian_Separable(atoi(argv[2]), atof(argv[3]));
		result.display("Separable Filter");
	}
	else if (argv[1] == std::string("denoise")) {
		Image img(argv[4]); img.display("Image");
		result = img.denoise(atoi(argv[2]), atof(argv[3]));
		result.display("Denoise");
	}
	else if (argv[1] == std::string("shift_hue")) {
		Image img(argv[3]); img.display("Image");
		result = img.shift_Hue(atoi(argv[2]));
		result.display("Shift Hue - LCH Space");
	}
	else if (argv[1] == std::string("pde")) {
		Image img(argv[5]); img.display("Image");
		result = img.PDE(atoi(argv[2]), atof(argv[3]), atof(argv[4]));
		result.display("PDE");
	}
	else {
		std::cout << "Wrong Command" << std::endl;
	}

	/*
	Image img1("im1.png");
	Image img2("im2.jpg");
	Image img3;
	Image img4("resize.jpg");

	
	// img1.display("Image 1");
	// img2.display("Image 2");
	// img3.display("Image 3");
	img4.display("Image 4");

	auto begin = std::chrono::high_resolution_clock::now();

	Image img_scale_NN, img_scale_Bilinear, sum, subt, mult, div, offset, scale, C;
	Image blur, laplacian, separable, denoise, XYZ, Lab, LCh, pde;
	const int iter = 1;	
	for (int it = 0; it < iter; it++){

		// img_scale_NN			= img4.scale_NN(2);
		// img_scale_Bilinear	= img4.scale_Bilinear(2); 

		// sum					= img1 + img2;
		// subt					= img1 - img2; 
		// mult					= img1 * img2;
		// div					= img1 / img2; 
		// offset				= img3 + 0.7;
		// scale				= img3 * 2; 

		// C = ((img1 + img2) * 2) + 0.2;

		// blur			= img3.Blur(3, 1);	//(16, 3.4); 
		// laplacian	= img3.laplacian_Filter(); 

		// separable	= img3.gaussian_Separable(3, 1); //(16, 3.4); 

		// denoise	= img3.denoise(3, 50); //(7, 100);
		// XYZ		= img3.to_XYZ();
		// Lab		= img3.to_Lab(); 
		// LCh		= img3.shift_Hue(100);
		pde		= img3.PDE(50, 0.05, 0.1);
	}

	// img_scale_NN.display("Image Scaling - Nearest Neighbor");
	// img_scale_Bilinear.display("Image Scaling - Bilinear");
	// sum.display("Addition Operation");
	// subt.display("Subtraction Operation");
	// mult.display("Multiplication Operation");
	// div.display("Division Operation");
	// offset.display("Offset");
	// scale.display("Scaling Pixel");
	// C.display("All Arithmetic Operators");
	// blur.display("Blur");
	// laplacian.display("Laplacian");
	// separable.display("Separable");
	// denoise.display("Denoise");
	// XYZ.display("XYZ");
	// Lab.display("Lab");
	// LCh.display("LCh");
	pde.display("PDE");



	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - begin;

	std::cout << "Total time: " << diff.count() << " s" << std::endl;
	std::cout << "Time for 1 iteration: " << diff.count() / iter << " s" << std::endl;
	std::cout << "IPS: " << iter / diff.count() << std::endl;

	*/

	cv::waitKey();
	return 0;
}

