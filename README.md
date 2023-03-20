RUN:
./application scale -bilinear 3 resize.jpg
./application add 1.0 0.0 im1.png im2.jpg
./application subtract 1.0 0.0 im1.png im2.jpg
./application multiply 1.0 0.0 im1.png im2.jpg
./application divide 1.0 0.0 im1.png im2.jpg
./application blur 5 3.7 grenouille.jpg
./application laplacian grenouille.jpg
./application separable 5 3.7 grenouille.jpg
./application denoise 3 50 grenouille.jpg
./application shift_hue 100 grenouille.jpg
./application pde 50, 0.05, 0.1 grenouille.jpg
