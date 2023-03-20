RUN:<br>
./application scale -bilinear 3 resize.jpg<br>
./application add 1.0 0.0 im1.png im2.jpg<br>
./application subtract 1.0 0.0 im1.png im2.jpg<br>
./application multiply 1.0 0.0 im1.png im2.jpg<br>
./application divide 1.0 0.0 im1.png im2.jpg<br>
./application blur 5 3.7 grenouille.jpg<br>
./application laplacian grenouille.jpg<br>
./application separable 5 3.7 grenouille.jpg<br>
./application denoise 3 50 grenouille.jpg<br>
./application shift_hue 100 grenouille.jpg<br>
./application pde 50, 0.05, 0.1 grenouille.jpg<br>
