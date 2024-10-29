﻿// AffineMatrix.cpp : Bu dosya 'main' işlevi içeriyor. Program yürütme orada başlayıp biter.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp> 

using namespace std;
using namespace cv;

bool inverseOfAffineMax(Mat& affineMax);
Vec3b calculateBilinearValue(const Mat& image, const double& y, const double& x);

void forwardMapping(const Mat &image);
void backwardMapping(const Mat& image);
void backwardMappingWithInterpolation(const Mat& image);

int main()
{
    cv::Mat image = cv::imread("istanbul.jpg");

    // Görüntü dosyası kontrolü
    if (image.empty()) {
        cout << "Image didnt found in the file" << endl;
        return -1;
    }

    Mat newImage; // create a new matris of the image
    
    imshow("Orijinal Resim", image);
    
	forwardMapping(image);
	backwardMapping(image);
    backwardMappingWithInterpolation(image);

    

    cv::waitKey(0);

}


void forwardMapping(const Mat& image) {
    int width = image.cols;
    int height = image.rows;
    // affine matrix values
    double a, b, tx;
    double c, d, ty;

#if 1 // zoom image 1.4x
    
    a = 1.4, b = 0, tx = 1;
    c = 0, d = 1.4, ty = 1;
    Mat zoomAffineMax = (Mat_<double>(2, 3) << a, b, tx, c, d, ty); 
    Mat zoomImage(height, width, CV_8UC3, cv::Scalar(0, 0, 0)); // new Image 

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int newX = static_cast<int>(x * zoomAffineMax.at<double>(0, 0)); // zoom bolgesi -> 1 * (zoomAffineMax.at<double>(0, 0) - tx)
            int newY = static_cast<int>(y * zoomAffineMax.at<double>(1, 1));
            if (newX > 0 && newX < width && newY > 0 && newY < height) {
                zoomImage.at<cv::Vec3b>(newY, newX) = image.at<cv::Vec3b>(y, x); // Forward mapping
            }
        }
    }

    imshow("zoomed image", zoomImage);
#endif

#if 0 // horizotanl shear image 1.4x
    
    a = 1, b = 1.4, tx = 1;
    c = 0, d = 1, ty = 1;
    int newWidth = width + (height * b);
    Mat horizantalShearAffMax= (Mat_<double>(2, 3) << a, b, tx, c, d, ty);
    Mat shearImage(height, newWidth, CV_8UC3, cv::Scalar(0, 0, 0)); // new Image 
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int newX =  static_cast<int>(x + (y * horizantalShearAffMax.at<double>(0, 1)));
            int newY = static_cast<int>(y); 
            if (newX >= 0 && newX < newWidth && newY >= 0 && newY < height) {
                shearImage.at<cv::Vec3b>(newY, newX) = image.at<cv::Vec3b>(y, x); // Forward mapping
            }
        }
    }

    imshow("Horizantal Shear", shearImage);
#endif

#if 0 // scale image 1.4x

    a = 1.4, b = 1, tx = 0;
    c = 0, d = 1.4, ty = 0;
    int newHeight = a * height;
    int newWidth = d * width;
    Mat scaleAffMax = (Mat_<double>(2, 3) << a, b, tx, c, d, ty);
    Mat scaleImage(newHeight, newWidth, CV_8UC3, cv::Scalar(0, 0, 0)); // new Image 

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int newX = static_cast<int>(x * scaleAffMax.at<double>(0, 0));
            int newY = static_cast<int>(y * scaleAffMax.at<double>(1, 1));
            if (newX >= 0 && newX < newWidth && newY >= 0 && newY < newHeight) {
                scaleImage.at<cv::Vec3b>(newY, newX) = image.at<cv::Vec3b>(y, x); // Forward mapping
            }
        }
    }

    imshow("scaled image", scaleImage);
#endif

#if 0 // rotate image 60 degree

    int centerX0 = width/ 2;
    int centerY0 = height / 2;
    int degre = 60;
    double radyan = degre * 3.14159 / 180;
    
    a = cos(radyan), b = -sin(radyan);
    c = sin(radyan), d = cos(radyan); 
    
    int newHeight = static_cast<int>(height * fabs(a)) + static_cast<int>(width * fabs(c));
    int newWidth = static_cast<int>(width * fabs(a)) + static_cast<int>(height * fabs(c));

    tx = (newWidth / 2.0) - (centerX0 * a + centerY0 * b);
    ty = (newHeight / 2.0) - (centerX0 * c + centerY0 * d);

    Mat scaleAffMax = (Mat_<double>(2, 3) << a, b, tx, c, d, ty);
    Mat rotateImage(newHeight, newWidth, CV_8UC3, cv::Scalar(0, 0, 0)); // new Image 

    // rotate
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int newX = static_cast<int>((x * scaleAffMax.at<double>(0, 0)) + (y * scaleAffMax.at<double>(0, 1)) + scaleAffMax.at<double>(0, 2));
            int newY = static_cast<int>((x * scaleAffMax.at<double>(1, 0)) + (y * scaleAffMax.at<double>(1, 1)) + scaleAffMax.at<double>(1, 2));
            if (newX >= 0 && newX < newWidth +200 && newY >= 0 && newY < newHeight) {
                rotateImage.at<cv::Vec3b>(newY, newX) = image.at<cv::Vec3b>(y, x); // Forward mapping
            }
        }
    }


    imshow("rotated image", rotateImage);
    /*if (!imwrite("rotated_image.jpg", rotateImage)) {
        std::cerr << "nanay!" << std::endl;
    }*/
#endif
    return;
}

void backwardMapping(const Mat& image) {
    int width = image.cols;
    int height = image.rows;
    // affine matrix values
    double a, b, tx;
    double c, d, ty;

#if 0 // zoom image 1.4x

    a = 1.4, b = 0, tx = 1;
    c = 0, d = 1.4, ty = 1;
    Mat zoomAffineMax = (Mat_<double>(2, 3) << a, b, tx, c, d, ty);
    Mat zoomImage(height , width, CV_8UC3, cv::Scalar(0, 0, 0)); // new Image wiht same size

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int newX = static_cast<int>(x / zoomAffineMax.at<double>(0, 0)); // zoom bolgesi -> 1 * (zoomAffineMax.at<double>(0, 0) - tx)
            int newY = static_cast<int>(y / zoomAffineMax.at<double>(1, 1));
            if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                zoomImage.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(newY, newX); // Backward mapping
            }
        }
    }

    imshow("zoomed image", zoomImage);
#endif

#if 0 // horizotanl shear image 1.4x

    a = 1, b = 1.4, tx = 1;
    c = 0, d = 1, ty = 1;
    int newWidth = width + (height * b);
    Mat horizantalShearAffMax = (Mat_<double>(2, 3) << a, b, tx, c, d, ty);
    Mat shearImage(height, newWidth, CV_8UC3, cv::Scalar(0, 0, 0)); // new Image 

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            int imageX = static_cast<int>(x - (y * horizantalShearAffMax.at<double>(0, 1)));
            int imageY = static_cast<int>(y);
            if (imageX >= 0 && imageX < width && imageY >= 0 && imageY < height) {
                shearImage.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(imageY, imageX); // Backward mapping
            }
        }
    }

    imshow("Horizantal Shear", shearImage);
#endif

#if 0 // scale image 1.4x

    a = 1.4, b = 1, tx = 0;
    c = 0, d = 1.4, ty = 0;
    int newHeight = a * height;
    int newWidth = d * width;
    Mat scaleAffMax = (Mat_<double>(2, 3) << a, b, tx, c, d, ty);
    Mat scaleImage(newHeight, newWidth, CV_8UC3, cv::Scalar(0, 0, 0)); // new Image 

    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            int imageX = static_cast<int>(x / scaleAffMax.at<double>(0, 0));
            int imageY = static_cast<int>(y / scaleAffMax.at<double>(1, 1));
            if (imageX >= 0 && imageX < width && imageY >= 0 && imageY < height) {
                scaleImage.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(imageY, imageX); // Backward mapping
            }
        }
    }

    imshow("scaled image", scaleImage);
#endif

#if 0 // rotate image 60 degree

    int centerX0 = width / 2;
    int centerY0 = height / 2;
    int degre = 60;
    double radyan = degre * 3.14159 / 180;

    a = cos(radyan), b = -sin(radyan);
    c = sin(radyan), d = cos(radyan);

    int newHeight = static_cast<int>(height * fabs(a)) + static_cast<int>(width * fabs(c));
    int newWidth = static_cast<int>(width * fabs(a)) + static_cast<int>(height * fabs(c));

    tx = (newWidth / 2.0) - (centerX0 * a + centerY0 * b);
    ty = (newHeight / 2.0) - (centerX0 * c + centerY0 * d);

    Mat scaleAffMax = (Mat_<double>(2, 3) << a, b, tx, c, d, ty);
    Mat rotateImage(newHeight, newWidth, CV_8UC3, cv::Scalar(0, 0, 0)); // new Image 

    // before inverse
    cout << " a ,b, tx, c, d, ty " << (scaleAffMax.at<double>(0, 0))<<" " << a << " " << b << " " << tx << " " << c << " " << d << " " << ty << endl;

	if (inverseOfAffineMax(scaleAffMax) == false) {
		std::cerr << "Affine matrix didnt turn inverse itself" << std::endl;
		return ;
	}
    cout << " a ,b, tx, c, d, ty " << (scaleAffMax.at<double>(0, 0)) << " " << a << " " << b << " " << tx << " " << c << " " << d << " " << ty << endl;
     
    // rotate
    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            int imageX = static_cast<int>((x * scaleAffMax.at<double>(0, 0)) + (y * scaleAffMax.at<double>(0, 1)) + scaleAffMax.at<double>(0, 2));
            int imageY = static_cast<int>((x * scaleAffMax.at<double>(1, 0)) + (y * scaleAffMax.at<double>(1, 1)) + scaleAffMax.at<double>(1, 2));
            //cout << "image x " << imageX;
            if (imageX >= 0 && imageX < width && imageY >= 0 && imageY < height) {
                rotateImage.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(imageY, imageX); // Backward mapping
            }
        }
    }


    imshow("rotated image", rotateImage);
    if (!imwrite("rotated_imageBackward.jpg", rotateImage)) {
        std::cerr << "nanay!" << std::endl;
    }
#endif
    return;
}

void backwardMappingWithInterpolation(const Mat& image) {
    int width = image.cols;
    int height = image.rows;
    // affine matrix values
    double a, b, tx;
    double c, d, ty;

#if 0 // zoom image 1.4x

    a = 1.4, b = 0, tx = 1;
    c = 0, d = 1.4, ty = 1;
    Mat zoomAffineMax = (Mat_<double>(2, 3) << a, b, tx, c, d, ty);
     
    Mat zoomImage(height, width, CV_8UC3, Scalar(0, 0, 0)); // new Image wiht new size

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double newX = x / (zoomAffineMax.at<double>(0, 0));
            double newY = y / (zoomAffineMax.at<double>(1, 1));
            if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                zoomImage.at<Vec3b>(y, x) =  calculateBilinearValue(image, newY, newX); // Backward mapping with interpolation
            }
        }
    }

    imshow("zoomed image", zoomImage);
#endif

#if 0 // horizotanl shear image 1.4x

    a = 1, b = 1.4, tx = 1;
    c = 0, d = 1, ty = 1;
    int newWidth = width + (height * b);
    Mat horizantalShearAffMax = (Mat_<double>(2, 3) << a, b, tx, c, d, ty);
    Mat shearImage(height, newWidth, CV_8UC3, cv::Scalar(0, 0, 0)); // new Image 

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            int imageX = static_cast<int>(x - (y * horizantalShearAffMax.at<double>(0, 1)));
            int imageY = static_cast<int>(y);
            if (imageX >= 0 && imageX < width && imageY >= 0 && imageY < height) {
                shearImage.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(imageY, imageX); // Backward mapping
            }
        }
    }

    imshow("Horizantal Shear", shearImage);
#endif

#if 0 // scale image 1.4x

    a = 1.4, b = 1, tx = 0;
    c = 0, d = 1.4, ty = 0;
    int newHeight = a * height;
    int newWidth = d * width;
    Mat scaleAffMax = (Mat_<double>(2, 3) << a, b, tx, c, d, ty);
    Mat scaleImage(newHeight, newWidth, CV_8UC3, cv::Scalar(0, 0, 0)); // new Image 

    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            int imageX = static_cast<int>(x / scaleAffMax.at<double>(0, 0));
            int imageY = static_cast<int>(y / scaleAffMax.at<double>(1, 1));
            if (imageX >= 0 && imageX < width && imageY >= 0 && imageY < height) {
                scaleImage.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(imageY, imageX); // Backward mapping
            }
        }
    }

    imshow("scaled image", scaleImage);
#endif

#if 0 // rotate image 60 degree

    int centerX0 = width / 2;
    int centerY0 = height / 2;
    int degre = 60;
    double radyan = degre * 3.14159 / 180;

    a = cos(radyan), b = -sin(radyan);
    c = sin(radyan), d = cos(radyan);

    int newHeight = static_cast<int>(height * fabs(a)) + static_cast<int>(width * fabs(c));
    int newWidth = static_cast<int>(width * fabs(a)) + static_cast<int>(height * fabs(c));

    tx = (newWidth / 2.0) - (centerX0 * a + centerY0 * b);
    ty = (newHeight / 2.0) - (centerX0 * c + centerY0 * d);

    Mat scaleAffMax = (Mat_<double>(2, 3) << a, b, tx, c, d, ty);
    Mat rotateImage(newHeight, newWidth, CV_8UC3, cv::Scalar(0, 0, 0)); // new Image 

    // before inverse
    cout << " a ,b, tx, c, d, ty " << (scaleAffMax.at<double>(0, 0)) << " " << a << " " << b << " " << tx << " " << c << " " << d << " " << ty << endl;

    if (inverseOfAffineMax(scaleAffMax) == false) {
        std::cerr << "Affine matrix didnt turn inverse itself" << std::endl;
        return;
    }
    cout << " a ,b, tx, c, d, ty " << (scaleAffMax.at<double>(0, 0)) << " " << a << " " << b << " " << tx << " " << c << " " << d << " " << ty << endl;

    // rotate
    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            int imageX = static_cast<int>((x * scaleAffMax.at<double>(0, 0)) + (y * scaleAffMax.at<double>(0, 1)) + scaleAffMax.at<double>(0, 2));
            int imageY = static_cast<int>((x * scaleAffMax.at<double>(1, 0)) + (y * scaleAffMax.at<double>(1, 1)) + scaleAffMax.at<double>(1, 2));
            //cout << "image x " << imageX;
            if (imageX >= 0 && imageX < width && imageY >= 0 && imageY < height) {
                rotateImage.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(imageY, imageX); // Backward mapping
            }
        }
    }


    imshow("rotated image", rotateImage);
    if (!imwrite("rotated_imageBackward.jpg", rotateImage)) {
        std::cerr << "nanay!" << std::endl;
    }
#endif
    return;
}

bool inverseOfAffineMax(Mat& affineMax) {
    // 2x3 boyutundaki affine dönüşüm matrisini 2x2 boyutuna dönüştürme
    double a = affineMax.at<double>(0, 0);
    double b = affineMax.at<double>(0, 1);
    double tx = affineMax.at<double>(0, 2);
    double c = affineMax.at<double>(1, 0);
    double d = affineMax.at<double>(1, 1);
    double ty = affineMax.at<double>(1, 2);

    // Determinantı hesapla
    double det = a * d - b * c;
    if (det == 0) {
        return false; // Tersini hesaplamak mümkün değil
    }

    // Ters matrisin hesaplanması
    double invDet = 1.0 / det;

    // Ters matrisin bileşenlerini ayarlama
    Mat inverseMat = (Mat_<double>(2, 3) <<
        d * invDet, -b * invDet, (b * ty - d * tx) * invDet,
        -c * invDet, a * invDet, (c * tx - a * ty) * invDet
        );

    // Sonucu affineMax matrisine yaz
    inverseMat.copyTo(affineMax);

    return true; // Ters hesaplama başarılı
}

Vec3b calculateBilinearValue(const Mat& image, const double &y, const double& x) {
    int x1 = static_cast<int>(x); // min round x
    int y1 = static_cast<int>(y); // min round y
    int x2 = min(x1 + 1, image.cols - 1); //  x1 + 1;
    int y2 = min(y1 + 1, image.rows - 1); //y1 + 1;
     

    Vec3b Q11 = image.at<Vec3b>(y1, x1);
    Vec3b Q21 = image.at<Vec3b>(y1, x2);
    Vec3b Q12 = image.at<Vec3b>(y2, x1);
    Vec3b Q22 = image.at<Vec3b>(y2, x2);


    double div = ((x2 - x1) * (y2 - y1));

    if (div <= 0) {
        cout << " e1"; 
        return Q11;
    }

    Vec3b resultDens;
    for (int i = 0; i < 3; ++i) { 
        resultDens[i] = static_cast<uchar>(
            (Q11[i] * (x2 - x) * (y2 - y) / div) +
            (Q21[i] * (x - x1) * (y2 - y) / div) +
            (Q12[i] * (x2 - x) * (y - y1) / div) +
            (Q22[i] * (x - x1) * (y - y1) / div)
            ); 
    }
    return resultDens;
}