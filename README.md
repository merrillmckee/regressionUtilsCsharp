This repository contains statistical regression utilities written in CSharp in Visual Studio 2019.  The utilities include standard least-squares linear, quadratic, cubic, and elliptical regression.  In addition to standard least squares regression, they contain "consensus regression" that automatically detects outliers in a dataset, removes them from the inliers, and updates the least-squares regression model quickly.

The image below is an example of linear regression and linear regression consensus.  The red line is a linear regression on all data points (blue).  The consensus algorithm identifies outliers (red) and then adjusts the regression model accordingly.
![image](https://user-images.githubusercontent.com/79757625/117515877-8a31a880-af65-11eb-92d2-d2359db90acb.png)

Here is an example that may apply to edge points detected in an image at a corner.  The original regression (red) fits a line through both lines.  The consensus algorithm identifies outliers (red) and fits its model to the dominant line.
![image](https://user-images.githubusercontent.com/79757625/117516097-19d75700-af66-11eb-8ada-33db30ab0081.png)

Anscombe's quartet is four sets of data that have virtually the same linear regression function.  In 3 of the 4 cases, outliers have a noticeable impact.  Two of the cases, a single outlier is enough to make the linear regression of poorer to unusable quality.  This quartet helps demonstate the sensitivity of common regression tools to outliers.  For more reading see https://en.wikipedia.org/wiki/Anscombe%27s_quartet.

The following images illustrate linear regression on the Anscombe's quartet:

![image](https://user-images.githubusercontent.com/79757625/117516460-260fe400-af67-11eb-94b9-02d05308799f.png)
![image](https://user-images.githubusercontent.com/79757625/117516341-db8e6780-af66-11eb-9e51-ccb08444e5d1.png)
![image](https://user-images.githubusercontent.com/79757625/117516368-ed700a80-af66-11eb-9b7e-da9a76444143.png)
![image](https://user-images.githubusercontent.com/79757625/117516357-e34e0c00-af66-11eb-8bb5-a39087cda84a.png)
![image](https://user-images.githubusercontent.com/79757625/117516395-fc56bd00-af66-11eb-84a8-a676fc86f429.png)

Some examples of quadratic regression and quadratic regression consensus:
![image](https://user-images.githubusercontent.com/79757625/117516567-83a43080-af67-11eb-9d4b-84a0e41f2d5a.png)
![image](https://user-images.githubusercontent.com/79757625/117516609-93bc1000-af67-11eb-92a5-242812e3ede4.png)
![image](https://user-images.githubusercontent.com/79757625/117516620-99b1f100-af67-11eb-9a48-621b33d128f4.png)
![image](https://user-images.githubusercontent.com/79757625/117516630-a0406880-af67-11eb-9c7c-d0957c7e86a7.png)
![image](https://user-images.githubusercontent.com/79757625/117516643-a6cee000-af67-11eb-993d-a34d4b61a81e.png)
![image](https://user-images.githubusercontent.com/79757625/117516653-acc4c100-af67-11eb-8e20-a21fa2e6a496.png)
![image](https://user-images.githubusercontent.com/79757625/117516662-b3533880-af67-11eb-9546-609595cdc0b2.png)
![image](https://user-images.githubusercontent.com/79757625/117516666-b8b08300-af67-11eb-9bb1-4732e251a71c.png)

Some examples of cubic regression and cubic regression consensus:
![image](https://user-images.githubusercontent.com/79757625/117516679-c82fcc00-af67-11eb-97e0-2731ec5906e5.png)
![image](https://user-images.githubusercontent.com/79757625/117516686-ccf48000-af67-11eb-971a-80b304e7c149.png)
![image](https://user-images.githubusercontent.com/79757625/117516693-d382f780-af67-11eb-8888-14e329727f9e.png)
![image](https://user-images.githubusercontent.com/79757625/117516698-d8e04200-af67-11eb-883d-692b24864844.png)

Some examples of elliptical regression and elliptical regression consensus:
![image](https://user-images.githubusercontent.com/79757625/117516715-e85f8b00-af67-11eb-9d7b-7817573d46be.png)
![image](https://user-images.githubusercontent.com/79757625/117516736-ee556c00-af67-11eb-90f0-580fca70ce4a.png)
![image](https://user-images.githubusercontent.com/79757625/117516745-f3b2b680-af67-11eb-9a9f-cf6877e02992.png)
![image](https://user-images.githubusercontent.com/79757625/117516757-f8776a80-af67-11eb-8116-bdb770c7c66b.png)
![image](https://user-images.githubusercontent.com/79757625/117516771-fca38800-af67-11eb-8fca-2591fe5e3890.png)

On the algorithm.  The algorithm takes some inspiration from RANSAC but where RANSAC builds random models from the smallest subsamples, this algorithm starts with all datapoints and iteratively labels and removes outliers.  Instead of an exhaustive search the the worst outlier to remove, only 3 candidates are considered.  Two of the candidates are those candidates with the greatest regression error "above" and "below" the model (substitute "left"/"right"/"inside"/"outside").  Since the summations was kept for the least squares calculations, removing an outlier is generally as simple as decrementing these summations.  Originally, only two candidates were going to be considered but in testing this initial algorithm it was not removing outliers for some "easy" cases.  I added a third candidate which is the candidate with maximum "influence"; this is generally the data point with maximum x * y.  Finally, iteration is stopped when the model's average regression error goes below a threshold.  As the domain I use the most is pixels in image processing, I generally have a good idea to what subpixel error to stop iteration.


