This repository contains statistical regression utilities written in CSharp in Visual Studio 2019.  The utilities include standard least-squares linear, quadratic, cubic, and elliptical regression.  In addition to standard least squares regression, they contain "consensus regression" that automatically detects outliers in a dataset, removes them from the inliers, and updates the least-squares regression model quickly.

The image below is an example of linear regression and linear regression consensus.  The red line is a linear regression on all data points (blue).  The consensus algorithm identifies outliers (red) and then adjusts the regression model accordingly.
![image](https://user-images.githubusercontent.com/79757625/117515877-8a31a880-af65-11eb-92d2-d2359db90acb.png)

Here is an example that may apply to edge points detected in an image at a corner.  The original regression (red) fits a line through both lines.  The consensus algorithm identifies outliers (red) and fits its model to the dominant line.
![image](https://user-images.githubusercontent.com/79757625/117516097-19d75700-af66-11eb-8ada-33db30ab0081.png)

Anscombe's quartet is four sets of data that have virtually the same linear regression function.  In 3 of the 4 cases, outliers have a noticeable impact.  Two of the cases, a single outlier is enough to make the linear regression of poorer to unusable quality.  This quartet helps demonstate the sensitivity of common regression tools to outliers.  For more reading see https://en.wikipedia.org/wiki/Anscombe%27s_quartet.

The following 4 images are linear regression on the Anscombe's quartet:

![image](https://user-images.githubusercontent.com/79757625/117516460-260fe400-af67-11eb-94b9-02d05308799f.png)
![image](https://user-images.githubusercontent.com/79757625/117516341-db8e6780-af66-11eb-9e51-ccb08444e5d1.png)
![image](https://user-images.githubusercontent.com/79757625/117516368-ed700a80-af66-11eb-9b7e-da9a76444143.png)
![image](https://user-images.githubusercontent.com/79757625/117516357-e34e0c00-af66-11eb-8bb5-a39087cda84a.png)
![image](https://user-images.githubusercontent.com/79757625/117516395-fc56bd00-af66-11eb-84a8-a676fc86f429.png)



