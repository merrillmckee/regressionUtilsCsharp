using System;
using System.Collections;
using System.Drawing;

// For matrix inversion
using MathNet.Numerics.LinearAlgebra;

// For unit testing
using System.Windows.Forms;
using System.Collections.Generic;
using System.Diagnostics;
using PointF = System.Drawing.PointF;

namespace Tools.RegressionUtilities
{
    /// <summary>
    /// EllipticalRegression
    /// Author: Merrill McKee
    /// Description:  The purpose of this class is to find the ellipse/hyperbola that best fits through a set of 2D 
    ///   X, Y points.  It uses the Least Squares Fit method to do this.  A list of System.Drawing.PointF 
    ///   objects are passed in to the constructor.  These points are used to find the variables a, b, c, d, and e 
    ///   in the equation.  f is assumed to be -1.
    ///         ax^2 + bxy + cy^2 + dx + ey + f = 0
    ///   For more information on the algorithm/formulas used, see the website 
    ///   http://www.mathworks.com/matlabcentral/fileexchange/3215-fit-ellipse/content/fit_ellipse.m and 
    ///   https://www.cs.cornell.edu/cv/OtherPdf/Ellipse.pdf and the notes 
    ///   below.  The method is slightly different than the existing algorithms used in the linear, quadratic, 
    ///   and cubic regressions implemented in this same project.  Those use explicit derivations where this 
    ///   method requires calculating the inverse of a 5x5 matrix.
    ///   
    ///   Notes: Additional matrix algebra details not in the website link:
    ///   Note:  Using the following shorthand:  Sxy == SUM(xy)  == SIGMA(xy)  over all data values
    ///                                          Sx2 == SUM(x^2) == SIGMA(x^2) over all data values
    ///          
    ///           A  = INV(X'X) * X
    ///          
    ///          [a] = [Sx4   Sx3y  Sx2y2 Sx3   Sx2y]-1 * [Sx2]
    ///          [b]   [Sx3y  Sx2y2 Sxy3  Sx2y  Sxy2]     [Sxy]
    ///          [c]   [Sx2y2 Sxy3  Sx2y2 Sxy2  Sy3 ]     [Sy2]
    ///          [d]   [Sx3   Sx2y  Sxy2  Sx2   Sxy ]     [Sx ]
    ///          [e]   [Sx2y  Sxy2  Sy3   Sxy   Sy2 ]     [Sy ]
    /// 
    /// </summary>
    [Serializable]
    public static class EllipticalRegression
    {
        private const double EPSILON = 0.0001;    // Near-zero value to check for division-by-zero or singular matrices
        const float ERROR_THRESHOLD_ORIGINAL = 0.2f;

        #region Public Ellipse Structures
        public enum EllipseHalves
        {
            TopHalf = 1,
            BottomHalf = 2,
            RightHalf = 3,
            LeftHalf = 4
        };

        public struct EllipseCoefficients
        {
            public double a;                        // Coefficients of ax^2 + bxy + cy^2 + dx + ey + f = 0
            public double b;
            public double c;
            public double d;
            public double e;
            public double f;

            public override string ToString()
            {
                return a + " " + b + " " + c + " " + d + " " + e + " " + f;
            }
        }

        public class EllipseModel : RegressionModel
        {
            internal EllipseCoefficients coefficients;
            internal float x0;                        // (x0, y0) is the center of the ellipse
            internal float y0;
            internal double tilt;                     // Tilt angle, in radians, CCW from the positive x-axis (CW in image coordinates if +y is down)
            internal float radiusX;                   // a
            internal float radiusY;                   // b
            internal float long_axis;                 // Long diameter.  Max(2a, 2b)
            internal float short_axis;                // Short diameter.  Min(2a, 2b)
            internal EllipseBias bias;

            #region Internal Properties of EllipseModel
            internal double a
            {
                get { return coefficients.a; }
                set { coefficients.a = value; }
            }

            internal double b
            {
                get { return coefficients.b; }
                set { coefficients.b = value; }
            }

            internal double c
            {
                get { return coefficients.c; }
                set { coefficients.c = value; }
            }

            internal double d
            {
                get { return coefficients.d; }
                set { coefficients.d = value; }
            }

            internal double e
            {
                get { return coefficients.e; }
                set { coefficients.e = value; }
            }

            internal double f
            {
                get { return coefficients.f; }
                set { coefficients.f = value; }
            }
            #endregion

            #region Public Properties of EllipseModel

            // (cz) A new version of CreateEllipse() with tilt parameter
            // the math notations follow this webpage https://www.maa.org/external_archive/joma/Volume8/Kalman/General.html
            public void CreateEllipse(float p_x0, float p_y0, float p_radiusX, float p_radiusY, float p_tilt = 0.0f)
            {
                if (Math.Abs(p_radiusX) < 0.000001)
                {
                    throw new ArgumentException("p_radiusX is almost zero");
                }

                if (Math.Abs(p_radiusY) < 0.000001)
                {
                    throw new ArgumentException("p_radiusY is almost zero");
                }

                x0 = p_x0;
                y0 = p_y0;
                radiusX = p_radiusX;
                radiusY = p_radiusY;
                tilt    = p_tilt;

                var cos_a  = Math.Cos(p_tilt);
                var sin_a  = Math.Sin(p_tilt);
                var cos_a2 = cos_a * cos_a;
                var sin_a2 = sin_a * sin_a;
                var a2     = p_radiusX * p_radiusX;
                var b2     = p_radiusY * p_radiusY;

                var A = cos_a2 / a2 + sin_a2 / b2;
                var B = 2.0f * cos_a * sin_a * (1 / a2 - 1 / b2);
                var C = sin_a2 / a2 + cos_a2 / b2;
                var h = p_x0;
                var k = p_y0;

                coefficients.a = A;
                coefficients.b = B;
                coefficients.c = C;
                coefficients.d = -(2.0f * A * h + k * B);
                coefficients.e = -(2.0f * C * k + B * h);
                coefficients.f = A * h * h + B * h * k + C * k * k - 1.0f;

                long_axis = 2.0f * (radiusX > radiusY ? radiusX : radiusY);
                short_axis = 2.0f *  (radiusX < radiusY ? radiusX : radiusY);

                ValidRegressionModel = true;
                averageRegressionError = 0.0f;
            }

            // Get the coefficients of ax^2 + bxy + cy^2 + dx + ey + f = 0
            public EllipseCoefficients Coefficients
            {
                get { return coefficients; }
            }

            // Get length of long axis
            public float LongAxis
            {
                get
                {
                    if (ValidRegressionModel)
                    {
                        return long_axis;
                    }
                    else
                    {
                        return float.MinValue;
                    }
                }
            }

            // Get length of short axis
            public float ShortAxis
            {
                get
                {
                    if (ValidRegressionModel)
                    {
                        return short_axis;
                    }
                    else
                    {
                        return float.MinValue;
                    }
                }
            }

            // Get length of long axis
            public float RadiusX
            {
                get
                {
                    if (ValidRegressionModel)
                    {
                        return radiusX;
                    }
                    else
                    {
                        return float.MinValue;
                    }
                }
            }

            // Get length of short axis
            public float RadiusY
            {
                get
                {
                    if (ValidRegressionModel)
                    {
                        return radiusY;
                    }
                    else
                    {
                        return float.MinValue;
                    }
                }
            }

            // Get tilt angle
            public double Tilt
            {
                get
                {
                    if (ValidRegressionModel)
                    {
                        return tilt;
                    }
                    else
                    {
                        return float.MinValue;
                    }
                }
            }

            // Center X0
            public float X0
            {
                get
                {
                    if (ValidRegressionModel)
                    {
                        return x0;
                    }
                    else
                    {
                        return float.MinValue;
                    }
                }
            }

            // Center Y0
            public float Y0
            {
                get
                {
                    if (ValidRegressionModel)
                    {
                        return y0;
                    }
                    else
                    {
                        return float.MinValue;
                    }
                }
            }

            // Calculate the single-point regression error
            public override float CalculateRegressionError(PointF point)
            {
                return CalculateResidual(this, point);
            }
            #endregion
        }

        public struct EllipseConsensusModel
        {
            internal List<PointF> inliers;
            internal List<PointF> outliers;
            internal EllipseModel model;
            internal EllipseModel original;

            #region Public Properties of EllipseConsensusModel
            public List<PointF> Inliers
            {
                get
                {
                    if (model.ValidRegressionModel)
                    {
                        return inliers;
                    }
                    else
                    {
                        return new List<PointF>();
                    }
                }
            }

            public List<PointF> Outliers
            {
                get
                {
                    if (model.ValidRegressionModel)
                    {
                        return outliers;
                    }
                    else
                    {
                        return new List<PointF>();
                    }
                }
            }

            // Get length of long axis
            public float LongAxis
            {
                get
                {
                    if (model.ValidRegressionModel)
                    {
                        return model.long_axis;
                    }
                    else
                    {
                        return float.MinValue;
                    }
                }
            }

            // Get length of short axis
            public float ShortAxis
            {
                get
                {
                    if (model.ValidRegressionModel)
                    {
                        return model.short_axis;
                    }
                    else
                    {
                        return float.MinValue;
                    }
                }
            }

            // Get length of long axis
            public float RadiusX
            {
                get
                {
                    if (model.ValidRegressionModel)
                    {
                        return model.radiusX;
                    }
                    else
                    {
                        return float.MinValue;
                    }
                }
            }

            // Get length of short axis
            public float RadiusY
            {
                get
                {
                    if (model.ValidRegressionModel)
                    {
                        return model.radiusY;
                    }
                    else
                    {
                        return float.MinValue;
                    }
                }
            }

            // Get tilt angle
            public double Tilt
            {
                get
                {
                    if (model.ValidRegressionModel)
                    {
                        return model.tilt;
                    }
                    else
                    {
                        return float.MinValue;
                    }
                }
            }

            // Center X0
            public float X0
            {
                get
                {
                    if (model.ValidRegressionModel)
                    {
                        return model.x0;
                    }
                    else
                    {
                        return float.MinValue;
                    }
                }
            }

            // Center Y0
            public float Y0
            {
                get
                {
                    if (model.ValidRegressionModel)
                    {
                        return model.y0;
                    }
                    else
                    {
                        return float.MinValue;
                    }
                }
            }

            // AverageRegressionError
            public float AverageRegressionError
            {
                get
                {
                    if (model.ValidRegressionModel)
                    {
                        return model.AverageRegressionError;
                    }
                    else
                    {
                        return float.MaxValue;
                    }
                }
            }

            #endregion
        }
        #endregion

        #region Internal Ellipse Structures

        internal struct EllipseBias
        {
            internal double x;
            internal double y;
        }

        internal struct EllipseSummations
        {
            // Initialize all the summations to zero
            internal int N;
            internal double x;
            internal double y;
            internal double x2;
            internal double y2;
            internal double xy;
            internal double x3;
            internal double y3;
            internal double x2y;   // (i.e.  SUM(x^2*y))
            internal double xy2;
            internal double x4;     // (i.e.  SUM(x^4))
            internal double y4;
            internal double x3y;
            internal double x2y2;
            internal double xy3;
        }

        #endregion

        public static EllipseConsensusModel CalculateEllipticalRegressionConsensus(List<PointF> points, float sensitivityInPixels = ERROR_THRESHOLD_ORIGINAL)
        {
            var ellipticalRegressionConsensus = new EllipseConsensusModel();

            if (points == null || points.Count < 5)
            {
                // Exit with error
                return ellipticalRegressionConsensus;
            }

            // Calculate the initial model.  Set the initial inliers and outliers (empty) lists.
            ellipticalRegressionConsensus.inliers = points;
            ellipticalRegressionConsensus.outliers = new List<PointF>();
            ellipticalRegressionConsensus.model = CalculateEllipticalRegressionModel(ellipticalRegressionConsensus.inliers);
            ellipticalRegressionConsensus.original = ellipticalRegressionConsensus.model;

            // Keep removing candidate points until the model is lower than some average error threshold
            while (ellipticalRegressionConsensus.model.AverageRegressionError > sensitivityInPixels && ellipticalRegressionConsensus.model.ValidRegressionModel)
            {
                int index1, index2, index3;
                var pointsWithoutPoint1 = new List<PointF>();
                var pointsWithoutPoint2 = new List<PointF>();
                var pointsWithoutPoint3 = new List<PointF>();
                var candidatePoint1 = GetInsideCandidate(ellipticalRegressionConsensus.inliers, ellipticalRegressionConsensus.model, out index1, out pointsWithoutPoint1);
                var candidatePoint2 = GetOutsideCandidate(ellipticalRegressionConsensus.inliers, ellipticalRegressionConsensus.model, out index2, out pointsWithoutPoint2);
                var candidatePoint3 = GetInfluenceCandidate(ellipticalRegressionConsensus.inliers, ellipticalRegressionConsensus.model, out index3, out pointsWithoutPoint3);

                if (candidatePoint1.IsEmpty || candidatePoint2.IsEmpty || candidatePoint3.IsEmpty || index1 < 0 || index2 < 0 || index3 < 0)
                {
                    // Exit with error
                    break;
                }

                var modelWithoutPoint1 = new EllipseModel();
                var modelWithoutPoint2 = new EllipseModel();
                var modelWithoutPoint3 = new EllipseModel();
                var newAverageError1 = RemovePointAndCalculateError(pointsWithoutPoint1, out modelWithoutPoint1);
                var newAverageError2 = RemovePointAndCalculateError(pointsWithoutPoint2, out modelWithoutPoint2);
                var newAverageError3 = RemovePointAndCalculateError(pointsWithoutPoint3, out modelWithoutPoint3);

                if (newAverageError1 < newAverageError2 && newAverageError1 < newAverageError3)
                {
                    ellipticalRegressionConsensus.inliers = pointsWithoutPoint1;
                    ellipticalRegressionConsensus.outliers.Add(candidatePoint1);
                    ellipticalRegressionConsensus.model = modelWithoutPoint1;
                }
                else if (newAverageError2 < newAverageError3)
                {
                    ellipticalRegressionConsensus.inliers = pointsWithoutPoint2;
                    ellipticalRegressionConsensus.outliers.Add(candidatePoint2);
                    ellipticalRegressionConsensus.model = modelWithoutPoint2;
                }
                else
                {
                    ellipticalRegressionConsensus.inliers = pointsWithoutPoint3;
                    ellipticalRegressionConsensus.outliers.Add(candidatePoint3);
                    ellipticalRegressionConsensus.model = modelWithoutPoint3;
                }
            }

            return ellipticalRegressionConsensus;
        }

        public static EllipseModel CalculateEllipticalRegressionModel(List<PointF> points)
        {
            var ellipticalRegressionModel = new EllipseModel();

            // Calculate the bias
            var bias = CalculateEllipseBias(points);
            if (bias.x == double.MaxValue)
            {
                ellipticalRegressionModel.ValidRegressionModel = false;
                return ellipticalRegressionModel;
            }
            ellipticalRegressionModel.bias = bias;

            // Remove the bias
            var pointsNoBias = RemoveEllipseBias(points, bias);
            if (pointsNoBias == null || pointsNoBias.Count == 0)
            {
                ellipticalRegressionModel.ValidRegressionModel = false;
                return ellipticalRegressionModel;
            }

            // Calculate the summations on the points after the bias has been removed
            var sum = CalculateEllipseSummations(pointsNoBias);
            if (sum.N <= 0)
            {
                ellipticalRegressionModel.ValidRegressionModel = false;
                return ellipticalRegressionModel;
            }
            /////////////////////////////////////////////////ellipticalRegressionModel.sum = sum;

            // Calculate the initial regression model
            ellipticalRegressionModel = CalculateInitialEllipseModel(sum, bias);
            if (!ellipticalRegressionModel.ValidRegressionModel)
            {
                return ellipticalRegressionModel;
            }

            // Calculate a new model by removing the tilt so the ellipse is symmetric about its vertical and horizontal axes.  Doing this 
            // allows simpler calculations of the ellipse features.
            var modelNoTilt = CalculateNoTiltEllipseModel(ellipticalRegressionModel, bias);
            if (!modelNoTilt.ValidRegressionModel)
            {
                ellipticalRegressionModel.ValidRegressionModel = false;
                return ellipticalRegressionModel;
            }

            // Calculate the ellipse features
            CalculateEllipseFeatures(ref ellipticalRegressionModel, modelNoTilt);
            if (!ellipticalRegressionModel.ValidRegressionModel)
            {
                return ellipticalRegressionModel;
            }
            ellipticalRegressionModel.ValidRegressionModel = true;

            // Calculate the average residual error
            ellipticalRegressionModel.AverageRegressionError = ellipticalRegressionModel.CalculateAverageRegressionError(points);
            if (ellipticalRegressionModel.AverageRegressionError == float.MaxValue)
            {
                ellipticalRegressionModel.ValidRegressionModel = false;
            }

            return ellipticalRegressionModel;
        }

        public static float CalculateResidual(EllipseModel model, PointF point)
        {
            var x = point.X;
            var y = point.Y;

            var errorV1 = Math.Abs(ModeledY(model, x, EllipseHalves.TopHalf) - y);
            var errorV2 = Math.Abs(ModeledY(model, x, EllipseHalves.BottomHalf) - y);
            var errorV = Math.Min(errorV1, errorV2);
            var errorH1 = Math.Abs(ModeledX(model, y, EllipseHalves.RightHalf) - x);
            var errorH2 = Math.Abs(ModeledX(model, y, EllipseHalves.LeftHalf) - x);
            var errorH = Math.Min(errorH1, errorH2);
            var error = Math.Min(errorV, errorH);

            if (error >= 0.0f && error != float.MaxValue)
            {
                return error;
            }
            else if (!point.IsEmpty && point.X != float.MaxValue && point.Y != float.MaxValue)
            {
                return Math.Max(Math.Abs(model.X0 - point.X), Math.Abs(model.Y0 - point.Y));
            }
            else
            {
                return float.MaxValue;
            }
        }

        #region Elliptical Regression Private Helper Functions
        private static EllipseBias CalculateEllipseBias(List<PointF> points)
        {
            EllipseBias bias;
            if (points == null || points.Count < 5)
            {
                // The minimum number of points to define an elliptical regression is 5
                bias.x = double.MaxValue;
                bias.y = double.MaxValue;
                return bias;
            }

            // Shorthand that better matches the math formulas
            var N = points.Count;

            //// Remove the bias (i.e. center the data at zero)
            //// Calculate the mean of a set of points
            var meanX = 0.0;
            var meanY = 0.0;
            for (var i = 0; i < N; ++i)
            {
                meanX += points[i].X;
                meanY += points[i].Y;
            }
            meanX /= (float)N;
            meanY /= (float)N;
            bias.x = meanX;
            bias.y = meanY;

            return bias;
        }

        private static List<PointF> RemoveEllipseBias(List<PointF> points, EllipseBias bias)
        {
            if (points == null || points.Count < 5)
            {
                return new List<PointF>();
            }

            if (bias.x == double.MaxValue)
            {
                return new List<PointF>();
            }

            // Shorthand that better matches the math formulas
            var N = points.Count;

            //// Remove the mean from the set of points
            var pointsNoBias = new List<PointF>();
            for (var i = 0; i < N; ++i)
            {
                var x = points[i].X - (float)bias.x;
                var y = points[i].Y - (float)bias.y;
                pointsNoBias.Add(new PointF(x, y));
            }

            return pointsNoBias;
        }

        private static EllipseSummations CalculateEllipseSummations(List<PointF> points)
        {
            var sum = new EllipseSummations();
            if (points == null || points.Count < 5)
            {
                sum.N = 0;
                return sum;
            }

            // Initialize all the summations to zero
            sum.x = 0.0;
            sum.y = 0.0;
            sum.x2 = 0.0;
            sum.y2 = 0.0;
            sum.xy = 0.0;
            sum.x3 = 0.0;
            sum.y3 = 0.0;
            sum.x2y = 0.0;  // (i.e.  SUM(x^2*y))
            sum.xy2 = 0.0;
            sum.x4 = 0.0;   // (i.e.  SUM(x^4))
            sum.y4 = 0.0;
            sum.x3y = 0.0;
            sum.x2y2 = 0.0;
            sum.xy3 = 0.0;

            // Shorthand that better matches the math formulas
            var N = sum.N = points.Count;

            // Calculate the summations
            for (var i = 0; i < N; ++i)
            {
                // Shorthand
                var x   = points[i].X;
                var y   = points[i].Y;
                var xx  = x * x;
                var xy  = x * y;
                var yy  = y * y;

                // Sums
                sum.x       += x;
                sum.y       += y;
                sum.x2      += xx;
                sum.y2      += yy;
                sum.xy      += xy;
                sum.x3      += x * xx;
                sum.y3      += y * yy;
                sum.x2y     += xx * y;
                sum.xy2     += x * yy;
                sum.x4      += xx * xx;
                sum.y4      += yy * yy;
                sum.x3y     += xx * xy;
                sum.x2y2    += xx * yy;
                sum.xy3     += xy * yy;
            }

            return sum;
        }

        private static EllipseModel CalculateInitialEllipseModel(EllipseSummations sum, EllipseBias bias)
        {
            var model = new EllipseModel();
            if (sum.N <= 0)
            {
                model.ValidRegressionModel = false;
                return model;
            }

            model.bias = bias;

            // Calculate A = INV(X'X) * X 
            //     or    A = INV(S)   * X

            Matrix<double> S = Matrix<double>.Build.Dense(5, 5);    // X'X
            Vector<double> X = Vector<double>.Build.Dense(5);       // X' = [Sx2 Sxy Sy2 Sx Sy]
            Vector<double> A = Vector<double>.Build.Dense(5);       // A  = [a b c d e]

            // Fill the matrices
            X[0] = sum.x2;
            X[1] = sum.xy;
            X[2] = sum.y2;
            X[3] = sum.x;
            X[4] = sum.y;

            // S[column, row]
            S[0, 0] = sum.x4;
            S[1, 0] = S[0, 1] = sum.x3y;
            S[2, 0] = S[1, 1] = S[0, 2] = sum.x2y2;
            S[3, 0] = S[0, 3] = sum.x3;
            S[4, 0] = S[3, 1] = S[1, 3] = S[0, 4] = sum.x2y;
            S[2, 1] = S[1, 2] = sum.xy3;
            S[2, 2] = sum.y4;
            S[4, 1] = S[3, 2] = S[2, 3] = S[1, 4] = sum.xy2;
            S[4, 2] = S[2, 4] = sum.y3;
            S[3, 3] = sum.x2;
            S[4, 3] = S[3, 4] = sum.xy;
            S[4, 4] = sum.y2;

            // Check for a singular matrix
            if (Math.Abs(S.Determinant()) <= EPSILON)
            {
                model.ValidRegressionModel = false;
                return model;
            }

            // A = INV(S) * X
            A = S.Inverse() * X;

            // Calculate the coefficients of ax^2 + bxy + cy^2 + dx + ey + f = 0
            model.a = A[0];
            model.b = A[1];
            model.c = A[2];
            model.d = A[3];
            model.e = A[4];
            model.f = -1.0;

            // Shorthand
            var a = model.a;
            var b = model.b;
            var c = model.c;

            if (Math.Abs(b / a) > EPSILON || Math.Abs(b / c) > EPSILON)
            {
                // Tilt angle is not zero
                model.tilt = 0.5 * Math.Atan(b / (a - c));
            }
            else
            {
                model.tilt = 0.0;
            }

            model.ValidRegressionModel = true;

            return model;
        }

        private static EllipseModel CalculateNoTiltEllipseModel(EllipseModel modelWithTilt, EllipseBias bias)
        {
            var modelNoTilt = new EllipseModel();

            // Check input parameters
            if (!modelWithTilt.ValidRegressionModel || bias.x == double.MaxValue)
            {
                modelNoTilt.ValidRegressionModel = false;
                return modelNoTilt;
            }

            // Shorthand
            var a = modelWithTilt.a;
            var b = modelWithTilt.b;
            var c = modelWithTilt.c;
            var d = modelWithTilt.d;
            var e = modelWithTilt.e;
            var f = modelWithTilt.f;

            // Remove the orientation from the ellipse
            var cos_phi = 1.0; // Default value for zero rotation angle
            var sin_phi = 0.0; // Default value for zero rotation angle
            if (Math.Abs(modelWithTilt.tilt) > EPSILON)
            {
                // Tilt angle is not zero
                cos_phi = Math.Cos(modelWithTilt.tilt);
                sin_phi = Math.Sin(modelWithTilt.tilt);
                modelNoTilt.a = a * cos_phi * cos_phi + b * cos_phi * sin_phi + c * sin_phi * sin_phi;
                modelNoTilt.c = a * sin_phi * sin_phi - b * cos_phi * sin_phi + c * cos_phi * cos_phi;
                modelNoTilt.d = d * cos_phi + e * sin_phi;
                modelNoTilt.e = -d * sin_phi + e * cos_phi;

                modelNoTilt.bias.x = cos_phi * bias.x + sin_phi * bias.y;
                modelNoTilt.bias.y = -sin_phi * bias.x + cos_phi * bias.y;
            }
            else
            {
                // Tilt angle is zero
                modelNoTilt.a = a;
                modelNoTilt.c = c;
                modelNoTilt.d = d;
                modelNoTilt.e = e;
                modelNoTilt.bias = bias;
            }

            // Common sets
            modelNoTilt.b = 0.0;
            modelNoTilt.f = -1.0;
            modelNoTilt.tilt = 0.0;

            // Check if the model is actually a parabola (aa*cc==0) or a hyperbola
            if (modelNoTilt.a * modelNoTilt.c <= 0.0f)
            {
                // Stop here
                modelNoTilt.ValidRegressionModel = true;
                return modelNoTilt;
            }

            // Force coefficient a to be positive
            if (modelNoTilt.a < 0.0)
            {
                modelNoTilt.a = -modelNoTilt.a;
                modelNoTilt.c = -modelNoTilt.c;
                modelNoTilt.d = -modelNoTilt.d;
                modelNoTilt.e = -modelNoTilt.e;
            }

            // Shorthand
            var aa = modelNoTilt.a;
            var cc = modelNoTilt.c;
            var dd = modelNoTilt.d;
            var ee = modelNoTilt.e;

            modelNoTilt.f = 1.0 + (dd * dd) / (4.0 * aa) + (ee * ee) / (4.0 * cc);          // fit_ellipse.m // F = -1 // F' = -F + (D*D / 4A) + (E*E / 4C)

            modelNoTilt.x0 = (float)(modelNoTilt.bias.x - dd / (2.0 * aa));                             // Slide 17 // h = -D / (2A) ..... if B is zero
            modelNoTilt.y0 = (float)(modelNoTilt.bias.y - ee / (2.0 * cc));                             // Slide 17 // k = -E / (2C) ..... if B is zero

            modelNoTilt.ValidRegressionModel = true;

            return modelNoTilt;
        }

        private static void CalculateEllipseFeatures(ref EllipseModel modelWithTilt, EllipseModel modelNoTilt)
        {
            // Short-hand
            var cos_phi = Math.Cos(modelWithTilt.tilt);
            var sin_phi = Math.Sin(modelWithTilt.tilt);

            modelWithTilt.x0 = (float)cos_phi * modelNoTilt.x0 - (float)sin_phi * modelNoTilt.y0;
            modelWithTilt.y0 = (float)sin_phi * modelNoTilt.x0 + (float)cos_phi * modelNoTilt.y0;
            modelWithTilt.radiusX = (float)Math.Sqrt(modelNoTilt.f / modelNoTilt.a);
            modelWithTilt.radiusY = (float)Math.Sqrt(modelNoTilt.f / modelNoTilt.c);
            modelWithTilt.long_axis = 2.0f * Math.Max(modelWithTilt.radiusX, modelWithTilt.radiusY);
            modelWithTilt.short_axis = 2.0f * Math.Min(modelWithTilt.radiusX, modelWithTilt.radiusY);

            // Short-hand
            var x0 = modelWithTilt.x0;
            var y0 = modelWithTilt.y0;
            var radiusX = modelWithTilt.radiusX;
            var radiusY = modelWithTilt.radiusY;

            // Actual polynomial parameters.  There seems to be a bug in the existing values.
            // Deriving the values from the tilt, center, and radii.
            modelWithTilt.a = (radiusY * cos_phi) * (radiusY * cos_phi) + (radiusX * sin_phi) * (radiusX * sin_phi);
            modelWithTilt.b = -2.0 * cos_phi * sin_phi * (radiusX * radiusX - radiusY * radiusY);
            modelWithTilt.c = (radiusY * sin_phi) * (radiusY * sin_phi) + (radiusX * cos_phi) * (radiusX * cos_phi);

            // Short-hand
            var a = modelWithTilt.a;
            var b = modelWithTilt.b;
            var c = modelWithTilt.c;

            modelWithTilt.d = -2.0 * a * x0 - y0 * b;
            modelWithTilt.e = -2.0 * c * y0 - x0 * b;
            modelWithTilt.f = -(radiusX * radiusX * radiusY * radiusY) + a * x0 * x0 + b * x0 * y0 + c * y0 * y0;
        }
        #endregion

        #region Elliptical Consensus Private Helper Functions
        private static PointF GetInsideCandidate(List<PointF> points, EllipseModel ellipse, out int index, out List<PointF> pointsWithoutCandidate)
        {
            if (points == null || points.Count <= 5)
            {
                index = -1;
                pointsWithoutCandidate = new List<PointF>();
                return new PointF();
            }

            var maxRegressionError = float.MinValue;
            index = 0;
            for (var i = 0; i < points.Count; ++i)
            {
                var point = points[i];
                if (WhichSideOfEllipse(ellipse, point) == SideOfEllipse.Inside)
                {
                    var error = CalculateResidual(ellipse, point);
                    if (error > maxRegressionError)
                    {
                        maxRegressionError = error;
                        index = i;
                    }
                }
            }

            pointsWithoutCandidate = new List<PointF>(points);
            pointsWithoutCandidate.RemoveAt(index);

            return points[index];
        }

        private static PointF GetOutsideCandidate(List<PointF> points, EllipseModel ellipse, out int index, out List<PointF> pointsWithoutCandidate)
        {
            if (points == null || points.Count <= 5)
            {
                index = -1;
                pointsWithoutCandidate = new List<PointF>();
                return new PointF();
            }

            var maxRegressionError = float.MinValue;
            index = 0;
            for (var i = 0; i < points.Count; ++i)
            {
                var point = points[i];
                if (WhichSideOfEllipse(ellipse, point) == SideOfEllipse.Outside)
                {
                    var error = CalculateResidual(ellipse, point);
                    if (error > maxRegressionError)
                    {
                        maxRegressionError = error;
                        index = i;
                    }
                }
            }

            pointsWithoutCandidate = new List<PointF>(points);
            pointsWithoutCandidate.RemoveAt(index);

            return points[index];
        }

        private static PointF GetInfluenceCandidate(List<PointF> points, EllipseModel ellipse, out int index, out List<PointF> pointsWithoutCandidate)
        {
            if (points == null || points.Count <= 5)
            {
                index = -1;
                pointsWithoutCandidate = new List<PointF>();
                return new PointF();
            }

            index = 0;
            double maxPower = 0;
            for (var i = 0; i < points.Count; ++i)
            {
                var point = points[i];
                var power = Math.Abs(point.X * point.Y);
                if (power > maxPower)
                {
                    maxPower = power;
                    index = i;
                }
            }

            pointsWithoutCandidate = new List<PointF>(points);
            pointsWithoutCandidate.RemoveAt(index);

            return points[index];
        }

        private static float RemovePointAndCalculateError(List<PointF> pointsWithoutCandidate, out EllipseModel modelWithoutCandidate)
        {
            modelWithoutCandidate = CalculateEllipticalRegressionModel(pointsWithoutCandidate);
            return modelWithoutCandidate.AverageRegressionError;
        }

        #endregion

        #region Ellipse Private Utilities

        // Returns the real roots of the quadratic equation
        // root1 > root2
        private static void QuadraticEquation(double a, double b, double c, out int numberOfRoots, out float root1, out float root2)
        {
            // -b +/- sqrt( b^2 - 4ac )
            // ------------------------
            //           2a

            root1 = float.MinValue;
            root2 = float.MinValue;

            var discriminant = b * b - 4 * a * c;
            if (discriminant > 0.0)
            {
                numberOfRoots = 2;
                root1 = (float)((-b + Math.Sqrt(discriminant)) / (2.0 * a));
                root2 = (float)((-b - Math.Sqrt(discriminant)) / (2.0 * a));
            }
            else if (discriminant == 0.0)
            {
                numberOfRoots = 1;
                root1 = (float)(-b / (2.0 * a));
                root2 = root1;
            }
            else
            {
                numberOfRoots = 0;
            }
        }

        #endregion

        #region Ellipse Public Utilities

        public enum SideOfEllipse
        {
            Inside = -1,
            OnPerimeter = 0,
            Outside = 1
        }

        public static SideOfEllipse WhichSideOfEllipse(EllipseModel ellipse, PointF point)
        {
            const double ON_PERIMETER_THRESHOLD = 1.0;  // A high degree of precision is required to label a point on the perimeter

            var x = (double)point.X;
            var y = (double)point.Y;

            var a = ellipse.a;
            var b = ellipse.b;
            var c = ellipse.c;
            var d = ellipse.d;
            var e = ellipse.e;
            var f = ellipse.f;

            var testMetric =    a * x * x + 
                                b * x * y + 
                                c * y * y + 
                                d * x + 
                                e * y + 
                                f;

            if (testMetric < -ON_PERIMETER_THRESHOLD)
            {
                return SideOfEllipse.Inside;
            }
            else if (testMetric > ON_PERIMETER_THRESHOLD)
            {
                return SideOfEllipse.Outside;
            }
            else
            {
                return SideOfEllipse.OnPerimeter;
            }
        }

        // Returns the modeled y-value of an ellipse
        public static float ModeledY(EllipseModel model, float x, EllipseHalves half = EllipseHalves.TopHalf)
        {
            if (!model.ValidRegressionModel)
            {
                // ERROR:  Invalid model
                return float.MinValue;
            }
            else if (half == EllipseHalves.LeftHalf || half == EllipseHalves.RightHalf)
            {
                // ERROR:  Invalid EllipseHalves parameter
                return float.MinValue;
            }

            // Short-hand
            var a = model.a;
            var b = model.b;
            var c = model.c;
            var d = model.d;
            var e = model.e;
            var f = model.f;

            int numberOfRoots;
            float root1, root2;
            QuadraticEquation(c, (b * x + e), (a * x * x + d * x + f), out numberOfRoots, out root1, out root2);
            if (numberOfRoots > 0)
            {
                if (half == EllipseHalves.TopHalf)
                {
                    return root1;
                }
                else
                {
                    return root2;
                }
            }
            else
            {
                // No y-value for this x-value
                return float.MinValue;
            }
        }

        // Returns the modeled x-value of an ellipse
        public static float ModeledX(EllipseModel model, float y, EllipseHalves half = EllipseHalves.RightHalf)
        {
            if (!model.ValidRegressionModel)
            {
                // ERROR:  Invalid model
                return float.MinValue;
            }
            else if (half == EllipseHalves.TopHalf || half == EllipseHalves.BottomHalf)
            {
                // ERROR:  Invalid EllipseHalves parameter
                return float.MinValue;
            }

            // Short-hand
            var a = model.a;
            var b = model.b;
            var c = model.c;
            var d = model.d;
            var e = model.e;
            var f = model.f;

            int numberOfRoots;
            float root1, root2;
            QuadraticEquation(a, (b * y + d), (c * y * y + e * y + f), out numberOfRoots, out root1, out root2);
            if (numberOfRoots > 0)
            {
                if (half == EllipseHalves.RightHalf)
                {
                    return root1;
                }
                else
                {
                    return root2;
                }
            }
            else
            {
                // No x-value for this y-value
                return float.MinValue;
            }
        }

        public static float PointToPolarAngle(PointF point, EllipseModel ellipse)
        {
            if (!ellipse.ValidRegressionModel)
            {
                return 0.0f;
            }

            var deltaX = point.X - ellipse.X0;
            var deltaY = point.Y - ellipse.Y0;

            // Note:  This calculation does not take ellipse eccentricity into account.  To calculate the 
            //        parametric angle, tau, use PointToParametricAngle instead.
            var thetaNoTilt = (float)Math.Atan2(deltaY, deltaX);
            var theta = thetaNoTilt - (float)ellipse.Tilt;

            // Return an angle in the range [0, 2pi)
            const float TWO_PI = 2.0f * (float)Math.PI;
            return (theta + TWO_PI) % TWO_PI;
        }

        public static float PointToParametricAngle(PointF point, EllipseModel ellipse)
        {
            if (!ellipse.ValidRegressionModel)
            {
                return 0.0f;
            }

            var deltaX = point.X - ellipse.X0;
            var deltaY = point.Y - ellipse.Y0;

            // Note:  If this were a circle, we would use the following
            var thetaNoTilt = (float)Math.Atan2(deltaY, deltaX);
            var theta = thetaNoTilt - (float)ellipse.Tilt;

            // Return an angle in the range [0, 2pi)
            const float TWO_PI = 2.0f * (float)Math.PI;
            theta = (theta + TWO_PI) % TWO_PI;

            // Note (cont.):  Since it is an ellipse, we are actually calculating the parametric angle, tau, which is related to 
            //                the polar angle, theta, by the following equation from http://mathworld.wolfram.com/Ellipse.html 
            //                "The relationship between the polar angle from the ellipse center theta and the parameter tau follows 
            //                    theta = atan(b / a * tan(tau))"
            //                        or
            //                    tau = atan(a / b * tan(theta))
            //
            const float _90 = (float)Math.PI / 2.0f;
            const float _180 = (float)Math.PI;
            const float _270 = 3.0f * (float)Math.PI / 2.0f;

            float tau;
            if (theta >= 0.0f && theta < _90)
            {
                tau = (float)Math.Atan2(Math.Abs(ellipse.RadiusX / ellipse.RadiusY * (float)Math.Tan(theta)), 1.0f);
            }
            else if (theta >= _90 && theta < _180)
            {
                tau = (float)Math.Atan2(Math.Abs(ellipse.RadiusX / ellipse.RadiusY * (float)Math.Tan(theta)), -1.0f);
            }
            else if (theta >= _180 && theta < _270)
            {
                tau = (float)Math.Atan2(-1.0f * Math.Abs(ellipse.RadiusX / ellipse.RadiusY * (float)Math.Tan(theta)), -1.0f);
            }
            else // if (theta >= _270 && theta < _360)
            {
                tau = (float)Math.Atan2(-1.0f * Math.Abs(ellipse.RadiusX / ellipse.RadiusY * (float)Math.Tan(theta)), 1.0f);
            }

            // Return an angle in the range [0, 2pi)
            return (tau + TWO_PI) % TWO_PI;
        }

        public static PointF RadialAngleToEllipsePoint(float tauRadians, EllipseModel ellipse)
        {
            if (!ellipse.ValidRegressionModel)
            {
                return new PointF();
            }

            var x0 = ellipse.X0;
            var y0 = ellipse.Y0;
            var a = ellipse.RadiusX;
            var b = ellipse.RadiusY;
            var tilt = ellipse.Tilt;

            // For the math, see slide 11 of https://www.cs.cornell.edu/cv/OtherPdf/Ellipse.pdf
            //
            var x = x0 + a * Math.Cos(tilt) * Math.Cos(tauRadians) - b * Math.Sin(tilt) * Math.Sin(tauRadians);
            var y = y0 + a * Math.Sin(tilt) * Math.Cos(tauRadians) + b * Math.Cos(tilt) * Math.Sin(tauRadians);

            return new PointF((float)x, (float)y);
        }

        #endregion

        #region Unit Testing
        public static void UnitTest1(out List<PointF> points, out EllipseModel fit, out List<PointF> outliers, out EllipseModel orig)
        {
            ///////////////////
            // Unit test #1: //
            ///////////////////

            // A centered ellipse using the equation:     x^2/4 + y^2/9 = 1
            // [-2 0]
            // [+2 0]
            // [0 -3]
            // [0 +3]
            // [1 sqrt(6.75)]

            points = new List<PointF>();
            points.Add(new PointF(-2.0f, 0.0f));
            points.Add(new PointF(2.0f, 0.0f));
            points.Add(new PointF(0.0f, -3.0f));
            points.Add(new PointF(0.0f, 3.0f));
            points.Add(new PointF(1.0f, (float)Math.Sqrt(6.75)));
            points.Add(new PointF(1.0f, -(float)Math.Sqrt(6.75)));

            var model = CalculateEllipticalRegressionConsensus(points);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }

        public static void UnitTest2(out List<PointF> points1b, out EllipseModel fit, out List<PointF> outliers, out EllipseModel orig)
        {
            ///////////////////
            // Unit test #1b: //
            ///////////////////

            // A centered ellipse using the equation:     x^2/4 + y^2/9 = 1
            // [-2 0]
            // [+2 0]
            // [0 -3]
            // [0 +3]
            // [1 sqrt(6.75)]

            points1b = new List<PointF>();
            points1b.Add(new PointF(498.0f, 400.0f));
            points1b.Add(new PointF(502.0f, 400.0f));
            points1b.Add(new PointF(500.0f, 397.0f));
            points1b.Add(new PointF(500.0f, 403.0f));
            points1b.Add(new PointF(501.0f, 400.0f + (float)Math.Sqrt(6.75)));
            points1b.Add(new PointF(501.0f, 400.0f - (float)Math.Sqrt(6.75)));

            var model = CalculateEllipticalRegressionConsensus(points1b);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }

        public static void UnitTest3(out List<PointF> points2, out EllipseModel fit, out List<PointF> outliers, out EllipseModel orig)
        {
            ///////////////////
            // Unit test #2: //
            ///////////////////

            // A off-center ellipse with 30 degree tilt:
            //              Center: (2,1)
            //                   a: 5
            //                   b: 3
            //                Tilt: 30 degrees counter-clockwise
            // [6.33 3.5]
            // [5.0  4.46]
            // [2.87 4.5]
            // [0.5  3.60]
            // [-1.46 2.0]
            // [-2.5  0.134]
            // [-2.33 -1.5]

            points2 = new List<PointF>();
            points2.Add(new PointF(6.33013f, 3.5f));
            points2.Add(new PointF(5.0f, 4.46410f));
            points2.Add(new PointF(2.86603f, 4.5f));
            points2.Add(new PointF(0.5f, 3.59808f));
            points2.Add(new PointF(-1.46410f, 2.0f));
            points2.Add(new PointF(-2.5f, 0.13397f));
            points2.Add(new PointF(-2.33013f, -1.5f));

            var model = CalculateEllipticalRegressionConsensus(points2);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }

        public static void UnitTest4(out List<PointF> points3, out EllipseModel fit, out List<PointF> outliers, out EllipseModel orig)
        {
            ///////////////////
            // Unit test #3: //
            ///////////////////

            // A centered ellipse using the equation:     x^2/4 + y^2/9 = 1
            // [-2 0]
            // [+2 0]
            // [0 -3]
            // [0 +3]
            // [1 sqrt(6.75)]

            points3 = new List<PointF>();
            points3.Add(new PointF(-2.0f, 0.0f));
            points3.Add(new PointF(2.0f, 0.0f));
            points3.Add(new PointF(0.0f, -3.0f));
            points3.Add(new PointF(0.0f, 3.0f));
            points3.Add(new PointF(1.0f, (float)Math.Sqrt(6.75)));
            points3.Add(new PointF(1.0f, -(float)Math.Sqrt(6.75)));
            points3.Add(new PointF(-1.0f, (float)Math.Sqrt(6.75)));
            points3.Add(new PointF(-1.0f, -(float)Math.Sqrt(6.75)));
            points3.Add(new PointF(0.0f, 1.0f));

            var model = CalculateEllipticalRegressionConsensus(points3);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }

        public static void UnitTest5(out List<PointF> points3, out EllipseModel fit, out List<PointF> outliers, out EllipseModel orig)
        {
            ///////////////////
            // Unit test #3: //
            ///////////////////

            // A centered ellipse using the equation:     x^2/4 + y^2/9 = 1
            // [-2 0]
            // [+2 0]
            // [0 -3]
            // [0 +3]
            // [1 sqrt(6.75)]

            points3 = new List<PointF>();
            points3.Add(new PointF(-2.0f, 0.0f));
            points3.Add(new PointF(2.0f, 0.0f));
            points3.Add(new PointF(0.0f, -3.0f));
            points3.Add(new PointF(0.0f, 3.0f));
            points3.Add(new PointF(1.0f, (float)Math.Sqrt(6.75)));
            points3.Add(new PointF(1.0f, -(float)Math.Sqrt(6.75)));
            points3.Add(new PointF(-1.0f, (float)Math.Sqrt(6.75)));
            points3.Add(new PointF(-1.0f, -(float)Math.Sqrt(6.75)));
            points3.Add(new PointF(4.0f, 4.0f));
            points3.Add(new PointF(3.0f, 3.0f));
            points3.Add(new PointF(2.5f, 2.5f));

            var model = CalculateEllipticalRegressionConsensus(points3);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }

        public static void UnitTest6(out List<PointF> points4, out EllipseModel fit, out List<PointF> outliers, out EllipseModel orig)
        {
            ///////////////////
            // Unit test #4: //
            ///////////////////

            // A off-center ellipse with 30 degree tilt:
            //              Center: (2,1)
            //                   a: 5
            //                   b: 3
            //                Tilt: 30 degrees counter-clockwise
            // [6.33 3.5]
            // [5.0  4.46]
            // [2.87 4.5]
            // [0.5  3.60]
            // [-1.46 2.0]
            // [-2.5  0.134]
            // [-2.33 -1.5]

            // Plus one noisy points

            points4 = new List<PointF>();
            points4.Add(new PointF(4.0f, -7.0f));
            points4.Add(new PointF(4.0f, -3.0f));
            points4.Add(new PointF(6.33013f, 3.5f));
            points4.Add(new PointF(5.0f, 4.46410f));
            points4.Add(new PointF(2.86603f, 4.5f));
            points4.Add(new PointF(0.5f, 3.59808f));
            points4.Add(new PointF(-1.46410f, 2.0f));
            points4.Add(new PointF(-2.5f, 0.13397f));
            points4.Add(new PointF(-2.33013f, -1.5f));
            points4.Add(new PointF(5.8f, 0.45f));
            points4.Add(new PointF(0.63f, -2.5f));

            var model = CalculateEllipticalRegressionConsensus(points4);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }
        #endregion
    }
}
