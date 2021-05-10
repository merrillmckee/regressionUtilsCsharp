using System;

// For matrix inversion
using MathNet.Numerics.LinearAlgebra;

// For unit testing
using System.Collections.Generic;
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

            public EllipseModel()
            {
                MinimumPoints = 5;
            }

            public EllipseModel(EllipseModel copy) : base(copy)
            {
                coefficients = copy.coefficients;
                x0 = copy.x0;
                y0 = copy.y0;
                tilt = copy.tilt;
                radiusX = copy.radiusX;
                radiusY = copy.radiusY;
                long_axis = copy.long_axis;
                short_axis = copy.short_axis;
            }

            public override RegressionModel Clone()
            {
                return new EllipseModel(this);
            }

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

            protected class EllipseSummations : Summations
            {
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

            public override Summations CalculateSummations(List<PointF> points)
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
                    var x = points[i].X;
                    var y = points[i].Y;
                    var xx = x * x;
                    var xy = x * y;
                    var yy = y * y;

                    // Sums
                    sum.x += x;
                    sum.y += y;
                    sum.x2 += xx;
                    sum.y2 += yy;
                    sum.xy += xy;
                    sum.x3 += x * xx;
                    sum.y3 += y * yy;
                    sum.x2y += xx * y;
                    sum.xy2 += x * yy;
                    sum.x4 += xx * xx;
                    sum.y4 += yy * yy;
                    sum.x3y += xx * xy;
                    sum.x2y2 += xx * yy;
                    sum.xy3 += xy * yy;
                }

                return sum;
            }

            public override void CalculateModel(Summations sums)
            {
                EllipseSummations sum = sums as EllipseSummations;

                if (sum.N <= 0)
                {
                    ValidRegressionModel = false;
                    return;
                }

                // Calculate A = INV(X'X) * X 
                //     or    A = INV(S)   * X

                Matrix<double> S = Matrix<double>.Build.Dense(5, 5);    // X'X
                Vector<double> X = Vector<double>.Build.Dense(5);       // X' = [Sx2 Sxy Sy2 Sx Sy]
                Vector<double> A;// = Vector<double>.Build.Dense(5);       // A  = [a b c d e]

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
                    ValidRegressionModel = false;
                    return;
                }

                A = S.Inverse() * X;

                // Calculate the coefficients of ax^2 + bxy + cy^2 + dx + ey + f = 0
                a = A[0];
                b = A[1];
                c = A[2];
                d = A[3];
                e = A[4];
                f = -1.0;

                if (Math.Abs(b / a) > EPSILON || Math.Abs(b / c) > EPSILON)
                {
                    // Tilt angle is not zero
                    tilt = 0.5 * Math.Atan(b / (c - a)); // todo: move tilt to CalculateFeatures
                }
                else
                {
                    tilt = 0.0;
                }

                ValidRegressionModel = true;
            }

            public override void CalculateFeatures()
            {
                // Short-hand to match notation in https://www.mathworks.com/matlabcentral/fileexchange/3215-fit_ellipse
                var A = this.a;
                var B = this.b;
                var C = this.c;
                var D = this.d;
                var E = this.e;
                var c = Math.Cos(tilt); // locally redefines "c" to match link notation; careful with "c" vs "C"
                var s = Math.Sin(tilt);

                // Create a model ellipse with the tilt removed
                //    ap*xx + bp*xy + cp*yy + dp*x + ep*y + fp = 0   // where ap denotes a' or "a prime"
                // Substitute x with "cx+sy" and y with "-sx+cy"
                var cc = c * c;
                var ss = s * s;
                var cs = c * s;
                var Ap = A * cc - B * cs + C * ss;
                var Bp = 2.0 * A * cs + (cc - ss) * B - 2.0 * C * cs; // zero if the tilt is correctly removed
                var Cp = A * ss + B * cs + C * cc;
                var Dp = D * c - E * s;
                var Ep = D * s + E * c;
                /// Fp = -1.0
                var Fpp = 1.0 + (Dp * Dp) / (4.0 * Ap) + (Ep * Ep) / (4.0 * Cp);

                if (Math.Abs(Bp) > EPSILON)
                {
                    Console.WriteLine("");
                    //throw new Exception("CalculateFeatures:  When de-tilting ellipse, the new ellipse does not have a zero coefficient for the xy term");
                }

                if (Ap < 0)
                {
                    Ap *= -1.0;
                    Cp *= -1.0;
                    Dp *= -1.0;
                    Ep *= -1.0;
                }

                // Features
                var x0p = -Dp / (2.0 * Ap) + c * bias.x - s * bias.y; // center of de-tilted ellipse
                var y0p = -Ep / (2.0 * Cp) + s * bias.x + c * bias.y; // center of de-tilted ellipse
                x0 = (float)(c * x0p + s * y0p);
                y0 = (float)(-s * x0p + c * y0p);
                radiusX = (float)Math.Sqrt(Math.Abs(Fpp / Ap));
                radiusY = (float)Math.Sqrt(Math.Abs(Fpp / Cp));
                long_axis = 2.0f * (float)Math.Max(radiusX, radiusY);
                short_axis = 2.0f * (float)Math.Min(radiusX, radiusY);
            }
        }

        public class EllipseConsensusModel : RegressionConsensusModel
        {
            public EllipseConsensusModel()
            {
                inliers = null;
                outliers = null;
                model = new EllipseModel();
                original = new EllipseModel();
                influenceError = InfluenceError.L2;
            }

            protected override float CalculateError(RegressionModel modelr, PointF point, out bool pointOnPositiveSide)
            {
                EllipseModel model = modelr as EllipseModel;

                var whichSideOfEllipse = WhichSideOfEllipse(model, point);
                if (whichSideOfEllipse == SideOfEllipse.OnPerimeter)
                {
                    pointOnPositiveSide = true;
                    return 0;
                }
                else if (whichSideOfEllipse == SideOfEllipse.Outside)
                {
                    pointOnPositiveSide = true;
                }
                else
                {
                    pointOnPositiveSide = false;
                }

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
        }
        #endregion

        public static EllipseConsensusModel CalculateEllipticalRegressionConsensus(List<PointF> points, float sensitivityInPixels = ERROR_THRESHOLD_ORIGINAL)
        {
            var consensus = new EllipseConsensusModel();
            consensus.Calculate(points, sensitivityInPixels);

            return consensus;
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
            const double ON_PERIMETER_THRESHOLD = 0.0001;  // A high degree of precision is required to label a point on the perimeter

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
        public static float ModeledY(EllipseModel model, float x_orig, EllipseHalves half = EllipseHalves.TopHalf)
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

            var x = x_orig - model.bias.x;

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
                    return root1 + (float)model.bias.y;
                }
                else
                {
                    return root2 + (float)model.bias.y;
                }
            }
            else
            {
                // No y-value for this x-value
                return float.MinValue;
            }
        }

        // Returns the modeled x-value of an ellipse
        public static float ModeledX(EllipseModel model, float y_orig, EllipseHalves half = EllipseHalves.RightHalf)
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

            var y = y_orig - model.bias.y;

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
                    return root1 + (float)model.bias.x;
                }
                else
                {
                    return root2 + (float)model.bias.x;
                }
            }
            else
            {
                // No x-value for this y-value
                return float.MinValue;
            }
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
        public static EllipseConsensusModel UnitTest1(out List<PointF> points)
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

            return CalculateEllipticalRegressionConsensus(points);
        }

        public static EllipseConsensusModel UnitTest2(out List<PointF> points1b)
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

            return CalculateEllipticalRegressionConsensus(points1b);
        }

        public static EllipseConsensusModel UnitTest3(out List<PointF> points2)
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

            return CalculateEllipticalRegressionConsensus(points2);
        }

        public static EllipseConsensusModel UnitTest4(out List<PointF> points3)
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
            points3.Add(new PointF(0.0f, 1.0f)); // internal outlier

            return CalculateEllipticalRegressionConsensus(points3);
        }

        public static EllipseConsensusModel UnitTest5(out List<PointF> points3)
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

            return CalculateEllipticalRegressionConsensus(points3);
        }

        public static EllipseConsensusModel UnitTest6(out List<PointF> points4)
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
            points4.Add(new PointF(4.0f, -4.0f));
            points4.Add(new PointF(6.33013f, 3.5f));
            points4.Add(new PointF(5.0f, 4.46410f));
            points4.Add(new PointF(2.86603f, 4.5f));
            points4.Add(new PointF(0.5f, 3.59808f));
            points4.Add(new PointF(-1.46410f, 2.0f));
            points4.Add(new PointF(-2.5f, 0.13397f));
            points4.Add(new PointF(-2.33013f, -1.5f));
            points4.Add(new PointF(5.8f, 0.45f));
            points4.Add(new PointF(0.63f, -2.5f));

            return CalculateEllipticalRegressionConsensus(points4);
        }
        #endregion
    }
}
