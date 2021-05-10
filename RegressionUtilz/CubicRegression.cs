using System;
using System.Collections.Generic;
using System.Drawing;

using enmIndependentVariable = Tools.RegressionUtilities.PolynomialModel.enmIndependentVariable;

namespace Tools.RegressionUtilities
{
    /// <summary>
    /// CubicRegression
    /// Author: Merrill McKee
    /// Description:  The purpose of this class is to find the curve that best fits through a set of 2D 
    ///   X, Y points.  It uses Cubic Regression or the Least Squares Fit method to do this.  An array
    ///   list of System.Drawing.PointF objects are passed in to the constructor.  These points are used to 
    ///   find the variables b1, b2, b3, and b4 in the equation y = b1 + b2*x + b3*x^2 + b4*x^3.  For more 
    ///   information on the formulas used, see the website 
    ///   http://math.stackexchange.com/questions/267865/equations-for-quadratic-regression and the notes 
    ///   below.  Also, look at my notes for the CubicRegression.cs implementation and use 
    ///   http://www.dr-lex.be/random/matrix-inv.html for the explicit equations for a 3x3 inverse.
    ///   
    ///   Once these variables have been calculated in the constructor then the user of the class can call ModeledX
    ///   and ModeledY to get the X value for any given Y position along the curve or get the Y value for any given
    ///   X position along the curve.
    ///   
    ///   Notes: Additional matrix algebra details not in the website link:
    ///          
    ///          [b2] = [s11 s12 s13]-1 * [sY1]
    ///          [b3]   [s12 s22 s23]     [sY2]
    ///          [b4]   [s13 s23 s33]     [sY3]
    ///          
    ///          [b2] = (1 / det(S)) [ (s22s33-s23s23)  (s13s23-s12s33)  (s12s23-s13s22) ] * [sY1]
    ///          [b3]                [ (s13s23-s12s33)  (s11s33-s13s13)  (s12s13-s11s23) ]   [sY2]
    ///          [b4]                [ (s12s23-s13s22)  (s12s13-s11s23)  (s11s22-s12s12) ]   [sY3]
    ///          
    ///             where  det(S) = s11(s22s33-s23s23) - s12(s12s33-s13s23) + s13(s12s23-s13s22)
    ///          
    ///             using t11 = (s22s33-s23s23)       OR       [t11 t12 t13]
    ///                   t12 = (s13s23-s12s33)                [t12 t22 t23]
    ///                   t13 = (s12s23-s13s22)                [t13 t23 t33]
    ///                   t22 = (s11s33-s13s13)
    ///                   t23 = (s12s13-s11s23)
    ///                   t33 = (s11s22-s12s12)
    /// 
    ///          [b2] = (1 / det(S)) [t11 t12 t13] * [sY1]
    ///          [b3]                [t12 t22 t23]   [sY2]
    ///          [b4]                [t13 t23 t33]   [sY3]
    ///          
    /// </summary>
    [Serializable]
    public class CubicRegression
    {
        protected const double EPSILON = 0.0001;    // Near-zero value to check for division-by-zero
        const float ERROR_THRESHOLD_ORIGINAL = 0.35f;

        public struct CubicCoefficients
        {
            public double b1;                        // Coefficients of   y = b1 + b2 * x + b3 * x^2 + b4 * x^3  -OR-   x = b1 + b2 * y + b3 * y^2 + b4 * y^3
            public double b2;
            public double b3;
            public double b4;

            public override string ToString()
            {
                return b1 + " " + b2 + " " + b3 + " " + b4;
            }
        }

        public class CubicModel : PolynomialModel
        {
            internal CubicCoefficients coefficients;

            public CubicModel(enmIndependentVariable independentVariable)
            {
                _degree = DegreeOfPolynomial.Cubic;
                MinimumPoints = 4;
                this.independentVariable = independentVariable;
            }

            public CubicModel(CubicModel copy) : base(copy)
            {
                coefficients = copy.coefficients;
                b1 = copy.b1;
                b2 = copy.b2;
                b3 = copy.b3;
                b4 = copy.b4;
            }

            public override RegressionModel Clone()
            {
                return new CubicModel(this);
            }

            internal double b1
            {
                get { return coefficients.b1; }
                set { coefficients.b1 = value; }
            }

            internal double b2
            {
                get { return coefficients.b2; }
                set { coefficients.b2 = value; }
            }

            internal double b3
            {
                get { return coefficients.b3; }
                set { coefficients.b3 = value; }
            }

            internal double b4
            {
                get { return coefficients.b4; }
                set { coefficients.b4 = value; }
            }

            public CubicCoefficients Coefficients
            {
                get { return coefficients; }
            }

            public override float ModeledY(float x)
            {
                if (ValidRegressionModel && independentVariable == enmIndependentVariable.X)
                {
                    return (float)(b1 + b2 * x + b3 * x * x + b4 * x * x * x);
                }
                else
                {
                    return float.MinValue;
                }
            }

            public override float ModeledX(float y)
            {
                if (ValidRegressionModel && independentVariable == enmIndependentVariable.Y)
                {
                    return (float)(b1 + b2 * y + b3 * y * y + b4 * y * y * y);
                }
                else
                {
                    return float.MinValue;
                }
            }

            protected class CubicSummations : Summations
            {
                internal double x2;
                internal double x3;
                internal double x4;
                internal double x5;
                internal double x6;
                internal double xy;
                internal double x2y;   // (i.e.  SUM(x^2*y))
                internal double x3y;
            }

            public override Summations CalculateSummations(List<PointF> points)
            {
                var sum = new CubicSummations();
                if (points == null || points.Count < MinimumPoints)
                {
                    sum.N = 0;
                    return sum;
                }

                // Initialize all the summations to zero
                sum.x = 0.0;
                sum.y = 0.0;
                sum.xy = 0.0;
                sum.x2y = 0.0;  // (i.e.  SUM(x^2*y))
                sum.x3y = 0.0;
                sum.x2 = 0.0;
                sum.x3 = 0.0;
                sum.x4 = 0.0;
                sum.x5 = 0.0;
                sum.x6 = 0.0;

                // Shorthand that better matches the math formulas
                var N = sum.N = points.Count;

                // Calculate the summations
                for (var i = 0; i < N; ++i)
                {
                    // Shorthand
                    var x = points[i].X;
                    var y = points[i].Y;

                    // Meh
                    if (independentVariable == enmIndependentVariable.Y)
                    {
                        // Swap the x and y coordinates to handle a y independent variable
                        x = points[i].Y;
                        y = points[i].X;
                    }

                    var xx = x * x;
                    var xy = x * y;
                    var xxx = xx * x;

                    // Sums
                    sum.x += x;
                    sum.y += y;
                    sum.xy += xy;
                    sum.x2y += xx * y;
                    sum.x3y += xxx * y;
                    sum.x2 += xx;
                    sum.x3 += xxx;
                    sum.x4 += xx * xx;
                    sum.x5 += xxx * xx;
                    sum.x6 += xxx * xxx;
                }

                return sum;
            }

            public override void CalculateModel(Summations sums)
            {
                CubicSummations sum = sums as CubicSummations;

                if (sum.N <= 0)
                {
                    ValidRegressionModel = false;
                    return;
                }

                // Calculate the means
                var XMean = sum.x / (double)sum.N;
                var YMean = sum.y / (double)sum.N;
                var XXMean = sum.x2 / (double)sum.N;
                var XXXMean = sum.x3 / (double)sum.N;

                // Calculate the S intermediate values
                var inv_N = (1.0 / (double)sum.N); // Shorthand
                var s11 = sum.x2 - inv_N * sum.x * sum.x;
                var s12 = sum.x3 - inv_N * sum.x * sum.x2;
                var s13 = sum.x4 - inv_N * sum.x * sum.x3;
                var s22 = sum.x4 - inv_N * sum.x2 * sum.x2;
                var s23 = sum.x5 - inv_N * sum.x2 * sum.x3;
                var s33 = sum.x6 - inv_N * sum.x3 * sum.x3;
                var sY1 = sum.xy - inv_N * sum.x * sum.y;
                var sY2 = sum.x2y - inv_N * sum.x2 * sum.y;
                var sY3 = sum.x3y - inv_N * sum.x3 * sum.y;

                // Calculate the inverse matrix of S (inv(S)) using T notation
                // (see notes above)
                var t11 = s22 * s33 - s23 * s23;
                var t12 = s13 * s23 - s12 * s33;
                var t13 = s12 * s23 - s13 * s22;
                var t22 = s11 * s33 - s13 * s13;
                var t23 = s12 * s13 - s11 * s23;
                var t33 = s11 * s22 - s12 * s12;
                var determinantS = s11 * (s22 * s33 - s23 * s23) - s12 * (s12 * s33 - s13 * s23) + s13 * (s12 * s23 - s13 * s22);

                // Don't divide by zero
                if (Math.Abs(determinantS) <= EPSILON)
                {
                    ValidRegressionModel = false;
                    return;
                }

                // Calculate the coefficients of y = b1 + b2*x + b3*x^2 + b4*x^3
                b2 = (sY1 * t11 + sY2 * t12 + sY3 * t13) / determinantS;
                b3 = (sY1 * t12 + sY2 * t22 + sY3 * t23) / determinantS;
                b4 = (sY1 * t13 + sY2 * t23 + sY3 * t33) / determinantS;
                b1 = YMean - b2 * XMean - b3 * XXMean - b4 * XXXMean;

                // Adjust for the bias
                if (independentVariable == enmIndependentVariable.X)
                {
                    b1 = b1 - b4 * bias.x * bias.x * bias.x + b3 * bias.x * bias.x - b2 * bias.x + bias.y;
                    b2 = b2 + 3.0f * b4 * bias.x * bias.x - 2.0f * b3 * bias.x;
                    b3 = b3 - 3.0f * b4 * bias.x;
                }
                else
                {
                    b1 = b1 - b4 * bias.y * bias.y * bias.y + b3 * bias.y * bias.y - b2 * bias.y + bias.x;
                    b2 = b2 + 3.0f * b4 * bias.y * bias.y - 2.0f * b3 * bias.y;
                    b3 = b3 - 3.0f * b4 * bias.y;
                }

                ValidRegressionModel = true;
            }

            public override void CalculateFeatures()
            {
            }
        }

        public class CubicConsensusModel : RegressionConsensusModel
        {
            public CubicConsensusModel(enmIndependentVariable independentVariable)
            {
                inliers = null;
                outliers = null;
                model = new CubicModel(independentVariable);
                original = new CubicModel(independentVariable);
            }

            protected override float CalculateError(RegressionModel rmodel, PointF point, out bool pointOnPositiveSide)
            {
                CubicModel model = rmodel as CubicModel;

                if (model == null || point == null)
                {
                    pointOnPositiveSide = false;
                    return float.MaxValue;
                }

                float error;
                if (model.independentVariable == enmIndependentVariable.X)
                {
                    error = ModeledY(model, point.X) - point.Y;
                }
                else
                {
                    error = ModeledX(model, point.Y) - point.X;
                }

                pointOnPositiveSide = error >= 0.0f;
                return Math.Abs(error);
            }
        }

        public static CubicConsensusModel CalculateCubicRegressionConsensus(List<PointF> points, enmIndependentVariable independentVariable = enmIndependentVariable.X, float sensitivityInPixels = ERROR_THRESHOLD_ORIGINAL)
        {
            var consensus = new CubicConsensusModel(independentVariable);
            consensus.Calculate(points, sensitivityInPixels);

            return consensus;
        }

        // Returns the modeled y-value of an ellipse
        public static float ModeledY(CubicModel model, float xf)
        {
            if (model != null && model.ValidRegressionModel && model.independentVariable == enmIndependentVariable.X)
            {
                double x = (double)xf;
                return (float)(model.b1 + model.b2 * x + model.b3 * x * x + model.b4 * x * x * x);
            }
            else
            {
                return float.MinValue;
            }
        }

        // Returns the modeled x-value of an ellipse
        public static float ModeledX(CubicModel model, float yf)
        {
            if (model != null && model.ValidRegressionModel && model.independentVariable == enmIndependentVariable.Y)
            {
                double y = (double)yf;
                return (float)(model.b1 + model.b2 * y + model.b3 * y * y + model.b4 * y * y * y);
            }
            else
            {
                return float.MinValue;
            }
        }

        public static CubicConsensusModel UnitTest1(out List<PointF> points)
        {
            ///////////////////
            // Unit test #1: //
            ///////////////////

            // A cubic y = x^3 + 2 has the following points
            // [0 2]
            // [1 3]
            // [2 10]
            // [3 29]
            // 
            // We should be able to fit to these points and return the coefficients [2 0 0 1].

            points = new List<PointF>();
            points.Add(new PointF(0.0f, 2.0f));
            points.Add(new PointF(1.0f, 3.0f));
            points.Add(new PointF(2.0f, 10.0f));
            points.Add(new PointF(3.0f, 29.0f));

            return CalculateCubicRegressionConsensus(points);
        }

        public static CubicConsensusModel UnitTest2(out List<PointF> points1a)
        {
            ////////////////////
            // Unit test #1a: //
            ////////////////////

            // A cubic y-400 = (x-500)^3 + 2 has the following points
            // [499 401]
            // [500 402]
            // [501 403]
            // [502 410]
            // [503 429]
            // 
            // We should be able to fit to these points and return the coefficients [2 0 0 1].

            points1a = new List<PointF>();
            points1a.Add(new PointF(499.0f, 401.0f));
            points1a.Add(new PointF(500.0f, 402.0f));
            points1a.Add(new PointF(501.0f, 403.0f));
            points1a.Add(new PointF(502.0f, 410.0f));
            points1a.Add(new PointF(503.0f, 429.0f));

            return CalculateCubicRegressionConsensus(points1a);
        }

        public static CubicConsensusModel UnitTest3(out List<PointF> pointsH)
        {
            ////////////////////
            // Unit test #2:  //
            ////////////////////

            // A cubic x = y^3 + y has the following points
            // [-2 -1]
            // [0 0]
            // [2 1]
            // [10 2]
            // 
            // We should be able to fit to these points and return the coefficients [0 1 0 1].

            pointsH = new List<PointF>();
            pointsH.Add(new PointF(-2.0f, -1.0f));
            pointsH.Add(new PointF(0.0f, 0.0f));
            pointsH.Add(new PointF(2.0f, 1.0f));
            pointsH.Add(new PointF(10.0f, 2.0f));

            return CalculateCubicRegressionConsensus(pointsH, enmIndependentVariable.Y);
        }

        public static CubicConsensusModel UnitTest4(out List<PointF> pointsH2)
        {
            ////////////////////
            // Unit test #2bias:  //
            ////////////////////

            // A cubic x = y^3 + y has the following points
            // [398 499]
            // [400 500]
            // [402 501]
            // [410 502]
            // 
            // We should be able to fit to these points and return the coefficients [-125000102 750001 -1500 1].

            pointsH2 = new List<PointF>();
            pointsH2.Add(new PointF(398.0f, 499.0f));
            pointsH2.Add(new PointF(400.0f, 500.0f));
            pointsH2.Add(new PointF(402.0f, 501.0f));
            pointsH2.Add(new PointF(410.0f, 502.0f));

            return CalculateCubicRegressionConsensus(pointsH2, enmIndependentVariable.Y);
        }

        public static CubicConsensusModel UnitTest5(out List<PointF> pointsH2a)
        {
            ///////////////////////////////////////////////////
            // Unit test #2a:  Horizontal Parabola with bias //
            ///////////////////////////////////////////////////

            // A simple horizontal parabola x - 400 = (y-500)^2 + (y-500) has the following points
            // [400 499]
            // [400 500]
            // [402 501]
            // [406 502]
            // 
            // We should be able to fit to these points and return the coefficients [249900 -999 1 0].

            pointsH2a = new List<PointF>();
            pointsH2a.Add(new PointF(400.0f, 499.0f));
            pointsH2a.Add(new PointF(400.0f, 500.0f));
            pointsH2a.Add(new PointF(402.0f, 501.0f));
            pointsH2a.Add(new PointF(406.0f, 502.0f));

            return CalculateCubicRegressionConsensus(pointsH2a, enmIndependentVariable.Y);
        }

        public static CubicConsensusModel UnitTest6(out List<PointF> pointsPAa)
        {
            ////////////////////////////////////////////////////////
            // Unit test #3:  Left Bead From Pacific Amore Bottle //
            ////////////////////////////////////////////////////////

            pointsPAa = new List<PointF>();
            pointsPAa.Add(new PointF(433.00f, 593f));
            pointsPAa.Add(new PointF(432.00f, 594f));
            pointsPAa.Add(new PointF(431.50f, 595f));
            pointsPAa.Add(new PointF(430.70f, 596f));
            pointsPAa.Add(new PointF(430.56f, 597f));
            pointsPAa.Add(new PointF(430.55f, 598f));
            pointsPAa.Add(new PointF(430.70f, 599f));
            pointsPAa.Add(new PointF(431.50f, 600f));
            pointsPAa.Add(new PointF(432.40f, 601f));
            pointsPAa.Add(new PointF(434.01f, 602f));
            pointsPAa.Add(new PointF(436.01f, 603f));

            return CalculateCubicRegressionConsensus(pointsPAa, enmIndependentVariable.Y);
        }

        public static CubicConsensusModel UnitTest7(out List<PointF> points1e)
        {
            /////////////////////////////////////////////////
            // Unit test #1d:  Vertical Parabola with bias //
            /////////////////////////////////////////////////

            // A simple vertical parabola y - 400 = (x-500)^2 + 2 has the following points
            // [500 402]
            // [501 403]
            // [502 406]
            // [503 411]
            // 
            // We should be able to fit to these points and return the coefficients [250402 -1000 1 0].

            points1e = new List<PointF>();
            var noise = new Random(1);
            var NOISE_LEVEL = 2.0f; // Inverse relation:  1.0f means +/- 0.5f ... 2.0f means +/- 0.25f ... 100.0f means +/- 0.005f
            var HALF = 1.0f / NOISE_LEVEL / 2.0f;
            points1e.Add(new PointF(496.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 418.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF));
            points1e.Add(new PointF(497.0f - 1.2f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 411.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF));
            points1e.Add(new PointF(498.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 406.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF));
            points1e.Add(new PointF(499.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 403.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF));
            points1e.Add(new PointF(500.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 402.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF));
            points1e.Add(new PointF(501.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 403.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF));
            points1e.Add(new PointF(502.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 406.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF));
            points1e.Add(new PointF(503.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 411.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF));

            return CalculateCubicRegressionConsensus(points1e);
        }

        public static CubicConsensusModel UnitTest8(out List<PointF> points1d)
        {
            points1d = new List<PointF>();
            var noise = new Random(1);
            var NOISE_LEVEL = 2.0f; // Inverse relation:  1.0f means +/- 0.5f ... 2.0f means +/- 0.25f ... 100.0f means +/- 0.005f
            var HALF = 1.0f / NOISE_LEVEL / 2.0f;
            points1d.Add(new PointF(496.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 418.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF));
            points1d.Add(new PointF(497.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 411.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF));
            points1d.Add(new PointF(498.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 406.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF));
            points1d.Add(new PointF(499.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 403.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF));
            points1d.Add(new PointF(500.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 402.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF));
            points1d.Add(new PointF(501.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 403.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF));
            points1d.Add(new PointF(502.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 406.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF));
            points1d.Add(new PointF(503.0f + 3.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 411.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF));

            return CalculateCubicRegressionConsensus(points1d);
        }

        public static CubicConsensusModel UnitTest9(out List<PointF> pointsPAb)
        {
            ////////////////////////////////////////////////////////
            // Unit test #3:  Left Bead From Pacific Amore Bottle //
            ////////////////////////////////////////////////////////

            pointsPAb = new List<PointF>();
            pointsPAb.Add(new PointF(433.00f, 593f));
            pointsPAb.Add(new PointF(432.00f, 594f));
            pointsPAb.Add(new PointF(431.50f, 595f));
            pointsPAb.Add(new PointF(430.70f, 596f));
            pointsPAb.Add(new PointF(430.56f, 597f));
            pointsPAb.Add(new PointF(430.55f, 598f));
            pointsPAb.Add(new PointF(430.70f, 599f));
            pointsPAb.Add(new PointF(431.50f, 600f));
            pointsPAb.Add(new PointF(432.40f, 601f));
            pointsPAb.Add(new PointF(434.01f, 602f));
            pointsPAb.Add(new PointF(436.01f, 603f));

            pointsPAb.Add(new PointF(437.01f, 604f));
            pointsPAb.Add(new PointF(437.01f, 605f));
            pointsPAb.Add(new PointF(437.01f, 606f));
            pointsPAb.Add(new PointF(437.01f, 607f));
            pointsPAb.Add(new PointF(437.01f, 608f));

            return CalculateCubicRegressionConsensus(pointsPAb, enmIndependentVariable.Y);
        }

        public static CubicConsensusModel UnitTest10(out List<PointF> pointsPAc)
        {
            ////////////////////////////////////////////////////////
            // Unit test #3:  Left Bead From Pacific Amore Bottle //
            ////////////////////////////////////////////////////////

            pointsPAc = new List<PointF>();
            pointsPAc.Add(new PointF(433.00f, 593f));
            pointsPAc.Add(new PointF(432.00f, 594f));
            pointsPAc.Add(new PointF(431.50f, 595f));
            pointsPAc.Add(new PointF(430.70f, 596f));
            pointsPAc.Add(new PointF(430.56f, 597f));
            pointsPAc.Add(new PointF(430.55f, 598f));
            pointsPAc.Add(new PointF(430.70f, 599f));
            pointsPAc.Add(new PointF(431.50f, 600f));
            pointsPAc.Add(new PointF(432.40f, 601f));
            pointsPAc.Add(new PointF(434.01f, 602f));
            pointsPAc.Add(new PointF(436.01f, 603f));

            pointsPAc.Add(new PointF(437.01f, 604f));
            pointsPAc.Add(new PointF(437.01f, 605f));
            pointsPAc.Add(new PointF(437.01f, 606f));
            pointsPAc.Add(new PointF(437.01f, 607f));
            pointsPAc.Add(new PointF(437.01f, 608f));

            pointsPAc.Add(new PointF(432.01f, 593f));
            pointsPAc.Add(new PointF(431.01f, 593f));
            pointsPAc.Add(new PointF(430.01f, 593f));

            return CalculateCubicRegressionConsensus(pointsPAc, enmIndependentVariable.Y);
        }

        public static CubicConsensusModel UnitTest11(out List<PointF> points3a)
        {
            ////////////////////
            // Unit test #3a: //
            ////////////////////

            // A cubic y-400 = (x-500)^3 + 2 has the following points
            // [499 401]
            // [500 402]
            // [501 403]
            // [502 410]
            // [503 429]
            // 
            // We should be able to fit to these points and return the coefficients [2 0 0 1].

            points3a = new List<PointF>();
            points3a.Add(new PointF(499.0f, 401.0f));
            points3a.Add(new PointF(500.0f, 402.0f));
            points3a.Add(new PointF(501.0f, 403.0f));
            points3a.Add(new PointF(502.0f, 410.0f));
            points3a.Add(new PointF(503.0f, 429.0f));

            return CalculateCubicRegressionConsensus(points3a);
        }

        public static CubicConsensusModel UnitTest12(out List<PointF> points3b)
        {
            ////////////////////
            // Unit test #3b: //
            ////////////////////

            points3b = new List<PointF>();
            points3b.Add(new PointF(399.7f, 515.0f));
            points3b.Add(new PointF(400.4f, 514.0f));
            points3b.Add(new PointF(401.3f, 513.0f));
            points3b.Add(new PointF(402.4f, 512.0f));
            points3b.Add(new PointF(405.9f, 511.0f));
            points3b.Add(new PointF(412.0f, 510.0f));
            points3b.Add(new PointF(418.1f, 509.0f));
            points3b.Add(new PointF(419.1f, 508.0f));
            points3b.Add(new PointF(420.6f, 507.0f));
            points3b.Add(new PointF(420.5f, 506.0f));
            points3b.Add(new PointF(414.1f, 505.0f)); // lint
            points3b.Add(new PointF(413.9f, 504.0f)); // lint
            points3b.Add(new PointF(421.4f, 503.0f));

            return CalculateCubicRegressionConsensus(points3b);
        }
    }
}
