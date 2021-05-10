using System;
using System.Collections.Generic;
using System.Drawing;

using enmIndependentVariable = Tools.RegressionUtilities.PolynomialModel.enmIndependentVariable;

namespace Tools.RegressionUtilities
{
    /// <summary>
    /// LinearRegression
    /// Author: Merrill McKee
    /// Description:  The purpose of this class is to find the line that best fits through a set of 2D 
    ///   X, Y points.  It uses Linear Regression or the Least Squares Fit method to do this.  An array
    ///   list of System.Drawing.PointF objects are passed in to the constructor.  These points are used to 
    ///   find the variables b1 and b2 in the equation y = b1 + b2*x.  The slope of this line is b2.  
    ///   The y-intercept is b1.  For more information on the formulas used, see the website 
    ///   http://math.stackexchange.com/questions/267865/equations-for-quadratic-regression.
    ///   Once these variables have been calculated in the constructor then the user of the class can call ModeledX
    ///   and ModeledY to get the X value for any given Y position along the parabola or get the Y value for any given
    ///   X position along the parabola.
    ///   
    ///   Note:  Horizontal lines require an independent x-value.  Vertical lines require an independent y-value.
    ///   
    ///   Notes: The linear case is a simplication of the matrix math for the quadratic case in the website link:
    ///          
    ///          [b2] = [s11]^(-1) * [sY1]
    ///          
    ///          [b2] = (1 / s11)  * [sY1]
    ///          
    ///          (see the implementation of the quadratic and cubic cases for how this extends to higher degrees)
    /// </summary>
    [Serializable]
    public class LinearRegression
    {
        protected const double EPSILON = 0.0001;    // Near-zero value to check for division-by-zero
        const float ERROR_THRESHOLD_ORIGINAL = 0.2f;

        public struct LineCoefficients
        {
            public double b1;                        // Coefficients of   y = b1 + b2 * x   -OR-   x = b1 + b2 * y
            public double b2;

            public override string ToString()
            {
                return b1 + " " + b2;
            }
        }

        public class LineModel : PolynomialModel
        {
            internal LineCoefficients coefficients;
            internal double slope;
            internal double intercept;

            public LineModel(enmIndependentVariable independentVariable)
            {
                _degree = DegreeOfPolynomial.Linear;
                MinimumPoints = 2;
                this.independentVariable = independentVariable;
            }

            public LineModel(LineModel copy) : base(copy)
            {
                coefficients = copy.coefficients;
                slope = copy.slope;
                intercept = copy.intercept;
                b1 = copy.b1;
                b2 = copy.b2;
            }

            public override RegressionModel Clone()
            {
                return new LineModel(this);
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

            // Get the coefficients of   y = b1 + b2 * x   -OR-   x = b1 + b2 * y
            public LineCoefficients Coefficients
            {
                get { return coefficients; }
            }

            // Get tilt angle
            public double Slope
            {
                get
                {
                    if (ValidRegressionModel)
                    {
                        return slope;
                    }
                    else
                    {
                        return float.MinValue;
                    }
                }
            }

            public override float ModeledY(float x)
            {
                if (ValidRegressionModel && independentVariable == enmIndependentVariable.X)
                {
                    return (float)(b1 + b2 * x);
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
                    return (float)(b1 + b2 * y);
                }
                else
                {
                    return float.MinValue;
                }
            }

            protected class LinearSummations : Summations
            {
                public double x2;
                public double xy;
            }

            public override Summations CalculateSummations(List<PointF> points)
            {
                var sum = new LinearSummations();
                if (points == null || points.Count < MinimumPoints)
                {
                    sum.N = 0;
                    return sum;
                }

                // Initialize all the summations to zero
                sum.x = 0.0;
                sum.y = 0.0;
                sum.x2 = 0.0;
                sum.xy = 0.0;

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

                    // Sums
                    sum.x += x;
                    sum.y += y;
                    sum.x2 += xx;
                    sum.xy += xy;
                }

                return sum;
            }

            public override void CalculateModel(Summations sums)
            {
                if (sums.N <= 0)
                {
                    ValidRegressionModel = false;
                    return;
                }

                LinearSummations sum = sums as LinearSummations;

                // Calculate the means
                var XMean = sum.x / sum.N;
                var YMean = sum.y / sum.N;

                // Calculate the S intermediate values
                var s11 = sum.x2 - (1.0 / sum.N) * sum.x * sum.x;
                var sY1 = sum.xy - (1.0 / sum.N) * sum.x * sum.y;

                // Don't divide by zero
                // Note:  Maintaining the matrix notation even though S or s11 is a 1x1 "matrix".  For higher degrees, 
                //        the notation will remain consistent.
                var determinantS = s11;
                if (Math.Abs(determinantS) <= EPSILON)
                {
                    ValidRegressionModel = false;
                }

                // Calculate the coefficients of y = b1 + b2*x
                b2 = sY1 / determinantS;
                b1 = YMean - b2 * XMean;

                // Adjust for the bias
                if (independentVariable == enmIndependentVariable.X)
                {
                    b1 = b1 + bias.y - b2 * bias.x;
                }
                else
                {
                    b1 = b1 + bias.x - b2 * bias.y;
                }

                ValidRegressionModel = true;
            }

            public override void CalculateFeatures()
            {
                slope = b2;
                intercept = b1;
            }
        }

        public class LinearConsensusModel : RegressionConsensusModel
        {
            public LinearConsensusModel(enmIndependentVariable independentVariable)
            {
                inliers = null;
                outliers = null;
                model = new LineModel(independentVariable);
                original = new LineModel(independentVariable);
            }

            // Calculate geometric error (Euclidean distance; point-to-line)
            protected override float CalculateError(RegressionModel model, PointF point, out bool pointOnPositiveSide)
            {
                LineModel line = model as LineModel;

                if (line == null || point == null)
                {
                    pointOnPositiveSide = false;
                    return float.MaxValue;
                }

                if (line.independentVariable == enmIndependentVariable.X)
                {
                    var numerator = -line.b2 * point.X + point.Y - line.b1;

                    pointOnPositiveSide = true;
                    if (numerator < 0.0f)
                    {
                        pointOnPositiveSide = false;
                    }

                    return (float)(Math.Abs(numerator) / Math.Sqrt(line.b2 * line.b2 + 1.0));
                }
                else
                {
                    var numerator = -line.b2 * point.Y + point.X - line.b1;

                    pointOnPositiveSide = true;
                    if (numerator < 0.0f)
                    {
                        pointOnPositiveSide = false;
                    }

                    return (float)(Math.Abs(numerator) / Math.Sqrt(line.b2 * line.b2 + 1.0));
                }
            }
        }

        public static RegressionConsensusModel CalculateLinearRegressionConsensus(List<PointF> points, enmIndependentVariable independentVariable = enmIndependentVariable.X, float sensitivityInPixels = ERROR_THRESHOLD_ORIGINAL)
        {
            var consensus = new LinearConsensusModel(independentVariable);
            consensus.Calculate(points, sensitivityInPixels);

            return consensus;
        }

        public static RegressionConsensusModel UnitTestA1(out List<PointF> anscombe1)
        {
            // Anscombe's quartet - https://en.wikipedia.org/wiki/Anscombe%27s_quartet

            anscombe1 = new List<PointF>();
            anscombe1.Add(new PointF(10.0f, 8.04f));
            anscombe1.Add(new PointF(8.0f, 6.95f));
            anscombe1.Add(new PointF(13.0f, 7.58f));
            anscombe1.Add(new PointF(9.0f, 8.81f));
            anscombe1.Add(new PointF(11.0f, 8.33f));
            anscombe1.Add(new PointF(14.0f, 9.96f));
            anscombe1.Add(new PointF(6.0f, 7.24f));
            anscombe1.Add(new PointF(4.0f, 4.26f));
            anscombe1.Add(new PointF(12.0f, 10.84f));
            anscombe1.Add(new PointF(7.0f, 4.82f));
            anscombe1.Add(new PointF(5.0f, 5.68f));

            return CalculateLinearRegressionConsensus(anscombe1);
        }

        public static RegressionConsensusModel UnitTestA2(out List<PointF> anscombe2)
        {
            // Anscombe's quartet - https://en.wikipedia.org/wiki/Anscombe%27s_quartet

            anscombe2 = new List<PointF>();
            anscombe2.Add(new PointF(10.0f, 9.14f));
            anscombe2.Add(new PointF(8.0f, 8.14f));
            anscombe2.Add(new PointF(13.0f, 8.74f));
            anscombe2.Add(new PointF(9.0f, 8.77f));
            anscombe2.Add(new PointF(11.0f, 9.26f));
            anscombe2.Add(new PointF(14.0f, 8.10f));
            anscombe2.Add(new PointF(6.0f, 6.13f));
            anscombe2.Add(new PointF(4.0f, 3.10f));
            anscombe2.Add(new PointF(12.0f, 9.13f));
            anscombe2.Add(new PointF(7.0f, 7.26f));
            anscombe2.Add(new PointF(5.0f, 4.74f));

            return CalculateLinearRegressionConsensus(anscombe2);
        }

        public static RegressionConsensusModel UnitTestA3(out List<PointF> anscombe3)
        {
            // Anscombe's quartet - https://en.wikipedia.org/wiki/Anscombe%27s_quartet

            anscombe3 = new List<PointF>();
            anscombe3.Add(new PointF(10.0f, 7.46f));
            anscombe3.Add(new PointF(8.0f, 6.77f));
            anscombe3.Add(new PointF(13.0f, 12.74f));
            anscombe3.Add(new PointF(9.0f, 7.11f));
            anscombe3.Add(new PointF(11.0f, 7.81f));
            anscombe3.Add(new PointF(14.0f, 8.84f));
            anscombe3.Add(new PointF(6.0f, 6.08f));
            anscombe3.Add(new PointF(4.0f, 5.39f));
            anscombe3.Add(new PointF(12.0f, 8.15f));
            anscombe3.Add(new PointF(7.0f, 6.42f));
            anscombe3.Add(new PointF(5.0f, 5.73f));

            return CalculateLinearRegressionConsensus(anscombe3);
        }

        public static RegressionConsensusModel UnitTestA4(out List<PointF> anscombe4)
        {
            // Anscombe's quartet - https://en.wikipedia.org/wiki/Anscombe%27s_quartet

            anscombe4 = new List<PointF>();
            anscombe4.Add(new PointF(8.01f, 5.25f));
            anscombe4.Add(new PointF(8.02f, 5.56f));
            anscombe4.Add(new PointF(8.03f, 5.76f));
            anscombe4.Add(new PointF(8.04f, 6.58f));
            anscombe4.Add(new PointF(8.05f, 6.89f));
            anscombe4.Add(new PointF(8.06f, 7.71f));
            anscombe4.Add(new PointF(8.07f, 7.91f));
            anscombe4.Add(new PointF(8.08f, 8.47f));
            anscombe4.Add(new PointF(8.09f, 8.84f));
            anscombe4.Add(new PointF(8.05f, 7.04f));
            anscombe4.Add(new PointF(19.0f, 12.5f));

            return CalculateLinearRegressionConsensus(anscombe4);
        }

        public static RegressionConsensusModel UnitTest1(out List<PointF> points)
        {
            ////////////////////////////////////////
            // Unit test #1:  Line with slope = 2 //
            ////////////////////////////////////////

            // A line y = 2x + 1 has the following points
            // [0 1]
            // [1 3]
            // [2 5]
            // 
            // We should be able to fit to these points and return the coefficients [1 2].

            points = new List<PointF>();
            points.Add(new PointF(-3.0f, -5.0f)); // True line point:  y = 2x + 1
            points.Add(new PointF(-2.0f, -3.0f)); // True line point:  y = 2x + 1
            points.Add(new PointF(-1.5f, -2.0f)); // True line point:  y = 2x + 1
            points.Add(new PointF(-1.0f, -1.0f)); // True line point:  y = 2x + 1
            points.Add(new PointF(-0.5f, 0.0f)); // True line point:  y = 2x + 1
            points.Add(new PointF(0.0f, 1.0f)); // True line point:  y = 2x + 1
            points.Add(new PointF(0.5f, 2.0f)); // True line point:  y = 2x + 1
            points.Add(new PointF(1.0f, 3.0f)); // True line point:  y = 2x + 1
            points.Add(new PointF(2.0f, 5.0f)); // True line point:  y = 2x + 1
            points.Add(new PointF(3.0f, 9.5f)); // <--- Adding in 2.5 noise
            points.Add(new PointF(4.0f, 7.0f)); // <--- Adding in -2.0 noise
            points.Add(new PointF(5.0f, 14.5f)); // <--- Adding in 3.5 noise
            points.Add(new PointF(5.0f, 11.0f)); // True line point:  y = 2x + 1
            points.Add(new PointF(7.0f, 11.0f)); // <--- Adding in 4.0 noise

            return CalculateLinearRegressionConsensus(points);
        }

        public static RegressionConsensusModel UnitTest2(out List<PointF> pointsH)
        {
            //////////////////////////////////
            // Unit test #2:  Vertical line //
            //////////////////////////////////

            // A simple vertical line x = 3 has the following points
            // [3 0]
            // [3 1]
            // [3 2]
            // 
            // We should be able to fit to these points and return the coefficients [3 0].

            pointsH = new List<PointF>();
            pointsH.Add(new PointF(3.0f, 0.0f));
            pointsH.Add(new PointF(3.0f, 1.0f));
            pointsH.Add(new PointF(3.0f, 2.0f));

            return CalculateLinearRegressionConsensus(pointsH, enmIndependentVariable.Y);
        }

        public static RegressionConsensusModel UnitTest3(out List<PointF> points3)
        {
            //////////////////////////////////////////////////
            // Unit test #3 with bias:  Line with slope = 2 //
            //////////////////////////////////////////////////

            // A line y-11000 = 2(x-1500) + 1 ... OR ... y = 2x + 8001 ... has the following points
            // [1500 11001]
            // [1501 11003]
            // [1502 11005]
            // 
            // We should be able to fit to these points and return the coefficients [1 2].

            points3 = new List<PointF>();
            points3.Add(new PointF(1500.0f, 11001.0f));
            points3.Add(new PointF(1501.0f, 11003.0f));
            points3.Add(new PointF(1502.0f, 11005.0f));
            points3.Add(new PointF(1503.0f, 11007.0f));
            points3.Add(new PointF(1504.0f, 11009.0f));
            points3.Add(new PointF(1505.0f, 11011.0f));
            points3.Add(new PointF(1506.0f, 11013.0f));

            return CalculateLinearRegressionConsensus(points3);
        }

        public static RegressionConsensusModel UnitTest4(out List<PointF> points4)
        {
            ////////////////////////////////////////
            // Unit test #4:  Line with slope = 2 //
            ////////////////////////////////////////

            // A line y = 2x + 1 has the following points
            // [0 1]
            // [1 3]
            // [2 5]
            // 
            // We should be able to fit to these points and return the coefficients [1 2].

            // Add some random noise to test the stopping condition
            var noise = new Random(1);

            points4 = new List<PointF>();
            var NOISE_LEVEL = 3.0f; // Inverse relation:  1.0f means +/- 0.5f ... 2.0f means +/- 0.25f ... 100.0f means +/- 0.005f
            var HALF = 1.0f / NOISE_LEVEL / 2.0f;
            points4.Add(new PointF(-3.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, -5.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF)); // True line point:  y = 2x + 1
            points4.Add(new PointF(-2.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, -3.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF)); // True line point:  y = 2x + 1
            points4.Add(new PointF(-1.5f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, -2.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF)); // True line point:  y = 2x + 1
            points4.Add(new PointF(-1.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, -1.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF)); // True line point:  y = 2x + 1
            points4.Add(new PointF(-0.5f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 0.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF)); // True line point:  y = 2x + 1
            points4.Add(new PointF(0.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 1.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF)); // True line point:  y = 2x + 1
            points4.Add(new PointF(0.5f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 2.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF)); // True line point:  y = 2x + 1
            points4.Add(new PointF(1.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 3.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF)); // True line point:  y = 2x + 1
            points4.Add(new PointF(2.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 5.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF)); // True line point:  y = 2x + 1
            points4.Add(new PointF(3.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 9.5f + (float)noise.NextDouble() / NOISE_LEVEL - HALF)); // <--- Adding in 2.5 noise
            points4.Add(new PointF(4.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 7.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF)); // <--- Adding in -2.0 noise
            points4.Add(new PointF(5.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 7.5f + (float)noise.NextDouble() / NOISE_LEVEL - HALF)); // <--- Adding in -3.5 noise
            points4.Add(new PointF(5.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 11.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF)); // True line point:  y = 2x + 1
            points4.Add(new PointF(7.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF, 13.0f + (float)noise.NextDouble() / NOISE_LEVEL - HALF)); // <--- Adding in -2.0 noise

            return CalculateLinearRegressionConsensus(points4);
        }

        public static RegressionConsensusModel UnitTest5(out List<PointF> points4)
        {
            /////////////////////////////////////////////////////////////////////////////
            // Unit test #4:  Line with slope = 2 meets another line (corner scenario) //
            /////////////////////////////////////////////////////////////////////////////

            // A line y = 2x + 1 has the following points
            // [0 1]
            // [1 3]
            // [2 5]
            // 
            // A line y = -x + 16 intersects the first line at (5,11)
            //

            points4 = new List<PointF>();
            points4.Add(new PointF(-3.0f, -5.0f));  // True line point:  y = 2x + 1
            points4.Add(new PointF(-2.0f, -3.0f));  // True line point:  y = 2x + 1
            points4.Add(new PointF(-1.5f, -2.0f));  // True line point:  y = 2x + 1
            points4.Add(new PointF(-1.0f, -1.0f));  // True line point:  y = 2x + 1
            points4.Add(new PointF(-0.5f, 0.0f));   // True line point:  y = 2x + 1
            points4.Add(new PointF(0.0f, 1.0f));    // True line point:  y = 2x + 1
            points4.Add(new PointF(0.5f, 2.0f));    // True line point:  y = 2x + 1
            points4.Add(new PointF(1.0f, 3.0f));    // True line point:  y = 2x + 1
            points4.Add(new PointF(2.0f, 5.0f));    // True line point:  y = 2x + 1
            points4.Add(new PointF(3.0f, 7.0f));    // True line point:  y = 2x + 1
            points4.Add(new PointF(4.0f, 9.0f));    // True line point:  y = 2x + 1
            points4.Add(new PointF(5.0f, 11.0f));   // True line point:  y = 2x + 1

            points4.Add(new PointF(6.0f, 10.0f));   // True line point:  y = -x + 16
            points4.Add(new PointF(7.0f, 9.0f));   // True line point:  y = -x + 16
            points4.Add(new PointF(8.0f, 8.0f));   // True line point:  y = -x + 16
            points4.Add(new PointF(9.0f, 7.0f));   // True line point:  y = -x + 16

            return CalculateLinearRegressionConsensus(points4);
        }
    }
}
