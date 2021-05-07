using System;
using System.Collections.Generic;
using System.Drawing;

using enmIndependentVariable = Tools.RegressionUtilities.PolynomialModel.enmIndependentVariable;
using Bias = Tools.RegressionUtilities.PolynomialModel.Bias;

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
            internal Bias bias;

            public LineModel()
            {
                _degree = DegreeOfPolynomial.Linear;
            }

            #region Internal Properties of LineModel
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
            #endregion

            #region Public Properties of LineModel

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
            #endregion

            public override float ModeledY(float x)
            {
                if (ValidRegressionModel && independentVariable == enmIndependentVariable.X)
                {
                    return (float)(b1 + b2 * (double)x);
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
                    return (float)(b1 + b2 * (double)y);
                }
                else
                {
                    return float.MinValue;
                }
            }
        }

        public struct LinearConsensusModel
        {
            internal List<PointF> inliers;
            internal List<PointF> outliers;

            #region Public Properties of LinearConsensusModel
            public LineModel model;
            public LineModel original;

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

            public double b1
            {
                get { return model.b1; }
            }

            public double b2
            {
                get { return model.b2; }
            }

            // AverageRegressionError
            public float AverageRegressionError
            {
                get
                {
                    if (model.ValidRegressionModel)
                    {
                        if (model.AverageRegressionError <= 0.0f || model.AverageRegressionError > 99999.9f)
                        {
                            model.AverageRegressionError = model.CalculateAverageRegressionError(inliers);
                        }

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

        internal struct LinearSummations
        {
            // Initialize all the summations to zero
            internal int N;
            internal double x;
            internal double y;
            internal double x2;
            internal double xy;
        }

        public static LinearConsensusModel CalculateLinearRegressionConsensus(List<PointF> points, enmIndependentVariable independentVariable = enmIndependentVariable.X, float sensitivityInPixels = ERROR_THRESHOLD_ORIGINAL)
        {
            var linearRegressionConsensus = new LinearConsensusModel();

            if (points == null || points.Count < 2)
            {
                // Exit with error
                return linearRegressionConsensus;
            }

            // Calculate the initial model.  Set the initial inliers and outliers (empty) lists.
            linearRegressionConsensus.inliers = points;
            linearRegressionConsensus.outliers = new List<PointF>();
            linearRegressionConsensus.model = CalculateLinearRegressionModel(linearRegressionConsensus.inliers, independentVariable);
            linearRegressionConsensus.original = linearRegressionConsensus.model;

            // Keep removing candidate points until the model is lower than some average error threshold
            while (linearRegressionConsensus.model.AverageRegressionError > sensitivityInPixels && linearRegressionConsensus.model.ValidRegressionModel)
            {
                int index1, index2, index3;
                var pointsWithoutPoint1 = new List<PointF>();
                var pointsWithoutPoint2 = new List<PointF>();
                var pointsWithoutPoint3 = new List<PointF>();
                var candidatePoint1 = GetPositiveCandidate(linearRegressionConsensus.inliers, linearRegressionConsensus.model, out index1, out pointsWithoutPoint1);
                var candidatePoint2 = GetNegativeCandidate(linearRegressionConsensus.inliers, linearRegressionConsensus.model, out index2, out pointsWithoutPoint2);
                var candidatePoint3 = GetInfluenceCandidate(linearRegressionConsensus.inliers, linearRegressionConsensus.model, out index3, out pointsWithoutPoint3);

                if (candidatePoint1.IsEmpty || candidatePoint2.IsEmpty || candidatePoint3.IsEmpty || index1 < 0 || index2 < 0 || index3 < 0)
                {
                    // Exit with error
                    break;
                }

                //var error1 = CalculateResidual(linearRegressionConsensus.model, candidatePoint1);
                //var error2 = CalculateResidual(linearRegressionConsensus.model, candidatePoint2);
                //var error3 = CalculateResidual(linearRegressionConsensus.model, candidatePoint3);

                var modelWithoutPoint1 = new LineModel();
                var modelWithoutPoint2 = new LineModel();
                var modelWithoutPoint3 = new LineModel();
                var newAverageError1 = RemovePointAndCalculateError(pointsWithoutPoint1, independentVariable, out modelWithoutPoint1);
                var newAverageError2 = RemovePointAndCalculateError(pointsWithoutPoint2, independentVariable, out modelWithoutPoint2);
                var newAverageError3 = RemovePointAndCalculateError(pointsWithoutPoint3, independentVariable, out modelWithoutPoint3);

                if (newAverageError1 < newAverageError2 && newAverageError1 < newAverageError3)
                {
                    linearRegressionConsensus.inliers = pointsWithoutPoint1;
                    linearRegressionConsensus.outliers.Add(candidatePoint1);
                    linearRegressionConsensus.model = modelWithoutPoint1;
                }
                else if (newAverageError2 < newAverageError3)
                {
                    linearRegressionConsensus.inliers = pointsWithoutPoint2;
                    linearRegressionConsensus.outliers.Add(candidatePoint2);
                    linearRegressionConsensus.model = modelWithoutPoint2;
                }
                else
                {
                    linearRegressionConsensus.inliers = pointsWithoutPoint3;
                    linearRegressionConsensus.outliers.Add(candidatePoint3);
                    linearRegressionConsensus.model = modelWithoutPoint3;
                }
            }

            return linearRegressionConsensus;
        }

        private static PointF GetPositiveCandidate(List<PointF> points, LineModel model, out int index, out List<PointF> pointsWithoutCandidate)
        {
            if (points == null || points.Count <= 2)
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
                bool pointOnPositiveSide;
                var error = CalculateGeometricError(model, point, out pointOnPositiveSide);

                if (pointOnPositiveSide)
                {
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

        private static PointF GetNegativeCandidate(List<PointF> points, LineModel model, out int index, out List<PointF> pointsWithoutCandidate)
        {
            if (points == null || points.Count <= 2)
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
                bool pointOnPositiveSide;
                var error = CalculateGeometricError(model, point, out pointOnPositiveSide);

                if (!pointOnPositiveSide)
                {
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

        private static PointF GetInfluenceCandidate(List<PointF> points, LineModel model, out int index, out List<PointF> pointsWithoutCandidate)
        {
            if (points == null || points.Count <= 2)
            {
                index = -1;
                pointsWithoutCandidate = new List<PointF>();
                return new PointF();
            }

            var maximumInfluence = 0.0f;
            index = 0;
            for (var i = 0; i < points.Count; ++i)
            {
                var point = points[i];
                var influence = (float)(Math.Abs(point.X - model.bias.x + point.Y - model.bias.y));
                if (influence > maximumInfluence)
                {
                    maximumInfluence = influence;
                    index = i;
                }
            }

            pointsWithoutCandidate = new List<PointF>(points);
            pointsWithoutCandidate.RemoveAt(index);

            return points[index];
        }

        // Calculate geometric error (Euclidean distance; point-to-line)
        protected static float CalculateGeometricError(LineModel line, PointF point, out bool pointOnPositiveSide)
        {
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

        public static LineModel CalculateLinearRegressionModel(List<PointF> points, enmIndependentVariable independentVariable = enmIndependentVariable.X)
        {
            var linearRegressionModel = new LineModel();
            linearRegressionModel.independentVariable = independentVariable;

            // Calculate the bias
            var bias = PolynomialModel.CalculateBias(points);
            if (bias.x == double.MaxValue)
            {
                linearRegressionModel.ValidRegressionModel = false;
                return linearRegressionModel;
            }
            linearRegressionModel.bias = bias;

            // Remove the bias
            var pointsNoBias = PolynomialModel.RemoveBias(points, bias);
            if (pointsNoBias == null || pointsNoBias.Count == 0)
            {
                linearRegressionModel.ValidRegressionModel = false;
                return linearRegressionModel;
            }

            // Calculate the summations on the points after the bias has been removed
            var sum = CalculateLinearSummations(linearRegressionModel.independentVariable, pointsNoBias);
            if (sum.N <= 0)
            {
                linearRegressionModel.ValidRegressionModel = false;
                return linearRegressionModel;
            }

            // Calculate the initial regression model
            linearRegressionModel = CalculateInitialLinearModel(sum, bias, linearRegressionModel.independentVariable);
            if (!linearRegressionModel.ValidRegressionModel)
            {
                linearRegressionModel.ValidRegressionModel = false;
                return linearRegressionModel;
            }

            // Calculate the line features
            CalculateLineFeatures(ref linearRegressionModel);
            if (!linearRegressionModel.ValidRegressionModel)
            {
                linearRegressionModel.ValidRegressionModel = false;
                return linearRegressionModel;
            }

            // Calculate the average residual error
            linearRegressionModel.AverageRegressionError = linearRegressionModel.CalculateAverageRegressionError(points);
            if (linearRegressionModel.AverageRegressionError >= float.MaxValue)
            {
                linearRegressionModel.ValidRegressionModel = false;
            }

            return linearRegressionModel;
        }

        private static float RemovePointAndCalculateError(List<PointF> pointsWithoutCandidate, enmIndependentVariable independentVariable, out LineModel modelWithoutCandidate)
        {
            modelWithoutCandidate = CalculateLinearRegressionModel(pointsWithoutCandidate, independentVariable);
            return modelWithoutCandidate.AverageRegressionError;
        }

        public static float CalculateResidual(LineModel model, PointF point)
        {
            if (model == null || point == null)
            {
                return float.MaxValue;
            }

            if (model.independentVariable == enmIndependentVariable.X)
            {
                return Math.Abs(ModeledY(model, point.X) - point.Y);
            }
            else
            {
                return Math.Abs(ModeledX(model, point.Y) - point.X);
            }
        }

        // Returns the modeled y-value of an ellipse
        public static float ModeledY(LineModel model, float x)
        {
            if (model != null && model.ValidRegressionModel && model.independentVariable == enmIndependentVariable.X)
            {
                return (float)(model.b1 + model.b2 * (double)x);
            }
            else
            {
                return float.MinValue;
            }
        }

        // Returns the modeled x-value of an ellipse
        public static float ModeledX(LineModel model, float y)
        {
            if (model != null && model.ValidRegressionModel && model.independentVariable == enmIndependentVariable.Y)
            {
                return (float)(model.b1 + model.b2 * (double)y);
            }
            else
            {
                return float.MinValue;
            }
        }

        private static void CalculateLineFeatures(ref LineModel model)
        {
            model.slope = model.b2;
            model.intercept = model.b1;
        }

        private static LinearSummations CalculateLinearSummations(enmIndependentVariable independentVariable, List<PointF> points)
        {
            var sum = new LinearSummations();
            if (points == null || points.Count < 2)
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

        private static LineModel CalculateInitialLinearModel(LinearSummations sum, Bias bias, enmIndependentVariable independentVariable)
        {
            var model = new LineModel();
            if (sum.N <= 0)
            {
                model.ValidRegressionModel = false;
                return model;
            }

            model.bias = bias;
            model.independentVariable = independentVariable;

            // Calculate the means
            var XMean = sum.x / (double)sum.N;
            var YMean = sum.y / (double)sum.N;

            // Calculate the S intermediate values
            var s11 = sum.x2 - (1.0 / (double)sum.N) * sum.x * sum.x;
            var sY1 = sum.xy - (1.0 / (double)sum.N) * sum.x * sum.y;

            // Don't divide by zero
            // Note:  Maintaining the matrix notation even though S or s11 is a 1x1 "matrix".  For higher degrees, 
            //        the notation will remain consistent.
            var determinantS = s11;
            if (Math.Abs(determinantS) <= EPSILON)
            {
                model.ValidRegressionModel = false;
                return model;
            }

            // Calculate the coefficients of y = b1 + b2*x
            model.b2 = sY1 / determinantS;
            model.b1 = YMean - model.b2 * XMean;

            // Adjust for the bias
            if (model.independentVariable == enmIndependentVariable.X)
            {
                model.b1 = model.b1 + bias.y - model.b2 * bias.x;
            }
            else
            {
                model.b1 = model.b1 + bias.x - model.b2 * bias.y;
            }

            model.ValidRegressionModel = true;

            return model;
        }

        public static void UnitTestA1(out List<PointF> anscombe1, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model_anscombe1a = CalculateLinearRegressionConsensus(anscombe1);
            fit = model_anscombe1a.model;
            outliers = model_anscombe1a.outliers;
            orig = model_anscombe1a.original;
        }

        public static void UnitTestA2(out List<PointF> anscombe2, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model_anscombe1a = CalculateLinearRegressionConsensus(anscombe2);
            fit = model_anscombe1a.model;
            outliers = model_anscombe1a.outliers;
            orig = model_anscombe1a.original;
        }

        public static void UnitTestA3(out List<PointF> anscombe3, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model_anscombe3 = CalculateLinearRegressionConsensus(anscombe3);
            fit = model_anscombe3.model;
            outliers = model_anscombe3.outliers;
            orig = model_anscombe3.original;
        }

        public static void UnitTestA4(out List<PointF> anscombe4, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model_anscombe4 = CalculateLinearRegressionConsensus(anscombe4);
            fit = model_anscombe4.model;
            outliers = model_anscombe4.outliers;
            orig = model_anscombe4.original;
        }

        public static void UnitTest1(out List<PointF> points, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var consensus = CalculateLinearRegressionConsensus(points);
            fit = consensus.model;
            outliers = consensus.outliers;
            orig = consensus.original;
        }

        public static void UnitTest2(out List<PointF> pointsH, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var modelH = CalculateLinearRegressionConsensus(pointsH, enmIndependentVariable.Y);
            fit = modelH.model;
            outliers = modelH.outliers;
            orig = modelH.original;
        }

        public static void UnitTest3(out List<PointF> points3, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model3 = CalculateLinearRegressionConsensus(points3);
            fit = model3.model;
            outliers = model3.outliers;
            orig = model3.original;
        }

        public static void UnitTest4(out List<PointF> points4, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model4 = CalculateLinearRegressionConsensus(points4);
            fit = model4.model;
            outliers = model4.outliers;
            orig = model4.original;
        }

        public static void UnitTest5(out List<PointF> points4, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model4 = CalculateLinearRegressionConsensus(points4);
            fit = model4.model;
            outliers = model4.outliers;
            orig = model4.original;
        }
    }
}
