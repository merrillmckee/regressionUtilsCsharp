using System;
using System.Collections.Generic;
using System.Drawing;

using enmIndependentVariable = Tools.RegressionUtilities.PolynomialModel.enmIndependentVariable;
using Bias = Tools.RegressionUtilities.PolynomialModel.Bias;

// Requires PolynomialRegression.cs

namespace Tools.RegressionUtilities
{
    /// <summary>
    /// QuadraticRegression
    /// Author: Merrill McKee
    /// Description:  The purpose of this class is to find the parabola that best fits through a set of 2D 
    ///   X, Y points.  It uses Quadratic Regression or the Least Squares Fit method to do this.  An array
    ///   list of System.Drawing.PointF objects are passed in to the constructor.  These points are used to 
    ///   find the variables b1, b2, and b3 in the equation y = b1 + b2*x + b3*x^2.  The vertex of this 
    ///   parabola is (x0, y0).  For more information on the formulas used, see the website 
    ///   http://math.stackexchange.com/questions/267865/equations-for-quadratic-regression.
    ///   Once these variables have been calculated in the constructor then the user of the class can call ModeledX
    ///   and ModeledY to get the X value for any given Y position along the parabola or get the Y value for any given
    ///   X position along the parabola.
    ///   
    ///   Note:  Vertically oriented parabolas assume an independent x-value.  Horizontally oriented parabolas 
    ///          assume an independent y-value.  Internally, a horizontally oriented parabola swaps the 
    ///          x and y values but the user's interface should not be affected.
    ///   
    ///   Notes: Additional matrix algebra details not in the website link:
    ///          
    ///          [b2] = [s11 s12]-1 * [sY1]
    ///          [b3]   [s12 s22]     [sY2]
    ///          
    ///          [b2] = (1 / det(S)) [ s22 -s12] * [sY1]
    ///          [b3]                [-s12  s11]   [sY2]
    ///          
    ///          This derivation will allow us to use this same algorithm for cubic regression.  The inverse of a 3x3 
    ///          matrix is a bit more tedious
    /// </summary>
    [Serializable]
    public class QuadraticRegression
    {
        protected const double EPSILON = 0.0001;    // Near-zero value to check for division-by-zero
        const float ERROR_THRESHOLD_ORIGINAL = 0.35f;

        public struct QuadraticCoefficients
        {
            public double b1;                        // Coefficients of   y = b1 + b2 * x + b3 * x^2   -OR-   x = b1 + b2 * y + b3 * y^2
            public double b2;
            public double b3;

            public override string ToString()
            {
                return b1 + " " + b2 + " " + b3;
            }
        }

        public class QuadraticModel : PolynomialModel
        {
            internal QuadraticCoefficients coefficients;
            internal Bias bias;

            public QuadraticModel()
            {
                _degree = DegreeOfPolynomial.Quadratic;
            }

            #region Internal Properties of QuadraticModel
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
            #endregion

            #region Public Properties of QuadraticModel

            // Get the coefficients of   y = b1 + b2 * x + b3 * x^2   -OR-   x = b1 + b2 * y + b3 * y^2
            public QuadraticCoefficients Coefficients
            {
                get { return coefficients; }
            }
            #endregion

            public override float ModeledY(float xf)
            {
                if (ValidRegressionModel && independentVariable == enmIndependentVariable.X)
                {
                    double x = (double)xf;
                    return (float)(b1 + b2 * x + b3 * x * x);
                }
                else
                {
                    return float.MinValue;
                }
            }

            public override float ModeledX(float yf)
            {
                if (ValidRegressionModel && independentVariable == enmIndependentVariable.Y)
                {
                    double y = (double)yf;
                    return (float)(b1 + b2 * y + b3 * y * y);
                }
                else
                {
                    return float.MinValue;
                }
            }
        }

        public struct QuadraticConsensusModel
        {
            internal List<PointF> inliers;
            internal List<PointF> outliers;

            #region Public Properties of QuadraticConsensusModel
            public QuadraticModel model;
            public QuadraticModel original;

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

            public double b3
            {
                get { return model.b3; }
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

        internal struct QuadraticSummations
        {
            // Initialize all the summations to zero
            internal int N;
            internal double x;
            internal double y;
            internal double x2;
            internal double xy;
            internal double x3;
            internal double x2y;   // (i.e.  SUM(x^2*y))
            internal double x4;
        }

        public static QuadraticConsensusModel CalculateQuadraticRegressionConsensus(List<PointF> points, enmIndependentVariable independentVariable = enmIndependentVariable.X, float sensitivityInPixels = ERROR_THRESHOLD_ORIGINAL)
        {
            var quadraticRegressionConsensus = new QuadraticConsensusModel();

            if (points == null || points.Count < 3)
            {
                // Exit with error
                return quadraticRegressionConsensus;
            }

            // Calculate the initial model.  Set the initial inliers and outliers (empty) lists.
            quadraticRegressionConsensus.inliers = points;
            quadraticRegressionConsensus.outliers = new List<PointF>();
            quadraticRegressionConsensus.model = CalculateQuadraticRegressionModel(quadraticRegressionConsensus.inliers, independentVariable);
            quadraticRegressionConsensus.original = quadraticRegressionConsensus.model;

            // Keep removing candidate points until the model is lower than some average error threshold
            while (quadraticRegressionConsensus.model.AverageRegressionError > sensitivityInPixels && quadraticRegressionConsensus.model.ValidRegressionModel)
            {
                int index1, index2, index3;
                var pointsWithoutPoint1 = new List<PointF>();
                var pointsWithoutPoint2 = new List<PointF>();
                var pointsWithoutPoint3 = new List<PointF>();
                var candidatePoint1 = GetPositiveCandidate(quadraticRegressionConsensus.inliers, quadraticRegressionConsensus.model, out index1, out pointsWithoutPoint1);
                var candidatePoint2 = GetNegativeCandidate(quadraticRegressionConsensus.inliers, quadraticRegressionConsensus.model, out index2, out pointsWithoutPoint2);
                var candidatePoint3 = GetInfluenceCandidate(quadraticRegressionConsensus.inliers, quadraticRegressionConsensus.model, out index3, out pointsWithoutPoint3);

                if (candidatePoint1.IsEmpty || candidatePoint2.IsEmpty || candidatePoint3.IsEmpty || index1 < 0 || index2 < 0 || index3 < 0)
                {
                    // Exit with error
                    break;
                }

                //var error1 = CalculateResidual(quadraticRegressionConsensus.model, candidatePoint1);
                //var error2 = CalculateResidual(quadraticRegressionConsensus.model, candidatePoint2);
                //var error3 = CalculateResidual(quadraticRegressionConsensus.model, candidatePoint3);

                var modelWithoutPoint1 = new QuadraticModel();
                var modelWithoutPoint2 = new QuadraticModel();
                var modelWithoutPoint3 = new QuadraticModel();
                var newAverageError1 = RemovePointAndCalculateError(pointsWithoutPoint1, independentVariable, out modelWithoutPoint1);
                var newAverageError2 = RemovePointAndCalculateError(pointsWithoutPoint2, independentVariable, out modelWithoutPoint2);
                var newAverageError3 = RemovePointAndCalculateError(pointsWithoutPoint3, independentVariable, out modelWithoutPoint3);

                if (newAverageError1 < newAverageError2 && newAverageError1 < newAverageError3)
                {
                    quadraticRegressionConsensus.inliers = pointsWithoutPoint1;
                    quadraticRegressionConsensus.outliers.Add(candidatePoint1);
                    quadraticRegressionConsensus.model = modelWithoutPoint1;
                }
                else if (newAverageError2 < newAverageError3)
                {
                    quadraticRegressionConsensus.inliers = pointsWithoutPoint2;
                    quadraticRegressionConsensus.outliers.Add(candidatePoint2);
                    quadraticRegressionConsensus.model = modelWithoutPoint2;
                }
                else
                {
                    quadraticRegressionConsensus.inliers = pointsWithoutPoint3;
                    quadraticRegressionConsensus.outliers.Add(candidatePoint3);
                    quadraticRegressionConsensus.model = modelWithoutPoint3;
                }
            }

            return quadraticRegressionConsensus;
        }

        private static float RemovePointAndCalculateError(List<PointF> pointsWithoutCandidate, enmIndependentVariable independentVariable, out QuadraticModel modelWithoutCandidate)
        {
            modelWithoutCandidate = CalculateQuadraticRegressionModel(pointsWithoutCandidate, independentVariable);
            return modelWithoutCandidate.AverageRegressionError;
        }

        private static PointF GetPositiveCandidate(List<PointF> points, QuadraticModel model, out int index, out List<PointF> pointsWithoutCandidate)
        {
            if (points == null || points.Count <= 3)
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
                var error = CalculateResidual(model, point, out pointOnPositiveSide);

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

        private static PointF GetNegativeCandidate(List<PointF> points, QuadraticModel model, out int index, out List<PointF> pointsWithoutCandidate)
        {
            if (points == null || points.Count <= 3)
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
                var error = CalculateResidual(model, point, out pointOnPositiveSide);

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

        private static PointF GetInfluenceCandidate(List<PointF> points, QuadraticModel model, out int index, out List<PointF> pointsWithoutCandidate)
        {
            if (points == null || points.Count <= 3)
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

        public static QuadraticModel CalculateQuadraticRegressionModel(List<PointF> points, enmIndependentVariable independentVariable = enmIndependentVariable.X)
        {
            var quadraticRegressionModel = new QuadraticModel();
            quadraticRegressionModel.independentVariable = independentVariable;

            // Calculate the bias
            var bias = PolynomialModel.CalculateBias(points);
            if (bias.x == double.MaxValue)
            {
                quadraticRegressionModel.ValidRegressionModel = false;
                return quadraticRegressionModel;
            }
            quadraticRegressionModel.bias = bias;

            // Remove the bias
            var pointsNoBias = PolynomialModel.RemoveBias(points, bias);
            if (pointsNoBias == null || pointsNoBias.Count == 0)
            {
                quadraticRegressionModel.ValidRegressionModel = false;
                return quadraticRegressionModel;
            }

            // Calculate the summations on the points after the bias has been removed
            var sum = CalculateQuadraticSummations(quadraticRegressionModel.independentVariable, pointsNoBias);
            if (sum.N <= 0)
            {
                quadraticRegressionModel.ValidRegressionModel = false;
                return quadraticRegressionModel;
            }

            // Calculate the initial regression model
            quadraticRegressionModel = CalculateInitialQuadraticModel(sum, bias, quadraticRegressionModel.independentVariable);
            if (!quadraticRegressionModel.ValidRegressionModel)
            {
                quadraticRegressionModel.ValidRegressionModel = false;
                return quadraticRegressionModel;
            }

            // Calculate the line features
            CalculateQuadraticFeatures(ref quadraticRegressionModel);
            if (!quadraticRegressionModel.ValidRegressionModel)
            {
                quadraticRegressionModel.ValidRegressionModel = false;
                return quadraticRegressionModel;
            }

            // Calculate the average residual error
            quadraticRegressionModel.AverageRegressionError = quadraticRegressionModel.CalculateAverageRegressionError(points);
            if (quadraticRegressionModel.AverageRegressionError >= float.MaxValue)
            {
                quadraticRegressionModel.ValidRegressionModel = false;
            }

            return quadraticRegressionModel;
        }

        // Calculate residual error
        public static float CalculateResidual(QuadraticModel model, PointF point, out bool pointIsAboveModel)
        {
            if (model == null || point == null)
            {
                pointIsAboveModel = false;
                return float.MaxValue;
            }

            var error = 0.0f;
            if (model.independentVariable == enmIndependentVariable.X)
            {
                error = ModeledY(model, point.X) - point.Y;
            }
            else
            {
                error = ModeledX(model, point.Y) - point.X;
            }

            pointIsAboveModel = error >= 0.0f;
            return Math.Abs(error);
        }

        public static float CalculateResidual(QuadraticModel model, PointF point)
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
        public static float ModeledY(QuadraticModel model, float xf)
        {
            if (model != null && model.ValidRegressionModel && model.independentVariable == enmIndependentVariable.X)
            {
                double x = (double)xf;
                return (float)(model.b1 + model.b2 * x + model.b3 * x * x);
            }
            else
            {
                return float.MinValue;
            }
        }

        // Returns the modeled x-value of an ellipse
        public static float ModeledX(QuadraticModel model, float yf)
        {
            if (model != null && model.ValidRegressionModel && model.independentVariable == enmIndependentVariable.Y)
            {
                double y = (double)yf;
                return (float)(model.b1 + model.b2 * y + model.b3 * y * y);
            }
            else
            {
                return float.MinValue;
            }
        }

        private static void CalculateQuadraticFeatures(ref QuadraticModel model)
        {
            // Don't divide by zero when calculating the vertex of the parabola
            // If there is a division-by-zero when calculating the vertex of the 
            // parabola, it means the model is near-linear.  The coefficient of the 
            // squared term is near-zero.
            //if (Math.Abs(b3) <= EPSILON)
            //{
            //    x0 = float.MinValue;
            //    y0 = float.MinValue;
            //}
            //else
            //{
            //    // Calculate the vertex of the parabola
            //    x0 = (float)(b2 / (-2.0 * b3));
            //    y0 = (float)(b1 - (b2 * b2) / (4.0 * b3));
            //}
        }

        private static QuadraticSummations CalculateQuadraticSummations(enmIndependentVariable independentVariable, List<PointF> points)
        {
            var sum = new QuadraticSummations();
            if (points == null || points.Count < 3)
            {
                sum.N = 0;
                return sum;
            }

            // Initialize all the summations to zero
            sum.x = 0.0;
            sum.y = 0.0;
            sum.x2 = 0.0;
            sum.xy = 0.0;
            sum.x3 = 0.0;
            sum.x2y = 0.0;  // (i.e.  SUM(x^2*y))
            sum.x4 = 0.0;

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
                sum.x3 += x * xx;
                sum.x2y += xx * y;
                sum.x4 += xx * xx;
            }

            return sum;
        }

        private static QuadraticModel CalculateInitialQuadraticModel(QuadraticSummations sum, Bias bias, enmIndependentVariable independentVariable)
        {
            var model = new QuadraticModel();
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
            var XXMean = sum.x2 / (float)sum.N;

            // Calculate the S intermediate values
            var s11 = sum.x2 - (1.0 / (double)sum.N) * sum.x * sum.x;
            var s12 = sum.x3 - (1.0 / (double)sum.N) * sum.x * sum.x2;
            var s22 = sum.x4 - (1.0 / (double)sum.N) * sum.x2 * sum.x2;
            var sY1 = sum.xy - (1.0 / (double)sum.N) * sum.x * sum.y;
            var sY2 = sum.x2y - (1.0 / (double)sum.N) * sum.x2 * sum.y;

            // Don't divide by zero
            var determinantS = s22 * s11 - s12 * s12;
            if (Math.Abs(determinantS) <= EPSILON)
            {
                model.ValidRegressionModel = false;
                return model;
            }

            // Calculate the coefficients of y = b1 + b2*x + b3*x^2
            model.b2 = (sY1 * s22 - sY2 * s12) / determinantS;
            model.b3 = (sY2 * s11 - sY1 * s12) / determinantS;
            model.b1 = YMean - model.b2 * XMean - model.b3 * XXMean;

            // Adjust for the bias
            if (independentVariable == enmIndependentVariable.X)
            {
                model.b1 = model.b1 + bias.y + model.b3 * bias.x * bias.x - model.b2 * bias.x;
                model.b2 = model.b2 - 2.0f * model.b3 * bias.x;
            }
            else
            {
                model.b1 = model.b1 + bias.x + model.b3 * bias.y * bias.y - model.b2 * bias.y;
                model.b2 = model.b2 - 2.0f * model.b3 * bias.y;
            }

            model.ValidRegressionModel = true;

            return model;
        }

        public static void UnitTestA2(out List<PointF> anscombe1, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
        {
            // Anscombe's quartet - https://en.wikipedia.org/wiki/Anscombe%27s_quartet

            anscombe1 = new List<PointF>();
            anscombe1.Add(new PointF(10.0f, 9.14f));
            anscombe1.Add(new PointF(8.0f, 8.14f));
            anscombe1.Add(new PointF(13.0f, 8.74f));
            anscombe1.Add(new PointF(9.0f, 8.77f));
            anscombe1.Add(new PointF(11.0f, 9.26f));
            anscombe1.Add(new PointF(14.0f, 8.10f));
            anscombe1.Add(new PointF(6.0f, 6.13f));
            anscombe1.Add(new PointF(4.0f, 3.10f));
            anscombe1.Add(new PointF(12.0f, 9.13f));
            anscombe1.Add(new PointF(7.0f, 7.26f));
            anscombe1.Add(new PointF(5.0f, 4.74f));

            var model_anscombe1a = CalculateQuadraticRegressionConsensus(anscombe1);
            fit = model_anscombe1a.model;
            outliers = model_anscombe1a.outliers;
            orig = model_anscombe1a.original;
        }

        public static void UnitTest1(out List<PointF> points, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
        {
            //////////////////////////////////////
            // Unit test #1:  Vertical Parabola //
            //////////////////////////////////////

            // A simple vertical parabola y = x^2 + 2 has the following points
            // [0 2]
            // [1 3]
            // [2 6]
            // [3 11]
            // 
            // We should be able to fit to these points and return the coefficients [2 0 1].

            points = new List<PointF>();
            points.Add(new PointF(0.0f, 2.0f));
            points.Add(new PointF(1.0f, 3.0f));
            points.Add(new PointF(2.0f, 6.0f));
            points.Add(new PointF(3.0f, 11.0f));

            var model = CalculateQuadraticRegressionConsensus(points);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }

        public static void UnitTest2(out List<PointF> points1b, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
        {
            /////////////////////////////////////////////////
            // Unit test #1a:  Vertical Parabola with bias //
            /////////////////////////////////////////////////

            // A simple vertical parabola y - 400 = (x-500)^2 + 2 has the following points
            // [500 402]
            // [501 403]
            // [502 406]
            // [503 411]
            // 
            // We should be able to fit to these points and return the coefficients [250402 -1000 1].

            points1b = new List<PointF>();
            points1b.Add(new PointF(500.0f, 402.0f));
            points1b.Add(new PointF(501.0f, 403.0f));
            points1b.Add(new PointF(502.0f, 406.0f));
            points1b.Add(new PointF(503.0f, 411.0f));

            var model1c = CalculateQuadraticRegressionConsensus(points1b);
            fit = model1c.model;
            outliers = model1c.outliers;
            orig = model1c.original;
        }

        public static void UnitTest3(out List<PointF> pointsH, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
        {
            ////////////////////////////////////////
            // Unit test #2:  Horizontal Parabola //
            ////////////////////////////////////////

            // A simple horizontal parabola x = y^2 + y has the following points
            // [0 -1]
            // [0 0]
            // [2 1]
            // 
            // We should be able to fit to these points and return the coefficients [0 1 1].

            pointsH = new List<PointF>();
            pointsH.Add(new PointF(0.0f, -1.0f));
            pointsH.Add(new PointF(0.0f, 0.0f));
            pointsH.Add(new PointF(2.0f, 1.0f));

            var modelHb = CalculateQuadraticRegressionConsensus(pointsH, enmIndependentVariable.Y);
            fit = modelHb.model;
            outliers = modelHb.outliers;
            orig = modelHb.original;
        }

        public static void UnitTest4(out List<PointF> pointsH2a, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
        {
            ///////////////////////////////////////////////////
            // Unit test #2a:  Horizontal Parabola with bias //
            ///////////////////////////////////////////////////

            // A simple horizontal parabola x - 400 = (y-500)^2 + (y-500) has the following points
            // [400 499]
            // [400 500]
            // [402 501]
            // 
            // We should be able to fit to these points and return the coefficients [249900 -999 1].

            pointsH2a = new List<PointF>();
            pointsH2a.Add(new PointF(400.0f, 499.0f));
            pointsH2a.Add(new PointF(400.0f, 500.0f));
            pointsH2a.Add(new PointF(402.0f, 501.0f));

            var modelH2b = CalculateQuadraticRegressionConsensus(pointsH2a, enmIndependentVariable.Y);
            fit = modelH2b.model;
            outliers = modelH2b.outliers;
            orig = modelH2b.original;
        }

        public static void UnitTest5(out List<PointF> points1e, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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
            // We should be able to fit to these points and return the coefficients [250402 -1000 1].

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

            var model1e = CalculateQuadraticRegressionConsensus(points1e);
            fit = model1e.model;
            outliers = model1e.outliers;
            orig = model1e.original;
        }

        public static void UnitTest6(out List<PointF> points1d, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model1d = CalculateQuadraticRegressionConsensus(points1d);
            fit = model1d.model;
            outliers = model1d.outliers;
            orig = model1d.original;
        }

        public static void UnitTest7(out List<PointF> pointsPA, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
        {
            ////////////////////////////////////////////////////////
            // Unit test #3:  Left Bead From Pacific Amore Bottle //
            ////////////////////////////////////////////////////////

            pointsPA = new List<PointF>();
            pointsPA.Add(new PointF(433.00f, 593f));
            pointsPA.Add(new PointF(432.00f, 594f));
            pointsPA.Add(new PointF(431.50f, 595f));
            pointsPA.Add(new PointF(430.70f, 596f));
            pointsPA.Add(new PointF(430.56f, 597f));
            pointsPA.Add(new PointF(430.55f, 598f));
            pointsPA.Add(new PointF(430.70f, 599f));
            pointsPA.Add(new PointF(431.50f, 600f));
            pointsPA.Add(new PointF(432.40f, 601f));
            pointsPA.Add(new PointF(434.01f, 602f));
            pointsPA.Add(new PointF(436.01f, 603f));

            var modelPAb = CalculateQuadraticRegressionConsensus(pointsPA, enmIndependentVariable.Y);

            fit = modelPAb.model;
            outliers = modelPAb.outliers;
            orig = modelPAb.original;
        }

        public static void UnitTest8(out List<PointF> pointsPAb, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
        {
            ////////////////////////////////////////////////////////
            // Unit test #3:  Left Bead From Pacific Amore Bottle //
            ////////////////////////////////////////////////////////

            // 
            // We should be able to fit to these points and return the coefficients [0 1 1].

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

            var modelPA = CalculateQuadraticRegressionConsensus(pointsPAb, enmIndependentVariable.Y);
            fit = modelPA.model;
            outliers = modelPA.outliers;
            orig = modelPA.original;
        }

        public static void UnitTest9(out List<PointF> pointsPAc, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
        {
            ////////////////////////////////////////////////////////
            // Unit test #3:  Left Bead From Pacific Amore Bottle //
            ////////////////////////////////////////////////////////

            // 
            // We should be able to fit to these points and return the coefficients [0 1 1].

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

            var modelPAc = CalculateQuadraticRegressionConsensus(pointsPAc, enmIndependentVariable.Y);
            fit = modelPAc.model;
            outliers = modelPAc.outliers;
            orig = modelPAc.original;
        }
    }
}
