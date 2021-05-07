using System;
using System.Collections.Generic;
using System.Drawing;

using enmIndependentVariable = Tools.RegressionUtilities.PolynomialModel.enmIndependentVariable;
using Bias = Tools.RegressionUtilities.PolynomialModel.Bias;

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
            internal Bias bias;

            public CubicModel()
            {
                _degree = DegreeOfPolynomial.Cubic;
            }

            #region Internal Properties of CubicModel
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
            #endregion

            #region Public Properties of CubicModel

            // Get the coefficients of   y = b1 + b2 * x + b3 * x^2   -OR-   x = b1 + b2 * y + b3 * y^2
            public CubicCoefficients Coefficients
            {
                get { return coefficients; }
            }
            #endregion

            public override float ModeledY(float xf)
            {
                if (ValidRegressionModel && independentVariable == enmIndependentVariable.X)
                {
                    double x = (double)xf;
                    return (float)(b1 + b2 * x + b3 * x * x + b4 * x * x * x);
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
                    return (float)(b1 + b2 * y + b3 * y * y + b4 * y * y * y);
                }
                else
                {
                    return float.MinValue;
                }
            }
        }

        public struct CubicConsensusModel
        {
            internal List<PointF> inliers;
            internal List<PointF> outliers;

            #region Public Properties of CubicConsensusModel
            public CubicModel model;
            public CubicModel original;

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

            public double b4
            {
                get { return model.b4; }
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

        internal struct CubicSummations
        {
            // Initialize all the summations to zero
            internal int N;
            internal double x;
            internal double y;
            internal double x2;
            internal double x3;
            internal double x4;
            internal double x5;
            internal double x6;
            internal double xy;
            internal double x2y;   // (i.e.  SUM(x^2*y))
            internal double x3y;
        }

        public static CubicConsensusModel CalculateCubicRegressionConsensus(List<PointF> points, enmIndependentVariable independentVariable = enmIndependentVariable.X, float sensitivityInPixels = ERROR_THRESHOLD_ORIGINAL)
        {
            var cubicRegressionConsensus = new CubicConsensusModel();

            if (points == null || points.Count < 4)
            {
                // Exit with error
                return cubicRegressionConsensus;
            }

            // Calculate the initial model.  Set the initial inliers and outliers (empty) lists.
            cubicRegressionConsensus.inliers = points;
            cubicRegressionConsensus.outliers = new List<PointF>();
            cubicRegressionConsensus.model = CalculateCubicRegressionModel(cubicRegressionConsensus.inliers, independentVariable);
            cubicRegressionConsensus.original = cubicRegressionConsensus.model;

            // Keep removing candidate points until the model is lower than some average error threshold
            while (cubicRegressionConsensus.model.AverageRegressionError > sensitivityInPixels && cubicRegressionConsensus.model.ValidRegressionModel)
            {
                int index1, index2, index3;
                var pointsWithoutPoint1 = new List<PointF>();
                var pointsWithoutPoint2 = new List<PointF>();
                var pointsWithoutPoint3 = new List<PointF>();
                var candidatePoint1 = GetPositiveCandidate(cubicRegressionConsensus.inliers, cubicRegressionConsensus.model, out index1, out pointsWithoutPoint1);
                var candidatePoint2 = GetNegativeCandidate(cubicRegressionConsensus.inliers, cubicRegressionConsensus.model, out index2, out pointsWithoutPoint2);
                var candidatePoint3 = GetInfluenceCandidate(cubicRegressionConsensus.inliers, cubicRegressionConsensus.model, out index3, out pointsWithoutPoint3);

                if (candidatePoint1.IsEmpty || candidatePoint2.IsEmpty || candidatePoint3.IsEmpty || index1 < 0 || index2 < 0 || index3 < 0)
                {
                    // Exit with error
                    break;
                }

                var error1 = CalculateResidual(cubicRegressionConsensus.model, candidatePoint1);
                var error2 = CalculateResidual(cubicRegressionConsensus.model, candidatePoint2);
                var error3 = CalculateResidual(cubicRegressionConsensus.model, candidatePoint3);

                var modelWithoutPoint1 = new CubicModel();
                var modelWithoutPoint2 = new CubicModel();
                var modelWithoutPoint3 = new CubicModel();
                var newAverageError1 = RemovePointAndCalculateError(pointsWithoutPoint1, independentVariable, out modelWithoutPoint1);
                var newAverageError2 = RemovePointAndCalculateError(pointsWithoutPoint2, independentVariable, out modelWithoutPoint2);
                var newAverageError3 = RemovePointAndCalculateError(pointsWithoutPoint3, independentVariable, out modelWithoutPoint3);

                if (newAverageError1 < newAverageError2 && newAverageError1 < newAverageError3)
                {
                    cubicRegressionConsensus.inliers = pointsWithoutPoint1;
                    cubicRegressionConsensus.outliers.Add(candidatePoint1);
                    cubicRegressionConsensus.model = modelWithoutPoint1;
                }
                else if (newAverageError2 < newAverageError3)
                {
                    cubicRegressionConsensus.inliers = pointsWithoutPoint2;
                    cubicRegressionConsensus.outliers.Add(candidatePoint2);
                    cubicRegressionConsensus.model = modelWithoutPoint2;
                }
                else
                {
                    cubicRegressionConsensus.inliers = pointsWithoutPoint3;
                    cubicRegressionConsensus.outliers.Add(candidatePoint3);
                    cubicRegressionConsensus.model = modelWithoutPoint3;
                }
            }

            return cubicRegressionConsensus;
        }

        private static float RemovePointAndCalculateError(List<PointF> pointsWithoutCandidate, enmIndependentVariable independentVariable, out CubicModel modelWithoutCandidate)
        {
            modelWithoutCandidate = CalculateCubicRegressionModel(pointsWithoutCandidate, independentVariable);
            return modelWithoutCandidate.AverageRegressionError;
        }

        private static PointF GetPositiveCandidate(List<PointF> points, CubicModel model, out int index, out List<PointF> pointsWithoutCandidate)
        {
            if (points == null || points.Count <= 4)
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

        private static PointF GetNegativeCandidate(List<PointF> points, CubicModel model, out int index, out List<PointF> pointsWithoutCandidate)
        {
            if (points == null || points.Count <= 4)
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

        private static PointF GetInfluenceCandidate(List<PointF> points, CubicModel model, out int index, out List<PointF> pointsWithoutCandidate)
        {
            if (points == null || points.Count <= 4)
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

        public static CubicModel CalculateCubicRegressionModel(List<PointF> points, enmIndependentVariable independentVariable = enmIndependentVariable.X)
        {
            var cubicRegressionModel = new CubicModel();
            cubicRegressionModel.independentVariable = independentVariable;

            // Calculate the bias
            var bias = PolynomialModel.CalculateBias(points);
            if (bias.x == double.MaxValue)
            {
                cubicRegressionModel.ValidRegressionModel = false;
                return cubicRegressionModel;
            }
            cubicRegressionModel.bias = bias;

            // Remove the bias
            var pointsNoBias = PolynomialModel.RemoveBias(points, bias);
            if (pointsNoBias == null || pointsNoBias.Count == 0)
            {
                cubicRegressionModel.ValidRegressionModel = false;
                return cubicRegressionModel;
            }

            // Calculate the summations on the points after the bias has been removed
            var sum = CalculateCubicSummations(cubicRegressionModel.independentVariable, pointsNoBias);
            if (sum.N <= 0)
            {
                cubicRegressionModel.ValidRegressionModel = false;
                return cubicRegressionModel;
            }

            // Calculate the initial regression model
            cubicRegressionModel = CalculateInitialCubicModel(sum, bias, cubicRegressionModel.independentVariable);
            if (!cubicRegressionModel.ValidRegressionModel)
            {
                cubicRegressionModel.ValidRegressionModel = false;
                return cubicRegressionModel;
            }

            // Calculate the line features
            CalculateCubicFeatures(ref cubicRegressionModel);
            if (!cubicRegressionModel.ValidRegressionModel)
            {
                cubicRegressionModel.ValidRegressionModel = false;
                return cubicRegressionModel;
            }

            // Calculate the average residual error
            cubicRegressionModel.AverageRegressionError = cubicRegressionModel.CalculateAverageRegressionError(points);
            if (cubicRegressionModel.AverageRegressionError >= float.MaxValue)
            {
                cubicRegressionModel.ValidRegressionModel = false;
            }

            return cubicRegressionModel;
        }

        // Calculate residual error
        public static float CalculateResidual(CubicModel model, PointF point, out bool pointIsAboveModel)
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

        public static float CalculateResidual(CubicModel model, PointF point)
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

        private static void CalculateCubicFeatures(ref CubicModel model)
        {

        }

        private static CubicSummations CalculateCubicSummations(enmIndependentVariable independentVariable, List<PointF> points)
        {
            var sum = new CubicSummations();
            if (points == null || points.Count < 4)
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

        private static CubicModel CalculateInitialCubicModel(CubicSummations sum, Bias bias, enmIndependentVariable independentVariable)
        {
            var model = new CubicModel();
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
            var t11 = s22*s33-s23*s23;
            var t12 = s13*s23-s12*s33;
            var t13 = s12*s23-s13*s22;
            var t22 = s11*s33-s13*s13;
            var t23 = s12*s13-s11*s23;
            var t33 = s11*s22-s12*s12;
            var determinantS = s11 * (s22 * s33 - s23 * s23) - s12 * (s12 * s33 - s13 * s23) + s13 * (s12 * s23 - s13 * s22);

            // Don't divide by zero
            if (Math.Abs(determinantS) <= EPSILON)
            {
                model.ValidRegressionModel = false;
                return model;
            }

            // Calculate the coefficients of y = b1 + b2*x + b3*x^2 + b4*x^3
            model.b2 = (sY1*t11 + sY2*t12 + sY3*t13) / determinantS;
            model.b3 = (sY1*t12 + sY2*t22 + sY3*t23) / determinantS;
            model.b4 = (sY1*t13 + sY2*t23 + sY3*t33) / determinantS;
            model.b1 = YMean - model.b2*XMean - model.b3*XXMean - model.b4*XXXMean;

            // Adjust for the bias
            if (independentVariable == enmIndependentVariable.X)
            {
                model.b1 = model.b1 - model.b4 * bias.x * bias.x * bias.x + model.b3 * bias.x * bias.x - model.b2 * bias.x + bias.y;
                model.b2 = model.b2 + 3.0f * model.b4 * bias.x * bias.x - 2.0f * model.b3 * bias.x;
                model.b3 = model.b3 - 3.0f * model.b4 * bias.x;
            }
            else
            {
                model.b1 = model.b1 - model.b4 * bias.y * bias.y * bias.y + model.b3 * bias.y * bias.y - model.b2 * bias.y + bias.x;
                model.b2 = model.b2 + 3.0f * model.b4 * bias.y * bias.y - 2.0f * model.b3 * bias.y;
                model.b3 = model.b3 - 3.0f * model.b4 * bias.y;
            }

            model.ValidRegressionModel = true;

            return model;
        }

        public static void UnitTest1(out List<PointF> points, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model = CalculateCubicRegressionConsensus(points);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }

        public static void UnitTest2(out List<PointF> points1a, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model = CalculateCubicRegressionConsensus(points1a);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }

        public static void UnitTest3(out List<PointF> pointsH, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model = CalculateCubicRegressionConsensus(pointsH, enmIndependentVariable.Y);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }

        public static void UnitTest4(out List<PointF> pointsH2, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model = CalculateCubicRegressionConsensus(pointsH2, enmIndependentVariable.Y);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }

        public static void UnitTest5(out List<PointF> pointsH2a, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model = CalculateCubicRegressionConsensus(pointsH2a, enmIndependentVariable.Y);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }

        public static void UnitTest6(out List<PointF> pointsPAa, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model = CalculateCubicRegressionConsensus(pointsPAa, enmIndependentVariable.Y);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }

        public static void UnitTest7(out List<PointF> points1e, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model = CalculateCubicRegressionConsensus(points1e);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }

        public static void UnitTest8(out List<PointF> points1d, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model = CalculateCubicRegressionConsensus(points1d);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }

        public static void UnitTest9(out List<PointF> pointsPAb, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model = CalculateCubicRegressionConsensus(pointsPAb, enmIndependentVariable.Y);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }
        public static void UnitTest10(out List<PointF> pointsPAc, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model = CalculateCubicRegressionConsensus(pointsPAc, enmIndependentVariable.Y);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }

        public static void UnitTest11(out List<PointF> points3a, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model = CalculateCubicRegressionConsensus(points3a);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }

        public static void UnitTest12(out List<PointF> points3b, out PolynomialModel fit, out List<PointF> outliers, out PolynomialModel orig)
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

            var model = CalculateCubicRegressionConsensus(points3b);
            fit = model.model;
            outliers = model.outliers;
            orig = model.original;
        }
    }
}
