using System;
using System.Collections.Generic;
using System.Drawing;

namespace Tools.RegressionUtilities
{
    /// <summary>
    /// RegressionConsensusModel
    /// Author: Merrill McKee
    /// Description:  A set of inliers and outliers plus 2 regression models:
    ///   the original least squares regression model (all data points)
    ///   the consensus least squares regression model (only inliers)
    /// </summary>
    [Serializable]
    public abstract class RegressionConsensusModel
    {
        protected List<PointF> inliers;
        protected List<PointF> outliers;

        protected RegressionModel model;
        protected RegressionModel original;

        protected float sensitivity;
        protected const float DEFAULT_SENSITIVITY = 0.35f;

        public List<PointF> Inliers
        {
            get
            {
                return inliers;
            }
        }

        public List<PointF> Outliers
        {
            get
            {
                return outliers;
            }
        }

        public RegressionModel Model
        {
            get
            {
                return model;
            }
        }

        public RegressionModel Original
        {
            get
            {
                return original;
            }
        }

        protected abstract float CalculateError(RegressionModel model, PointF point, out bool pointOnPositiveSide);

        protected PointF GetPositiveCandidate(List<PointF> points, RegressionModel model, out int index, out List<PointF> pointsWithoutCandidate)
        {
            if (points == null || points.Count <= model.MinimumPoints)
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
                var error = CalculateError(model, point, out pointOnPositiveSide);

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

        protected PointF GetNegativeCandidate(List<PointF> points, RegressionModel model, out int index, out List<PointF> pointsWithoutCandidate)
        {
            if (points == null || points.Count <= model.MinimumPoints)
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
                var error = CalculateError(model, point, out pointOnPositiveSide);

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

        protected enum InfluenceError
        {
            L1 = 1,
            L2 = 2,
        }
        protected InfluenceError influenceError = InfluenceError.L1;

        protected PointF GetInfluenceCandidate(List<PointF> points, RegressionModel model, out int index, out List<PointF> pointsWithoutCandidate)
        {
            if (points == null || points.Count <= model.MinimumPoints)
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

                var dx = point.X - model.bias.x;
                var dy = point.Y - model.bias.y;

                float influence;
                if (influenceError == InfluenceError.L1)
                {
                    influence = (float)Math.Abs(dx + dy);
                }
                else // L2
                {
                    influence = (float)(dx * dx + dy * dy);
                }

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

        protected float RemovePointAndCalculateError(List<PointF> pointsWithoutCandidate, ref RegressionModel modelWithoutCandidate)
        {
            modelWithoutCandidate.CalculateModel(pointsWithoutCandidate);
            return modelWithoutCandidate.AverageRegressionError;
        }

        // Derived class will use the appropriate least squares regression to initialize the model/original
        // Returns 0 on success, returns non-zero on failure
        public int Calculate(List<PointF> points, float sensitivity = DEFAULT_SENSITIVITY)
        {
            if (points == null || points.Count < model.MinimumPoints)
            {
                // Exit with error
                return 1;
            }

            inliers = points;
            outliers = new List<PointF>();
            model.CalculateModel(points);
            original = model.Clone();

            // Keep removing candidate points until the model is lower than some average error threshold
            while (model.AverageRegressionError > sensitivity && model.ValidRegressionModel)
            {
                int index1, index2, index3;
                List<PointF> pointsWithoutPoint1;
                List<PointF> pointsWithoutPoint2;
                List<PointF> pointsWithoutPoint3;
                var candidatePoint1 = GetPositiveCandidate(inliers, model, out index1, out pointsWithoutPoint1);
                var candidatePoint2 = GetNegativeCandidate(inliers, model, out index2, out pointsWithoutPoint2);
                var candidatePoint3 = GetInfluenceCandidate(inliers, model, out index3, out pointsWithoutPoint3);

                if (candidatePoint1.IsEmpty || candidatePoint2.IsEmpty || candidatePoint3.IsEmpty || index1 < 0 || index2 < 0 || index3 < 0)
                {
                    // Exit with error
                    break;
                }

                RegressionModel modelWithoutPoint1 = model.Clone();
                RegressionModel modelWithoutPoint2 = model.Clone();
                RegressionModel modelWithoutPoint3 = model.Clone();
                var newAverageError1 = RemovePointAndCalculateError(pointsWithoutPoint1, ref modelWithoutPoint1);
                var newAverageError2 = RemovePointAndCalculateError(pointsWithoutPoint2, ref modelWithoutPoint2);
                var newAverageError3 = RemovePointAndCalculateError(pointsWithoutPoint3, ref modelWithoutPoint3);

                if (newAverageError1 < newAverageError2 && newAverageError1 < newAverageError3)
                {
                    inliers = pointsWithoutPoint1;
                    outliers.Add(candidatePoint1);
                    model = modelWithoutPoint1;
                }
                else if (newAverageError2 < newAverageError3)
                {
                    inliers = pointsWithoutPoint2;
                    outliers.Add(candidatePoint2);
                    model = modelWithoutPoint2;
                }
                else
                {
                    inliers = pointsWithoutPoint3;
                    outliers.Add(candidatePoint3);
                    model = modelWithoutPoint3;
                }
            }

            return 0;
        }
    }

    /// <summary>
    /// RegressionModel
    /// Author: Merrill McKee
    /// Description:  This is the abstract parent class for regression models PolynomialModel (abstract) and EllipseModel
    ///   
    /// </summary>
    [Serializable]
    public abstract class RegressionModel
    {
        public int MinimumPoints;

        public bool ValidRegressionModel { get; set; }

        protected const double EPSILON = 0.0001;    // Near-zero value to check for division-by-zero

        protected RegressionModel()
        {
            ValidRegressionModel = false;
        }

        protected RegressionModel(RegressionModel copy)
        {
            ValidRegressionModel = copy.ValidRegressionModel;
            MinimumPoints = copy.MinimumPoints;
            bias = copy.bias;
        }

        public abstract RegressionModel Clone();

        public abstract class Summations
        {
            public int N;
            public double x;    // Sigma(x)
            public double y;    // Sigma(y)
        }

        public abstract Summations CalculateSummations(List<PointF> points);

        public abstract void CalculateModel(Summations sum);

        public abstract void CalculateFeatures();

        public void CalculateModel(List<PointF> points)
        {
            // Calculate the bias
            bias = CalculateBias(points);
            if (bias.x == double.MaxValue)
            {
                ValidRegressionModel = false;
                return;
            }

            // Remove the bias
            var pointsNoBias = RemoveBias(points, bias);
            if (pointsNoBias == null || pointsNoBias.Count == 0)
            {
                ValidRegressionModel = false;
                return;
            }

            // Calculate the summations on the points after the bias has been removed
            var sum = CalculateSummations(pointsNoBias);
            if (sum.N <= 0)
            {
                ValidRegressionModel = false;
                return;
            }

            CalculateModel(sum);
            if (!ValidRegressionModel)
            {
                return;
            }

            CalculateFeatures();
            if (!ValidRegressionModel)
            {
                return;
            }

            CalculateAverageRegressionError(points);
            if (AverageRegressionError >= float.MaxValue)
            {
                ValidRegressionModel = false;
                return;
            }
        }

        public struct Bias
        {
            internal double x;
            internal double y;
        }
        public Bias bias { protected set; get; }

        public static Bias CalculateBias(List<PointF> points)
        {
            Bias bias;
            if (points == null || points.Count < 1)
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

        public static List<PointF> RemoveBias(List<PointF> points, Bias bias)
        {
            if (points == null || points.Count < 1)
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

        // Calculate the average regression error
        public float CalculateAverageRegressionError(List<PointF> points)
        {
            if (points == null || points.Count == 0)
            {
                return float.MaxValue;
            }

            if (!ValidRegressionModel)
            {
                return float.MaxValue;
            }

            var sumRegressionErrors = 0.0f;
            for (int i = 0; i < points.Count; ++i)
            {
                sumRegressionErrors += CalculateRegressionError(points[i]);
            }

            // Save internally
            averageRegressionError = sumRegressionErrors / (float)points.Count;

            // Also return
            return averageRegressionError;
        }

        // Calculate the single-point regression error
        public abstract float CalculateRegressionError(PointF point);

        // Returns the model average regression error
        public float AverageRegressionError
        {
            get
            {
                if (ValidRegressionModel)
                {
                    return averageRegressionError;
                }
                else
                {
                    return float.MaxValue;
                }
            }
            set
            {
                averageRegressionError = value;
            }
        }
        protected float averageRegressionError;     // Average regression error

        // If the bias is known or a good estimate exists, remove it
        protected static List<PointF> ZeroBiasPoints(List<PointF> points, float xBias, float yBias)
        {
            if (points == null || points.Count == 0)
            {
                // Invalid input
                return new List<PointF>();
            }

            var newPoints = new List<PointF>();

            // Remove the bias
            for (var i = 0; i < points.Count; ++i)
            {
                newPoints.Add(new PointF(points[i].X - xBias, points[i].Y - yBias));
            }

            return newPoints;
        }

        // In an attempt to remove unknown bias, zero mean a set of points
        protected static List<PointF> ZeroMeanPoints(List<PointF> points, out float xMean, out float yMean)
        {
            if (points == null || points.Count == 0)
            {
                // Invalid input
                xMean = 0.0f;
                yMean = 0.0f;
                return new List<PointF>();
            }

            var xSum = 0.0f;
            var ySum = 0.0f;
            var newPoints = new List<PointF>();

            // Calculate the summations
            for (var i = 0; i < points.Count; ++i)
            {
                xSum += points[i].X;
                ySum += points[i].Y;
            }
            xMean = xSum / (float)points.Count;
            yMean = ySum / (float)points.Count;

            // Zero the means
            for (var i = 0; i < points.Count; ++i)
            {
                newPoints.Add(new PointF(points[i].X - xMean, points[i].Y - yMean));
            }

            return newPoints;
        }
    }
}

