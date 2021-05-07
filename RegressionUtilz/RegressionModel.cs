using System;
using System.Collections.Generic;
using System.Drawing;

namespace Tools.RegressionUtilities
{
    /// <summary>
    /// RegressionModel
    /// Author: Merrill McKee
    /// Description:  This is the abstract parent class for regression models PolynomialModel (abstract) and EllipseModel
    ///   
    /// </summary>
    [Serializable]
    public abstract class RegressionModel
    {
        protected bool _validRegressionModel;       // Remains false until a model is successfully created
        public bool ValidRegressionModel
        {
            get { return _validRegressionModel; }
            set { _validRegressionModel = value; }
        }

        protected float _averageResidual;           // The average of the absolute difference between a model and its input points
        protected const double EPSILON = 0.0001;    // Near-zero value to check for division-by-zero

        public struct Bias
        {
            internal double x;
            internal double y;
        }

        public static Bias CalculateBias(List<PointF> points)
        {
            Bias bias;
            if (points == null || points.Count < 2)
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
            if (points == null || points.Count < 2)
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
                if (_validRegressionModel)
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

