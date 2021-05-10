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
}
