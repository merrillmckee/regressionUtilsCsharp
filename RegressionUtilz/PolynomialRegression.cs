using System;
using System.Drawing;

namespace Tools.RegressionUtilities
{
    /// <summary>
    /// PolynomialModel
    /// Author: Merrill McKee
    /// Description:  This is the abstract parent class for linear, quadratic, cubic, and 
    ///   any other polynomial regression algorithms. (todo: combine into single polynomial regression)
    ///   
    /// </summary>
    [Serializable]
    public abstract class PolynomialModel : RegressionModel
    {
        public enum DegreeOfPolynomial
        {
            Linear    = 1,
            Quadratic = 2,
            Cubic     = 3
        };
        protected DegreeOfPolynomial _degree;

        public enum enmIndependentVariable          // Which variable is independent?
        {                                           //   Linear, x-independent:     Can model horizontal lines
            X = 0,                                  //   Linear, y-independent:     Can model vertical lines
            Y                                       //   Quadratic, x-independent:  Vertical parabola
        };                                          //   Quadratic, y-independent:  Horizontal parabola
        public enmIndependentVariable independentVariable;

        protected PolynomialModel()
        {
            independentVariable = enmIndependentVariable.X;
            _degree = DegreeOfPolynomial.Linear;
        }

        protected PolynomialModel(PolynomialModel copy) : base(copy)
        {
            independentVariable = copy.independentVariable;
            _degree = copy._degree;
        }

        // Returns the modeled y-value of a regression model with independent x-variable
        public abstract float ModeledY(float x);

        // Returns the modeled x-value of a regression model with independent y-variable
        public abstract float ModeledX(float y);

        // Calculate the single-point regression error
        public override float CalculateRegressionError(PointF point)
        {
            if (independentVariable == enmIndependentVariable.X)
            {
                return Math.Abs(ModeledY(point.X) - point.Y);
            }
            else
            {
                return Math.Abs(ModeledX(point.Y) - point.X);
            }
        }

        // Return the degree of the regression model
        public uint Degree()
        {
            return (uint)_degree;
        }
    }
}
