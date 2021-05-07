using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

using Tools.RegressionUtilities;

namespace RegressionUtilz
{
    public partial class DisplayRegressions : Form
    {
        // Display variables
        double minXDisp, maxXDisp, minYDisp, maxYDisp;

        public DisplayRegressions()
        {
            InitializeComponent();

            DisplayUnitTests();
        }

        protected static void GetDataBounds(List<PointF> points, out double minX, out double maxX, out double minY, out double maxY, List<PointF> outliers = null)
        {
            minX = double.MaxValue;
            maxX = double.MinValue;
            minY = double.MaxValue;
            maxY = double.MinValue;
            foreach (var point in points)
            {
                if (outliers != null && outliers.Contains(point))
                {
                    continue;
                }

                if (point.X < minX)
                {
                    minX = point.X;
                }
                if (point.X > maxX)
                {
                    maxX = point.X;
                }
                if (point.Y < minY)
                {
                    minY = point.Y;
                }
                if (point.Y > maxY)
                {
                    maxY = point.Y;
                }
            }
        }

        protected static void GetChartBounds(List<PointF> points, out double minXDisp, out double maxXDisp, out double minYDisp, out double maxYDisp, bool anscombes = false)
        {
            double minX, maxX, minY, maxY;
            GetDataBounds(points, out minX, out maxX, out minY, out maxY);

            double displayBuffer = 2.0;
            minXDisp = minX - displayBuffer;
            maxXDisp = maxX + displayBuffer;
            minYDisp = minY - displayBuffer;
            maxYDisp = maxY + displayBuffer;

            if (anscombes)
            {
                // Special case
                minXDisp = 4;
                minYDisp = 2;
                maxXDisp = 20;
                maxYDisp = 14;
            }
        }

        private double Clamp(double val, double min, double max)
        {
            if (val < min)
            {
                return min;
            }
            else if (val > max)
            {
                return max;
            }

            return val;
        }

        private void DrawLine(Series series, PolynomialModel model, List<PointF> points, List<PointF> outliers = null)
        {
            if (model.Degree() != 1)
            {
                // Not a line
                return;
            }

            double minX, maxX, minY, maxY;
            GetDataBounds(points, out minX, out maxX, out minY, out maxY, outliers);

            if (model.independentVariable == PolynomialModel.enmIndependentVariable.X)
            {
                series.Points.AddXY(minXDisp, model.ModeledY((float)minXDisp));
                series.Points.AddXY(maxXDisp, model.ModeledY((float)maxXDisp));
            }
            else
            {
                series.Points.AddXY(model.ModeledX((float)minYDisp), minYDisp);
                series.Points.AddXY(model.ModeledX((float)maxYDisp), maxYDisp);
            }
        }

        private void DrawPoly(Series series, PolynomialModel model, List<PointF> points, List<PointF> outliers = null)
        {
            if (model.Degree() == 1)
            {
                DrawLine(series, model, points, outliers);
            }

            double minX, maxX, minY, maxY;
            GetDataBounds(points, out minX, out maxX, out minY, out maxY, outliers);

            double numPoints = 30.0;
            if (model.independentVariable == PolynomialModel.enmIndependentVariable.X)
            {
                for (double x = minX; x <= maxX + 0.0001; x += (maxX - minX) / (numPoints - 1.0))
                {
                    series.Points.AddXY(x, Clamp(model.ModeledY((float)x), minYDisp, maxYDisp));
                }
            }
            else
            {
                for (double y = minY; y <= maxY + 0.0001; y += (maxY - minY) / (numPoints - 1.0))
                {
                    series.Points.AddXY(Clamp(model.ModeledX((float)y), minXDisp, maxXDisp), y);
                }
            }
        }

        private void DrawEllipse(Series series, EllipticalRegression.EllipseModel model, List<PointF> points, List<PointF> outliers = null)
        {
            double minX, maxX, minY, maxY;
            GetDataBounds(points, out minX, out maxX, out minY, out maxY, outliers);

            double numPoints = 1000.0;
            double ellipseMinX = model.X0 - model.LongAxis / 2.0;
            double ellipseMaxX = model.X0 + model.LongAxis / 2.0;

            for (double x = ellipseMinX; x <= ellipseMaxX + 0.0001; x += (ellipseMaxX - ellipseMinX) / (numPoints - 1.0))
            {
                var modeledY = EllipticalRegression.ModeledY(model, (float)x, EllipticalRegression.EllipseHalves.TopHalf);
                if (modeledY > minYDisp && modeledY < maxYDisp)
                {
                    series.Points.AddXY(x, modeledY);
                }
            }
            for (double x = ellipseMaxX; x >= ellipseMinX - 0.0001; x -= (ellipseMaxX - ellipseMinX) / (numPoints - 1.0))
            {
                var modeledY = EllipticalRegression.ModeledY(model, (float)x, EllipticalRegression.EllipseHalves.BottomHalf);
                if (modeledY > minYDisp && modeledY < maxYDisp)
                {
                    series.Points.AddXY(x, modeledY);
                }
            }
        }

        private void DrawModel(Series series, RegressionModel model, List<PointF> points, List<PointF> outliers = null)
        {
            if (model is EllipticalRegression.EllipseModel)
            {
                DrawEllipse(series, model as EllipticalRegression.EllipseModel, points, outliers);
            }
            else if (model is PolynomialModel)
            {
                DrawPoly(series, model as PolynomialModel, points, outliers);
            }
        }

        public async Task DisplayRegression(string title = "Title", List<PointF> points = null, RegressionModel consensus = null, List<PointF> outliers = null, RegressionModel orig = null)
        {
            if (points == null || points.Count == 0)
            {
                return;
            }

            chart.Series.Clear();
            chart.Titles.Clear();
            chart.Titles.Add(title);

            Series dataSeries = chart.Series.Add("data points");
            dataSeries.ChartType = SeriesChartType.Point;
            foreach (var point in points)
            {
                dataSeries.Points.AddXY(point.X, point.Y);
            }
            
            GetChartBounds(points, out minXDisp, out maxXDisp, out minYDisp, out maxYDisp, title.Contains("nscombe"));
            chart.ChartAreas[0].AxisX.Minimum = minXDisp;// Math.Min(minX, minY);
            chart.ChartAreas[0].AxisX.Maximum = maxXDisp;// Math.Max(maxX, maxY);
            chart.ChartAreas[0].AxisY.Minimum = minYDisp;// Math.Min(minX, minY);
            chart.ChartAreas[0].AxisY.Maximum = maxYDisp;// Math.Max(maxX, maxY);

            if (orig != null)
            {
                Series origRegression = chart.Series.Add("Original Regression");
                origRegression.ChartType = SeriesChartType.Line;
                origRegression.Color = Color.OrangeRed;

                DrawModel(origRegression, orig, points);
            }

            if (consensus != null)
            {
                Series consensusRegression = chart.Series.Add("Consensus Regression");
                consensusRegression.ChartType = SeriesChartType.Line;
                consensusRegression.Color = Color.Blue;

                DrawModel(consensusRegression, consensus, points, outliers);
            }

            if (outliers != null)
            {
                Series outliersSeries = chart.Series.Add("detected outliers");
                outliersSeries.ChartType = SeriesChartType.Point;
                outliersSeries.MarkerStyle = MarkerStyle.Circle;
                outliersSeries.MarkerBorderColor = Color.Red;
                outliersSeries.MarkerColor = Color.Transparent;
                outliersSeries.MarkerSize = 10;
                foreach (var point in outliers)
                {
                    outliersSeries.Points.AddXY(point.X, point.Y);
                }
            }

            await asyncButtonClick();
        }

        private bool nextButtonClicked = false;
        private async Task asyncButtonClick()
        {
            Stopwatch sw = new Stopwatch();

            while(true)
            {
                if (nextButtonClicked)
                {
                    break;
                }

                await Task.Delay(500);
            }

            nextButtonClicked = false;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            nextButtonClicked = true;
        }

        private async void DisplayUnitTests()
        {
            List<PointF> points = null;
            List<PointF> outliers = null;

            PolynomialModel fit = null;
            PolynomialModel orig = null;
            EllipticalRegression.EllipseModel fitEllipse = null;
            EllipticalRegression.EllipseModel origEllipse = null;

            LinearRegression.UnitTest4(out points, out fit, out outliers, out orig);
            await DisplayRegression("Consensus Regression splits data points into inliers and outliers\nSimilar to RANSAC except it starts with all points and removes outliers 1 by 1", points, fit, outliers, orig);
            LinearRegression.UnitTest5(out points, out fit, out outliers, out orig);
            await DisplayRegression("Linear Regression 3 - corner", points, fit, outliers, orig);

            LinearRegression.UnitTestA1(out points, out fit, out outliers, out orig);
            await DisplayRegression("Linear Regression A1 - Anscombe's Quartet\n1st (all 4 Anscombe's datasets have same original linear regression)", points, fit, outliers, orig);
            LinearRegression.UnitTestA3(out points, out fit, out outliers, out orig);
            await DisplayRegression("Linear Regression A3 - Anscombe's Quartet\n3rd (all 4 Anscombe's datasets have same original linear regression)", points, fit, outliers, orig);
            LinearRegression.UnitTestA4(out points, out fit, out outliers, out orig);
            await DisplayRegression("Linear Regression A4 - Anscombe's Quartet\n4th (all 4 Anscombe's datasets have same linear regression)", points, fit, outliers, orig);
            LinearRegression.UnitTestA2(out points, out fit, out outliers, out orig);
            await DisplayRegression("Linear Regression A2 - Anscombe's Quartet\n 2nd (all 4 Anscombe's datasets have same linear regression)", points, null, null, orig);
            QuadraticRegression.UnitTestA2(out points, out fit, out outliers, out orig);
            await DisplayRegression("Quadratic Regression A2 - Anscombe's Quartet\n2nd (all 4 Anscombe's datasets have same linear regression)", points, fit, outliers, orig);

            QuadraticRegression.UnitTest1(out points, out fit, out outliers, out orig);
            await DisplayRegression("Quadratic Test 1 - Vertical Parabola, no outliers", points, fit, outliers, orig);
            QuadraticRegression.UnitTest3(out points, out fit, out outliers, out orig);
            await DisplayRegression("Quadratic Test 2 - Horizontal Parabola, no outliers", points, fit, outliers, orig);
            QuadraticRegression.UnitTest5(out points, out fit, out outliers, out orig);
            await DisplayRegression("Quadratic Test 3 - One outlier", points, fit, outliers, orig);
            QuadraticRegression.UnitTest6(out points, out fit, out outliers, out orig);
            await DisplayRegression("Quadratic Test 4 - One outlier", points, fit, outliers, orig);
            QuadraticRegression.UnitTest7(out points, out fit, out outliers, out orig);
            await DisplayRegression("Quadratic Test 5 - Real data", points, fit, outliers, orig);
            QuadraticRegression.UnitTest8(out points, out fit, out outliers, out orig);
            await DisplayRegression("Quadratic Test 6 - Real data, plus syn outliers", points, fit, outliers, orig);
            QuadraticRegression.UnitTest9(out points, out fit, out outliers, out orig);
            await DisplayRegression("Quadratic Test 7 - Real data, plus syn outliers", points, fit, outliers, orig);

            CubicRegression.UnitTest2(out points, out fit, out outliers, out orig);
            await DisplayRegression("Cubic Test 1 - No outliers", points, fit, outliers, orig);
            CubicRegression.UnitTest3(out points, out fit, out outliers, out orig);
            await DisplayRegression("Cubic Test 2 - No outliers, y-independent", points, fit, outliers, orig);
            CubicRegression.UnitTest12(out points, out fit, out outliers, out orig);
            await DisplayRegression("Cubic Test 3 - real data", points, fit, outliers, orig);
            CubicRegression.UnitTest8(out points, out fit, out outliers, out orig);
            await DisplayRegression("Cubic Test 4 - cubic to quadratic data", points, fit, outliers, orig);


            EllipticalRegression.UnitTest1(out points, out fitEllipse, out outliers, out origEllipse);
            await DisplayRegression("Elliptical Test 1 - No outliers", points, fitEllipse, outliers, origEllipse);
            EllipticalRegression.UnitTest3(out points, out fitEllipse, out outliers, out origEllipse);
            await DisplayRegression("Elliptical Test 2 - 30 degree rotation", points, fitEllipse, outliers, origEllipse);
            EllipticalRegression.UnitTest4(out points, out fitEllipse, out outliers, out origEllipse);
            await DisplayRegression("Elliptical Test 3 - 1 internal outlier", points, fitEllipse, outliers, origEllipse);
            EllipticalRegression.UnitTest5(out points, out fitEllipse, out outliers, out origEllipse);
            await DisplayRegression("Elliptical Test 4 - 3 external outliers", points, fitEllipse, outliers, origEllipse);
            EllipticalRegression.UnitTest6(out points, out fitEllipse, out outliers, out origEllipse);
            await DisplayRegression("Elliptical Test 5 - 2 external outliers", points, fitEllipse, outliers, origEllipse);


            this.Close();
        }
    }
}
