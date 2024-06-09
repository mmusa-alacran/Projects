using Gtk;
using Cairo;
using System;
using System.Collections.Generic;
using System.Linq;

public class Program
{
    public static void Main()
    {
        Application.Init();
        new MainWindow();
        Application.Run();
    }
}

public partial class MainWindow : Window
{
    private DrawingArea canvas;
    private List<PointD> points = new List<PointD>();
    private List<PointD> hull = new List<PointD>();
    private Button calculateButton;
    private Button clearButton;

    public MainWindow() : base("Convex Hull Visualization")
    {
        SetDefaultSize(800, 600);
        SetPosition(WindowPosition.Center);
        DeleteEvent += delegate { Application.Quit(); };

        VBox vbox = new VBox();
        HBox hbox = new HBox();

        canvas = new DrawingArea();
        canvas.AddEvents((int)Gdk.EventMask.ButtonPressMask);
        canvas.ButtonPressEvent += OnCanvasButtonPressed;
        canvas.Drawn += OnDrawn;
        vbox.PackStart(canvas, expand: true, fill: true, padding: 0);

        calculateButton = new Button("Calculate Hull");
        calculateButton.Clicked += OnCalculateClicked;
        hbox.Add(calculateButton);

        clearButton = new Button("Clear");
        clearButton.Clicked += OnClearClicked;
        hbox.Add(clearButton);

        vbox.PackStart(hbox, expand: false, fill: true, padding: 5);
        Add(vbox);
        ShowAll();
    }

    private void OnCanvasButtonPressed(object sender, ButtonPressEventArgs args)
    {
        points.Add(new PointD(args.Event.X, args.Event.Y));
        canvas.QueueDraw();
    }

    private void OnClearClicked(object sender, EventArgs e)
    {
        points.Clear();
        hull.Clear();
        canvas.QueueDraw();
    }

    private void OnCalculateClicked(object sender, EventArgs e)
    {
        hull = CalculateConvexHull(points);
        canvas.QueueDraw();
    }

    private void OnDrawn(object sender, DrawnArgs args)
    {
        DrawingArea area = (DrawingArea)sender;
        Context cr = Gdk.CairoHelper.Create(area.GdkWindow);

        // Draw points
        cr.SetSourceRGB(0, 0, 0);
        cr.LineWidth = 2;
        foreach (var point in points)
        {
            cr.Arc(point.X, point.Y, 5, 0, 2 * Math.PI);
            cr.Fill();
        }

        // Draw Convex Hull
        if (hull.Count > 1)
        {
            cr.SetSourceRGB(1, 0, 0); // Red color for hull
            cr.MoveTo(hull[0].X, hull[0].Y);
            foreach (var point in hull)
            {
                cr.LineTo(point.X, point.Y);
            }
            cr.LineTo(hull[0].X, hull[0].Y);
            cr.Stroke();
        }

        ((IDisposable)cr.Target).Dispose();
        ((IDisposable)cr).Dispose();
    }

    private List<PointD> CalculateConvexHull(List<PointD> points)
    {
        if (points.Count < 3) return new List<PointD>(points);

        List<PointD> sortedPoints = points.OrderBy(p => p.Y).ThenBy(p => p.X).ToList();
        PointD pivot = sortedPoints[0];
        sortedPoints = sortedPoints.OrderBy(p => Math.Atan2(p.Y - pivot.Y, p.X - pivot.X)).ToList();

        Stack<PointD> hull = new Stack<PointD>();
        hull.Push(sortedPoints[0]);
        hull.Push(sortedPoints[1]);

        for (int i = 2; i < sortedPoints.Count; i++)
        {
            PointD top = hull.Pop();
            while (hull.Count > 0 && CrossProduct(hull.Peek(), top, sortedPoints[i]) <= 0)
            {
                top = hull.Pop();
            }
            hull.Push(top);
            hull.Push(sortedPoints[i]);
        }

        return new List<PointD>(hull);
    }

    private double CrossProduct(PointD o, PointD a, PointD b)
    {
        return (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);
    }
}
