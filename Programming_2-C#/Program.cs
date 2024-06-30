using Gtk;
using Cairo;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

// Main class to start the GTK application
public class Program
{
    public static void Main()
    {
        Application.Init(); // Initialize GTK application environment
        new MainWindow(); // Create a new instance of the main window
        Application.Run(); // Run the GTK application loop
    }
}

// Main window class containing all UI elements and logic for convex hull visualization
public partial class MainWindow : Window
{
    private DrawingArea canvas; // Area where points and convex hull will be drawn
    private List<PointD> points = new List<PointD>(); // List to store user-added points
    private List<PointD> hull = new List<PointD>(); // List to store points making up the convex hull
    private Button calculateButton; // Button to trigger convex hull calculation
    private Button clearButton; // Button to clear points and hull from canvas
    private bool isCalculating = false; // Flag to manage state during calculations
    private PointD? selectedPoint = null; // To hold the point currently being considered in the algorithm

    // Constructor for the UI
    public MainWindow() : base("Convex Hull Visualization")
    {
        SetDefaultSize(800, 600); // Set the initial size of the window
        SetPosition(WindowPosition.Center); // Center the window on the screen
        DeleteEvent += delegate { Application.Quit(); }; // Exit the application when window is closed

        Box mainBox = new Box(Orientation.Vertical, 10); // Vertical box for layout
        mainBox.Margin = 20; // Margin around the box

        Box buttonBox = new Box(Orientation.Horizontal, 10); // Horizontal box for buttons
        buttonBox.Halign = Align.Center; // Center-align the button box

        canvas = new DrawingArea(); // Canvas for drawing
        canvas.AddEvents((int)Gdk.EventMask.ButtonPressMask); // Subscribe to button press events
        canvas.ButtonPressEvent += OnCanvasButtonPressed; // Event handler for button press
        canvas.Drawn += OnDrawn; // Event handler for drawing on the canvas
        canvas.SetSizeRequest(800, 500); // Specify size of the canvas
        canvas.MarginBottom = 10; // Margin below the canvas
        mainBox.PackStart(canvas, expand: true, fill: true, padding: 0); // Add canvas to the main box

        calculateButton = new Button("Calculate Hull"); // Initialize the calculate button
        calculateButton.Clicked += async (sender, e) => await OnCalculateClickedAsync(sender, e); // Asynchronous event handler for button click
        calculateButton.SetSizeRequest(150, 50); // Size of the calculate button
        calculateButton.MarginEnd = 10; // Margin on the end of the calculate button
        buttonBox.PackStart(calculateButton, expand: false, fill: false, padding: 0); // Add calculate button to button box

        clearButton = new Button("Clear"); // Initialize the clear button
        clearButton.Clicked += OnClearClicked; // Event handler for clear button click
        clearButton.SetSizeRequest(150, 50); // Size of the clear button
        buttonBox.PackStart(clearButton, expand: false, fill: false, padding: 0); // Add clear button to button box

        mainBox.PackStart(buttonBox, expand: false, fill: false, padding: 0); // Add button box to the main box
        Add(mainBox); // Add main box to the window
        ShowAll(); // Show all widgets
    }

    // Event handler for canvas button press to add new points
    private void OnCanvasButtonPressed(object sender, ButtonPressEventArgs args)
    {
        if (isCalculating) return; // Prevent adding points during calculation

        points.Add(new PointD(args.Event.X, args.Event.Y)); // Add a new point where user clicks
        canvas.QueueDraw(); // Redraw the canvas
    }

    // Event handler to clear all points and reset the hull
    private void OnClearClicked(object? sender, EventArgs e)
    {
        points.Clear(); // Clear all points
        hull.Clear(); // Clear the hull
        isCalculating = false; // Reset calculation flag
        selectedPoint = null; // Deselect any selected point
        canvas.QueueDraw(); // Redraw the canvas
    }

    // Event handler to start convex hull calculation
    private async Task OnCalculateClickedAsync(object? sender, EventArgs e)
    {
        if (isCalculating) return; // Prevent starting a new calculation if already calculating

        isCalculating = true; // Set the calculating flag
        await CalculateConvexHullAsync(points); // Start the convex hull calculation
        isCalculating = false; // Reset the calculating flag after completion
    }

    // Drawing handler for the canvas
    private void OnDrawn(object sender, DrawnArgs args)
    {
        DrawingArea area = (DrawingArea)sender;
        Context cr = new Context(args.Cr.GetTarget()); // Create a new drawing context

        cr.SetSourceRGB(0.1, 0.1, 0.1); // Set background color to grey
        cr.Paint(); // Apply the background color

        // Draw each point added by the user
        foreach (var point in points)
        {
            if (selectedPoint.HasValue && point.Equals(selectedPoint.Value))
            {
                cr.SetSourceRGB(0, 1, 0); // Highlight the selected point with green
            }
            else
            {
                cr.SetSourceRGB(1, 1, 1); // Draw unselected points with white
            }
            cr.LineWidth = 1; // Set line width for drawing points
            cr.Arc(point.X, point.Y, 8, 0, 2 * Math.PI); // Draw an arc for the point
            cr.Fill(); // Fill the arc to make the point visible
        }

        // Draw the convex hull if it exists
        if (hull.Count > 1)
        {
            cr.SetSourceRGB(1, 0, 0); // Set color to red for the hull
            cr.LineWidth = 1; // Set line width for hull
            cr.MoveTo(hull[0].X, hull[0].Y); // Start at the first point of the hull
            foreach (var point in hull)
            {
                cr.LineTo(point.X, point.Y); // Draw line to the next point in the hull
            }
            cr.LineTo(hull[0].X, hull[0].Y); // Complete the loop back to the first point
            cr.Stroke(); // Stroke the path to draw the hull
        }

        ((IDisposable)cr.GetTarget()).Dispose(); // Dispose drawing resources
        ((IDisposable)cr).Dispose();
    }

    // Asynchronous method to calculate convex hull
    private async Task CalculateConvexHullAsync(List<PointD> points)
    {
        if (points.Count < 3) return; // Early return if not enough points to form a hull

        List<PointD> sortedPoints = points.OrderBy(p => p.Y).ThenBy(p => p.X).ToList(); // Sort points by Y and then by X
        PointD pivot = sortedPoints[0]; // Choose the first point as pivot
        sortedPoints = sortedPoints.OrderBy(p => Math.Atan2(p.Y - pivot.Y, p.X - pivot.X)).ToList(); // Sort points by angle to pivot

        Stack<PointD> tempHull = new Stack<PointD>(); // Stack to build the hull incrementally
        tempHull.Push(sortedPoints[0]); // Push the first point onto the stack
        tempHull.Push(sortedPoints[1]); // Push the second point onto the stack

        for (int i = 2; i < sortedPoints.Count; i++)
        {
            if (!isCalculating) return; // Stop if calculation was cancelled

            selectedPoint = sortedPoints[i]; // Mark the current point as selected
            PointD top = tempHull.Pop(); // Pop the top point from the stack
            while (tempHull.Count > 0 && CrossProduct(tempHull.Peek(), top, sortedPoints[i]) <= 0)
            {
                top = tempHull.Pop(); // Pop the next point if it does not make a left turn
            }
            tempHull.Push(top); // Push the last point back onto the stack
            tempHull.Push(sortedPoints[i]); // Push the current point onto the stack

            hull = new List<PointD>(tempHull); // Update the hull with the current stack content
            canvas.QueueDraw(); // Redraw the canvas with the updated hull
            await Task.Delay(500); // Delay to animate the step
        }

        hull = new List<PointD>(tempHull); // Set the final hull
        selectedPoint = null; // Clear the selected point
        canvas.QueueDraw(); // Redraw the canvas to show the final hull
    }

    // Helper method to calculate cross product
    private double CrossProduct(PointD o, PointD a, PointD b)
    {
        // Cross product calculation for determining the direction of the turn
        return (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);
    }
}
