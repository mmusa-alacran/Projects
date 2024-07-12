
---

# Convex Hull Visualization

This is a GTK-based application for visualizing the convex hull of a set of points. The application allows users to add points by clicking on a canvas, calculate the convex hull of those points, and view the result visually. The convex hull is calculated using the Graham scan algorithm.

## Features

- Add points by clicking on the canvas.
- Calculate and visualize the convex hull.
- Clear the canvas to reset points and hull.
- Responsive UI during long-running calculations using asynchronous tasks.

## Built With

- **C#** - The primary programming language used.
- **GTK#** - The graphical user interface toolkit.
- **Cairo** - The graphics library used for rendering the convex hull.

## Prerequisites

- .NET SDK (ensure you have the correct version installed)
- GTK# (GTK bindings for C#)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/420muxa/Projects.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Programming_2-C#
    ```
3. Restore dependencies:
    ```bash
    dotnet restore
    ```
4. Build the application:
    ```bash
    dotnet build
    ```
5. Run the application:
    ```bash
    dotnet run
    ```

## Usage

1. **Launch the Application:** Run the application following the build instructions above. A window titled "Convex Hull Visualization" will open.
2. **Add Points:** Click anywhere within the gray canvas area to add points (Add minimum 3 points to calculate the convex hull).
3. **Calculate Convex Hull:** Click on the "Calculate Hull" button to begin the convex hull calculation. The points will be connected step-by-step, highlighting the formation of the convex hull.
4. **Clear the Canvas:** Use the "Clear" button to remove all points and reset the canvas.


## Troubleshooting

### Error on Launch

If you encounter the following error on Linux:

```
symbol lookup error: /snap/core20/current/lib/x86_64-linux-gnu/libpthread.so.0: undefined symbol: __libc_pthread_init, version GLIBC_PRIVATE
```

This may be due to an environment configuration issue related to GTK. To resolve this, run the following command in your terminal before starting the application:

```bash
unset GTK_PATH
```

This command clears the `GTK_PATH` environment variable, which can interfere with the library path resolution.

## Code Explanation


### Main Program

The `Program` class initializes and starts the GTK application.

```csharp
public class Program
{
    public static void Main()
    {
        Application.Init();
        new MainWindow();
        Application.Run();
    }
}
```

### MainWindow Class

The `MainWindow` class contains all UI elements and logic for the convex hull visualization.

- **UI Elements**: `DrawingArea` for the canvas, `Button` for calculating the hull and clearing the canvas.
- **Event Handlers**: Handlers for button clicks and canvas drawing.

### Convex Hull Calculation

The convex hull is calculated using the Graham scan algorithm, which sorts the points and processes them to form the hull.

#### Asynchronous Calculation

The `CalculateConvexHullAsync` method performs the calculation asynchronously. This keeps the UI responsive during long computations.

```csharp
private async Task CalculateConvexHullAsync(List<PointD> points)
{
    if (points.Count < 3) return;

    List<PointD> sortedPoints = points.OrderBy(p => p.Y).ThenBy(p => p.X).ToList();
    PointD pivot = sortedPoints[0];
    sortedPoints = sortedPoints.OrderBy(p => Math.Atan2(p.Y - pivot.Y, p.X - pivot.X)).ToList();

    Stack<PointD> tempHull = new Stack<PointD>();
    tempHull.Push(sortedPoints[0]);
    tempHull.Push(sortedPoints[1]);

    for (int i = 2; i < sortedPoints.Count; i++)
    {
        if (!isCalculating) return;

        selectedPoint = sortedPoints[i];
        PointD top = tempHull.Pop();
        while (tempHull.Count > 0 && CrossProduct(tempHull.Peek(), top, sortedPoints[i]) <= 0)
        {
            top = tempHull.Pop();
        }
        tempHull.Push(top);
        tempHull.Push(sortedPoints[i]);

        hull = new List<PointD>(tempHull);
        canvas.QueueDraw();
        await Task.Delay(500);
    }

    hull = new List<PointD>(tempHull);
    selectedPoint = null;
    canvas.QueueDraw();
}
```

### Cross Product Calculation

A helper method to determine the direction of the turn using the cross product.

```csharp
private double CrossProduct(PointD o, PointD a, PointD b)
{
    return (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);
}
```

## License

This project is licensed under the MIT License.

---
