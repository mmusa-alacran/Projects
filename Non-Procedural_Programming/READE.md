```markdown
# Kruskal's Algorithm in Haskell

This repository contains a Haskell implementation of Kruskal's algorithm for finding the Minimum Spanning Tree (MST) of a graph. The program reads a list of edges from an input file, computes the MST, and generates a DOT file for visualizing the graph with the MST highlighted.

## Files

- `Kruskal.hs`: The main Haskell program that implements Kruskal's algorithm.
- `README.md`: This readme file.

## Requirements

To run this program, you need:

- [GHC (The Glasgow Haskell Compiler)](https://www.haskell.org/ghc/)
- [Graphviz](https://graphviz.org/)


### Using GHC

If you prefer to use GHC directly, you can compile and run the program with these commands:

```sh
ghc -package containers -o kruskal Kruskal.hs
./kruskal <inputFilePath> <outputFilePath>
```

## Usage

The program expects two command-line arguments:

1. `inputFilePath`: The path to the input file containing the graph edges.
2. `outputFilePath`: The path where the DOT file for the MST visualization will be saved.

### Input File Format

The input file should contain one edge per line, with the format:

```
vertex1 vertex2 weight
```

For example:

```
1 2 4
2 3 3
1 3 5
3 4 2
```

### Running the Program

To run the program, use the following command:

```sh
./kruskal <inputFilePath> <outputFilePath>
```

For example:

```sh
./kruskal graph.txt mst.dot
```

This will read the graph from `graph.txt`, compute the MST, and write the DOT file to `mst.dot`.

## Output

The program outputs the MST to the console and generates a DOT file for visualizing the graph with the MST highlighted in green. The DOT file can be viewed using tools like [Graphviz](http://www.graphviz.org/).

### Generating a PNG from the DOT File

To visualize the DOT file as a PNG image, you need to use Graphviz. Here are the steps:

1. **Install Graphviz**:
   - **On Ubuntu/Debian**: `sudo apt-get install graphviz`
   - **On macOS**: `brew install graphviz`
   - **On Windows**: Download and install from the [Graphviz website](https://graphviz.org/download/).

2. **Convert the DOT file to PNG**:
   Use the `dot` command to generate a PNG image from the DOT file:

   ```sh
   dot -Tpng <outputFilePath> -o <outputPngFilePath>
   ```

   For example, if your DOT file is `mst.dot` and you want to generate `mst.png`, run:

   ```sh
   dot -Tpng mst.dot -o mst.png
   ```

### Example

Given the following `graph.txt`:

```
1 2 4
2 3 3
1 3 5
3 4 2
```

Running the program:

```sh
./kruskal graph.txt mst.dot
```

Produces the following output:

```
Minimum Spanning Tree:
[(3,4,2),(2,3,3),(1,2,4)]
Generating MST visualization...
DOT file written to: mst.dot
```

And generates the following `mst.dot` file:

```
graph G {
1 -- 2 [label="4",color="green"];
2 -- 3 [label="3",color="green"];
1 -- 3 [label="5",color="black"];
3 -- 4 [label="2",color="green"];
}
```

To convert this DOT file to a PNG file:

```sh
dot -Tpng mst.dot -o mst.png
```

This will generate an image file `mst.png` that you can view with any image viewer.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation uses the Union-Find data structure to efficiently handle the merging of sets during the MST construction. The DOT file generation helps in visualizing the graph and the resulting MST.