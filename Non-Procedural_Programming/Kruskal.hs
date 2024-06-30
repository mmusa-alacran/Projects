import Data.List (sortBy)
import qualified Data.Map as Map
import qualified Data.Set as Set
import System.Environment (getArgs)

-- Define the data types
type Vertex = Int
type Edge = (Vertex, Vertex, Int)  -- (vertex1, vertex2, weight)
type Graph = [Edge]
type UnionFind = Map.Map Vertex Vertex

-- Create a new Union-Find structure from a list of vertices
makeUnionFind :: [Vertex] -> UnionFind
makeUnionFind vertices = Map.fromList [(v, v) | v <- vertices]

-- Find the root of a vertex with path compression
find :: UnionFind -> Vertex -> (UnionFind, Vertex)
find uf vertex =
  case Map.lookup vertex uf of
    Just parent -> if parent == vertex
                   then (uf, vertex)
                   else let (uf', root) = find uf parent
                        in (Map.insert vertex root uf', root)
    Nothing -> error "Vertex not found"

-- Union two sets
union :: UnionFind -> Vertex -> Vertex -> UnionFind
union uf x y =
  let (uf', rootX) = find uf x
      (uf'', rootY) = find uf' y
  in if rootX /= rootY
     then Map.insert rootX rootY uf''
     else uf''

-- Kruskal's algorithm implementation
kruskal :: Graph -> Graph
kruskal edges =
  let sortedEdges = sortBy (\(_, _, w1) (_, _, w2) -> compare w1 w2) edges
      vertices = Set.toList $ Set.fromList $ concat [[v1, v2] | (v1, v2, _) <- edges]
      uf = makeUnionFind vertices
  in snd $ foldl addEdge (uf, []) sortedEdges
  where
    addEdge (uf, mst) edge@(v1, v2, _) =
      let (uf', root1) = find uf v1
          (uf'', root2) = find uf' v2
      in if root1 /= root2
         then (union uf'' root1 root2, edge : mst)
         else (uf'', mst)

-- Function to create a DOT representation of the graph with MST highlighted
createDotGraph :: Graph -> Graph -> String
createDotGraph allEdges mstEdges = header ++ edges ++ footer
  where
    header = "graph G {\n"
    footer = "}"
    mstSet = [(v1, v2) | (v1, v2, _) <- mstEdges] ++ [(v2, v1) | (v1, v2, _) <- mstEdges]
    edges = concatMap formatEdge allEdges
    formatEdge (v1, v2, w) =
      let color = if (v1, v2) `elem` mstSet || (v2, v1) `elem` mstSet
                  then "green"
                  else "black"
          label = show w
      in show v1 ++ " -- " ++ show v2 ++ " [label=\"" ++ label ++ "\",color=\"" ++ color ++ "\"];\n"

-- Main function to read input and run the algorithm
main :: IO ()
main = do
  args <- getArgs
  case args of
    [inputFilePath, outputFilePath] -> do
      edges <- readEdgesFromFile inputFilePath
      let mst = kruskal edges
      putStrLn "Minimum Spanning Tree:"
      print mst
      putStrLn "Generating MST visualization..."
      let dotGraph = createDotGraph edges mst
      writeFile outputFilePath dotGraph
      putStrLn $ "DOT file written to: " ++ outputFilePath
    _ -> error "Usage: program inputFilePath outputFilePath"

-- Read edges from a file
readEdgesFromFile :: FilePath -> IO Graph
readEdgesFromFile filePath = do
  content <- readFile filePath
  return $ map parseEdge (lines content)

parseEdge :: String -> Edge
parseEdge input =
  let [v1, v2, w] = words input
  in (read v1, read v2, read w)
