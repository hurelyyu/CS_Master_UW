package foldfulkalgorithm;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
import java.io.*;
import java.util.*;

public class GraphInput {

    /**
     * Load graph data from a text file via user interaction.
     * This method asks the user for a directory and path name.
     * It returns a hashtable of (String, Vertex) pairs.
     * newgraph needs to already be initialized.
     * @param newgraph  a simple graph
     * @returns a hash table of (String, Vertex) pairs
     */
    public static Hashtable LoadSimpleGraph(SimpleGraph newgraph) {
        try {
        System.out.print("Please enter the full path and file name for the input data: ");
        String userinput;
        InputStreamReader isr = new InputStreamReader(System.in);
        BufferedReader br = new BufferedReader(isr);
        userinput = br.readLine();
        return LoadSimpleGraph(newgraph, userinput);
        }
        catch (Exception e){
            System.out.println(e.getMessage());    
        }
        return new Hashtable();
    }

    /**
     * Load graph data from a text file.
     * The format of the file is:
     * Each line of the file contains 3 tokens, where the first two are strings
     * representing vertex labels and the third is an edge weight (a double).
     * Each line represents one edge.
     * 
     * This method returns a hashtable of (String, Vertex) pairs.
     * 
     * @param newgraph  a graph to add edges to. newgraph should already be initialized
     * @param pathandfilename  the name of the file, including full path.
     * @returns  a hash table of (String, Vertex) pairs
     */
    public static Hashtable LoadSimpleGraph(SimpleGraph newgraph, String pathandfilename){
        BufferedReader  inbuf = InputLib.fopen(pathandfilename);
        System.out.println("Opened " + pathandfilename + " for input.");
        String  line = InputLib.getLine(inbuf); // get first line
        StringTokenizer sTok;
        int n, linenum = 0;
        Hashtable table = new Hashtable();
        SimpleGraph sg = newgraph;

        while (line != null) {
            linenum++;
            sTok = new StringTokenizer(line);
            n = sTok.countTokens();
            if (n==3) {
                Double edgedata;
                Vertex v1, v2;
                String v1name, v2name;

                v1name = sTok.nextToken();
                v2name = sTok.nextToken();
                edgedata = new Double(Double.parseDouble(sTok.nextToken()));
                v1 = (Vertex) table.get(v1name);
                if (v1 == null) {
                    // System.out.println("New vertex " + v1name);
                    v1 = sg.insertVertex(null, v1name);
                    table.put(v1name, v1);
                }
                v2 = (Vertex) table.get(v2name);
                if (v2 == null) {
                    // System.out.println("New vertex " + v2name);
                    v2 = sg.insertVertex(null, v2name);
                    table.put(v2name, v2);
                }
                // System.out.println("Inserting edge (" + v1name + "," + v2name + ")" + edgedata);
                // set the name of the edge as v1_v2_forwardedge
                sg.insertEdge(v1,v2,edgedata, v1.getName()+"_"+v2.getName()+"_forwardedge");
            }
            else {
                System.err.println("Error:invalid number of tokens found on line " +linenum+ "!");
                return null;
            }
            line = InputLib.getLine(inbuf);
        }

        InputLib.fclose(inbuf);
        System.out.println("Successfully loaded "+ linenum + " lines. ");
        return table;
    }


    /**
     * Code to test the methods of this class.
     */
    public static void main (String args[]) {
        SimpleGraph G;
        Vertex v;
        Edge e;
        G = new SimpleGraph();
        // randomly generated graph
        // LoadSimpleGraph(G, "/Users/XinhelovesMom/NetBeansProjects/FoldFulkAlgorithm/src/foldfulkalgorithm/n10-m10-cmin5-cmax10-f30.txt");
        // bipartite graph
        LoadSimpleGraph(G, "/Users/XinhelovesMom/NetBeansProjects/FoldFulkAlgorithm/src/foldfulkalgorithm/g1.txt");
//        Iterator i;
//
//        System.out.println("Iterating through vertices...");
//        for (i= G.vertices(); i.hasNext(); ) {
//            v = (Vertex) i.next();
//            System.out.println("found vertex " + v.getName());
//        }
//
//        System.out.println("Iterating through adjacency lists...");
//        for (i= G.vertices(); i.hasNext(); ) {
//            v = (Vertex) i.next();
//            System.out.println("Vertex "+v.getName());
//            Iterator j;
//            
//            for (j = G.incidentEdges(v); j.hasNext();) {
//                e = (Edge) j.next();
//                System.out.println("  found edge " + e.getName());
//            }
//        }
        Vertex source = (Vertex) G.vertexList.getFirst();
        Vertex target = (Vertex) G.vertexList.getLast();
        FoldFulkAlgorithm ffa = new FoldFulkAlgorithm(G,source,target);
        System.out.println(ffa.GetMaxFlow());
        // test if the avaliable path exist
//        DFSImpl impl = new DFSImpl(G,source,target);
//        List<Edge> path = impl.dfsImpl();
//        System.out.println(path.isEmpty());
//        for(Edge ee:path) {
//            System.out.print(ee.getName());
//        }
    }
}
