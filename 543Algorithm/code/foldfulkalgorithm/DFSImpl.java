
package foldfulkalgorithm;

import static foldfulkalgorithm.GraphInput.LoadSimpleGraph;
import java.util.Iterator;
import java.util.Hashtable;
import java.util.List;
import java.util.LinkedList;

public class DFSImpl {
    //private boolean[] marked;
    private Hashtable<Vertex,Boolean> marked;
    private int count;
    private LinkedList<Edge> path;
    public Double bottleneck;
    public Vertex source;
    public Vertex target;
    public SimpleGraph G;
    
    // constructor of deapth first search implementation
    public DFSImpl(SimpleGraph G, Vertex s, Vertex t) { 
        this.G = G;
        source = s;
        target = t;
        marked = new Hashtable();
        count = 0;
        path = new LinkedList();
        bottleneck = Double.MAX_VALUE;
        Iterator i;
        Vertex v;
        for (i = G.vertices();i.hasNext();) {
            v = (Vertex) i.next();
            marked.put(v, false);
        }
    }
    public List<Edge> dfsImpl() {
        dfs(G,source);
        if(path.isEmpty()) {
            path = new LinkedList();
        }
        else {
            Edge e = path.getLast();
//            System.out.println(e.getFirstEndpoint().getName());
//            System.out.println(e.getSecondEndpoint().getName());
            if(e.getSecondEndpoint() != target) {
                // System.out.print(e.getName()+" IS ");
                System.out.println("NOT REACHABLE");
                path = new LinkedList();
                //path = new LinkedList();
                dfsImpl();
            }
        }
        return path;
    }
    
    private boolean dfs(SimpleGraph G, Vertex v) {
        count++;
        marked.put(v,true);
        Iterator j;
        for(j = G.incidentEdges(v); j.hasNext();) {
            Edge e = (Edge) j.next();
            if((Double) e.getData() != 0.0 && e.getFirstEndpoint() == v) {
                Vertex adjv = G.opposite(v,e);
                //System.out.println(adjv.getName());
                if( !marked.get(adjv) ) {
                    System.out.println(e.getName());
                    path.add(e);
                    bottleneck = Math.min((Double) e.getData(), bottleneck);
//                    System.out.println(adjv.getName());
                    if(adjv == target) return true;
                    else return dfs(G,adjv);
                    //dfs(G,adjv);
                }
            }
        }
        return false;
    }
    
    
    public boolean marked(Vertex v) {
        if(!marked.containsKey(v)) return false;
        return marked.get(v);
    }
    
    // returns the number of vertices connected to the source vertex
    public int count() {
        return count;
    }
    
    public static void main(String[] args) {
        SimpleGraph G = new SimpleGraph();
//        Vertex s,t,u,v;
//        Edge a,b,c,d,e;
//        s = G.insertVertex(null, "Source");
//        u = G.insertVertex(null, "U");
//        a = G.insertEdge(s, u, 20.0, "a");
//        v = G.insertVertex(null, "V");
//        b = G.insertEdge(s, v, 10.0, "b");
//        c = G.insertEdge(u, v, 30.0, "c");
//        t = G.insertVertex(null, "Target");
//        d = G.insertEdge(u,t,10.0,"d");
//        e = G.insertEdge(v, t, 20.0, "e");
        // System.out.println(d.getData());
        LoadSimpleGraph(G, "/Users/XinhelovesMom/NetBeansProjects/FoldFulkAlgorithm/src/foldfulkalgorithm/g1.txt");
        Vertex source = (Vertex) G.vertexList.getFirst();
        Vertex target = (Vertex) G.vertexList.getLast();
        DFSImpl impl = new DFSImpl(G,source,target);
        //System.out.println(impl.dfs(G, source));
        List<Edge> path = impl.dfsImpl();
        System.out.println(path.isEmpty());
        for(Edge ee:path) {
            System.out.print(ee.getName());
        }        
    }
    
}
