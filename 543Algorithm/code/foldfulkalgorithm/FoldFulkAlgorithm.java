/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package foldfulkalgorithm;

/**
 *
 * @author XinhelovesMom
 */
import static foldfulkalgorithm.GraphInput.LoadSimpleGraph;
import java.util.Iterator;
import java.util.HashSet;
import java.util.List;
import java.util.LinkedList;
public class FoldFulkAlgorithm {
    // residual graph
    SimpleGraph RG;

    HashSet<String> backEdges = new HashSet();
    DFSImpl dfsimpl;
    Vertex source;
    Vertex target;
    List<Edge> path = new LinkedList();
    
    public FoldFulkAlgorithm (SimpleGraph RG, Vertex source,Vertex target) {
        this.RG = RG;
        this.source = source;
        this.target = target;
    }
    // the maximum flow should be the sum of all the edges value 
    // coming out form the source node when there is no 
    // augmenting path from source to target in residual graph
    public Double GetMaxFlow() {
        Double maxflow = 0.0;
        Iterator j;
        FoldFulkImpl();
        for(j = RG.incidentEdges(source); j.hasNext();) {
            Edge e = (Edge) j.next();
            String name = (String) e.getName();
            if(name.contains("backwardedge")) {
                System.out.print(e.getName()+" flows: ");
                System.out.println(e.getData());
                maxflow = maxflow + (Double) e.getData();
            }
        }
        return maxflow;
    }
    
    public void FoldFulkImpl() {
        dfsimpl = new DFSImpl(RG,source,target);
        // if there is an augmenting path in graph
        path = dfsimpl.dfsImpl();
        System.out.println(path.size());
        String n;
        while(!path.isEmpty()) {
            Iterator i;
            Double value;
            System.out.println("The Avaliable Augmenting Path is:");
            path.forEach(ele->System.out.print(ele.getName()+"->"));
            System.out.println();
            System.out.println("The Flow is: "+dfsimpl.bottleneck);
            for(Edge e:path){
                value = (Double) e.getData();
                Vertex v1 = e.getFirstEndpoint();
                Vertex v2 = e.getSecondEndpoint();
                e.setData(value - dfsimpl.bottleneck);
                n = (String) e.getName();
                //if this edge is not backward edge then check if it contains the backward
                // edge of itself 
                if(!n.contains("backwardedge")) {
                    // adding a backward edge and 
                    // record it in hashset backedges...
                    // if the name contains backedge means the edge is backward edge
                    String name = ""+v2.getName()+"_"+v1.getName()+"_backwardedge";
                    Edge backe;
                    if(!backEdges.contains(name)) {
                        // if the backward edge does not exist
                        // Edge backe = new Edge(v2, v1, dfsimpl.bottleneck, name);
                        RG.insertEdge(v2, v1, dfsimpl.bottleneck, name);
                        backEdges.add(name);
                        System.out.println("BACKWARD EDGE: "+name+" IS ADDED");
                    }
                    // if the graph does contains this backward edge
                    // add the backward edge by the value of 
                    else {
                        Iterator j;
                        for (j = RG.edges(); j.hasNext();) {
                            backe = (Edge) j.next();
                            // System.out.println("backward edge is: "+backe.getName());
                            String sname = (String) backe.getName();
                            if(sname.contains(name)) {
                                value = (Double) backe.getData();
                                backe.setData(value + dfsimpl.bottleneck);
                                System.out.println(name+" is added value "+dfsimpl.bottleneck);
                            }
                        }
                    } 
                }
                else {
                    Iterator j;
                    for(j = RG.edges();j.hasNext();) {
                        Edge forwarde = (Edge) j.next();
                        if(forwarde.getFirstEndpoint() == v2 && forwarde.getSecondEndpoint() == v1) {
                            value = (Double) forwarde.getData();
                            forwarde.setData(value + dfsimpl.bottleneck);
                            System.out.println("Edge name is: "+forwarde.getName()+"; value is: "+value+dfsimpl.bottleneck);
                        }
                        
                    }
                }
            }
            dfsimpl = new DFSImpl(RG,source,target);
            path = dfsimpl.dfsImpl();
        }
//        else {
        System.out.println("No Augmenting Path in the Residual Graph");
        System.out.println("PROCESS COMPLETED");
//        }
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
        LoadSimpleGraph(G, "/Users/yyq/Downloads/UW/foldfulkalgorithm/g1.txt");
        Vertex source = (Vertex) G.vertexList.getFirst();
        Vertex target = (Vertex) G.vertexList.getLast();
        FoldFulkAlgorithm ffa = new FoldFulkAlgorithm(G,source,target);
        System.out.println(ffa.GetMaxFlow());
    }
    
}
