
public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		//String filename = "src/n10-m10-cmin5-cmax10-f30.txt";
		//String filename = "src/n4-m4-f5.txt";
		String filename = "src/n10-m10-cmin5-cmax10-f30.txt";
		SimpleGraph G;
        G = new SimpleGraph();
        GraphInput.LoadSimpleGraph(G, filename);
        
        PreflowPush p = new PreflowPush(G);
        
        
        p.Preflow_Push_Algorithm();
	}

}
