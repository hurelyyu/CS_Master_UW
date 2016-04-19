import java.io.IOException;
import java.rmi.Remote;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;
import java.util.HashMap;
import java.rmi.Naming;
import java.util.*;

public class RMIserver{	

	
	public static void main(String[] args){
		//Open log
		//int serverPort =Integer.parseInt(args[0]);
	    
		  try{
            String name = "RMIinterface";
            RMIServerStoreData RSS = new RMIServerStoreData();
            //RMIServerStoreData RSS = new RMIServerStoreData();			
            RMIinterface stub = (RMIinterface) UnicastRemoteObject.exportObject(RSS,0);
            System.out.println("@@@@@@@@@@@@@@@@@@@");
           // Registry registry = LocateRegistry.createRegistry(serverPort);   
            System.out.println("Welcome to Yaqun & Xiaoliang's Project 3");
            Registry registry = LocateRegistry.getRegistry(1099);
			   registry.rebind(name, stub);
			   System.out.println("@@@@@@@@@@@@@@@@@@@");
			   //System.out.println("Server bound....");
			   myLog.normal("Server bind......", true);
            
            String name2 = "RMISync";
            RMI_two_phases_commit stub2 = (RMI_two_phases_commit)UnicastRemoteObject.toStub(RSS);
            registry.rebind(name2, stub2);
            myLog.normal("Sync......", true);
            
            List<RMI_two_phases_commit> Replicated_server = new ArrayList<RMI_two_phases_commit>();
            int i = 0; 
            ArrayList<String> missingServers = new ArrayList<String>(Arrays.asList(args));

            RMI_two_phases_commit tpc;
            
            while(!missingServers.isEmpty()){
               System.out.println("Trying Server: " + missingServers.get(0));
               try{
                  tpc = (RMI_two_phases_commit)Naming.lookup("//" + missingServers.get(0) + "/" + name2);
                  if(tpc != null)
                  {
                	  Replicated_server.add(tpc);
                     missingServers.remove(0);
                     System.out.println(missingServers.isEmpty());
                  
                  }
                 }catch(Exception e){
                     myLog.error("Failed to connect to " + missingServers.get(0));
		             System.out.println("Failed to connect to: " + missingServers.get(0));
		             //System.exit(0);
               }
                              
            }
            RSS.setReplicatedServer(Replicated_server);
            myLog.normal("Server is ready!", true);            
		}catch(Exception e){
		      myLog.error("Remote Exception" + e.getMessage());
            e.printStackTrace();
		      System.exit(0);
		  }
	}

}
