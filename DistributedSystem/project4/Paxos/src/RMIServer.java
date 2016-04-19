/**
 * @author yaqunyu
 */
import java.rmi.Naming;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.Scanner;
/**
 * A simple class that implements a RMIinterface, allowing for client machines to 
 * execute methods on a RMIServerStoreData.
 *
 */
public class RMIServer {

	public static void main(String[] args) {

		try{
			System.out.println ("we are here");
			BlockingQueue<String> requests_queue = new LinkedBlockingQueue<String>();
			//This class implements a concurrent variant of SkipLists providing expected average log(n) time cost 
			//for the containsKey, get, put and remove operations and their variants.
			ConcurrentMap<Float,String> my_log = new ConcurrentSkipListMap<Float, String>();
			String name = "RMIinterface";
			RMIServerStoreData RSS = new RMIServerStoreData(Integer.parseInt(args[0]),requests_queue,my_log);
			RMIinterface stub = (RMIinterface)UnicastRemoteObject.exportObject(RSS,0);
			System.out.println("Welcome to Yaqun & Xiaoliang's Project 4");
			Registry reg = LocateRegistry.createRegistry(1099);
			reg.bind(name, stub);
			myLog.normal("Server bind......", true);
			
			String name2 = "RMISync";
			Paxos stub2 = (Paxos)UnicastRemoteObject.toStub(RSS);
			reg.bind(name2, stub2);
			myLog.normal("Sync......", true);
			
			List<Paxos> replicated_servers = new ArrayList<Paxos>();

			System.out.println("Press enter to continue...");
            Scanner keyboard = new Scanner(System.in);
            keyboard.nextLine();

			int i = 0;
			ArrayList<String> missingServers = new ArrayList<String>(Arrays.asList(args));
			Paxos paxos;
			missingServers.remove(0); // because it is not a server address
			while(!missingServers.isEmpty()){
				myLog.error("Trying server: " +missingServers.get(0));
				try{
					paxos = (Paxos) Naming.lookup("//"+ missingServers.get(0) +"/"+name2);
					if(paxos != null){
						replicated_servers.add(paxos);
						missingServers.remove(i);
						System.out.println(missingServers.isEmpty());

					}
				}catch(Exception ex){
					myLog.error("Failed to connect to " + missingServers.get(0));
				}
			}

			RSS.setMy_replicated_servers(replicated_servers);
			Proposer p = new Proposer(Integer.parseInt(args[0]), 
					replicated_servers, 
					requests_queue, my_log);
			LogProcessor lp = new LogProcessor(my_log, RSS);
			Thread t = new Thread(p);
			t.start();
			Thread t2 = new Thread(lp);
		    t2.start();
			System.out.println("Server ready");
		}catch(Exception e){
//			System.out.println("RMIServer error: "+e.getMessage());
			myLog.error(e.getMessage());
			System.exit(0);
		}
	}

}