/*
 *Stephen Mosby
 *Kellen Han-Nin Cheng
 *Aqeel S Bin Rustum
 *Nai-Wei Chen
 *CSS558 Sp14 Project3
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

/**
 * A simple class that implements a KVService, allowing for client machines to 
 * execute methods on a KVStore.
 *
 */
public class RMIServer {

	public static void main(String[] args) {

		try{
			BlockingQueue<String> requests_queue = new LinkedBlockingQueue<String>();
			ConcurrentMap<Float,String> my_log = new ConcurrentSkipListMap<Float, String>();
			String name = "KVService";
			KVStore kvs = new KVStore(Integer.parseInt(args[0]), 
					requests_queue,my_log);
			KVService stub = (KVService)UnicastRemoteObject.exportObject(kvs,0);
			Registry reg = LocateRegistry.createRegistry(1099);
			reg.bind(name, stub);

			String name2 = "KVSync";
			IPaxos stub2 = (IPaxos)UnicastRemoteObject.toStub(kvs);
			reg.bind(name2, stub2);

			List<IPaxos> my_replicated_servers = new ArrayList<IPaxos>();

			int i = 0;
			ArrayList<String> missingServers = new ArrayList<String>(Arrays.asList(args));
			IPaxos paxos;
			missingServers.remove(0); // because it is not a server address
			while(!missingServers.isEmpty()){
				System.out.println("Trying server: " +missingServers.get(0));
				try{
					paxos = (IPaxos) Naming.lookup("//"+ missingServers.get(0) +"/"+name2);
					if(paxos != null){
						my_replicated_servers.add(paxos);
						missingServers.remove(i);
						System.out.println(missingServers.isEmpty());

					}
				}catch(Exception ex){
					System.out.println("Failed to connect to " + missingServers.get(0));
				}
			}

			kvs.setMy_replicated_servers(my_replicated_servers);
			Proposer p = new Proposer(Integer.parseInt(args[0]), 
					my_replicated_servers, 
					requests_queue, my_log);
			LogProcessor lp = new LogProcessor(my_log, kvs);
			Thread t = new Thread(p);
			t.start();
			Thread t2 = new Thread(lp);
		    t2.start();
			System.out.println("Server ready");
		}catch(Exception e){
//			System.out.println("RMIServer error: "+e.getMessage());
			e.printStackTrace();
			System.exit(0);
		}
	}

}