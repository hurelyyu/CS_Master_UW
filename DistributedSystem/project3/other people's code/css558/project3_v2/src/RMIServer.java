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
import java.util.*;

/**
 * A simple class that implements a KVService, allowing for client machines to 
 * execute methods on a KVStore.
 *
 */
public class RMIServer {

	public static void main(String[] args) {

		try{
			String name = "KVService";
			KVStore kvs = new KVStore();
			KVService stub = (KVService)UnicastRemoteObject.exportObject(kvs,0);
			Registry reg = LocateRegistry.createRegistry(1099);
			reg.bind(name, stub);

			String name2 = "KVSync";
			RMItwophasecommit stub2 = (RMItwophasecommit)UnicastRemoteObject.toStub(kvs);
//			Registry reg2 = LocateRegistry.createRegistry(1088);
			reg.bind(name2, stub2);

			List<RMItwophasecommit> my_replicated_servers = new ArrayList<RMItwophasecommit>();

			int i = 0;
			ArrayList<String> missingServers = new ArrayList<String>(Arrays.asList(args));
//			System.out.println(args[0]);
//			System.out.println(missingServers.size());
//			for(String s:missingServers){
//				System.out.println(s);
//			}
			RMItwophasecommit tpc;
			while(!missingServers.isEmpty()){
				System.out.println("Trying server: " +missingServers.get(0));
				try{
					tpc = (RMItwophasecommit) Naming.lookup("//"+ missingServers.get(0) +"/"+name2);
					if(tpc != null){
						my_replicated_servers.add(tpc);
						missingServers.remove(i);
						System.out.println(missingServers.isEmpty());

					}
				}catch(Exception ex){
					System.out.println("Failed to connect to " + missingServers.get(0));
				}
			}

			kvs.setMy_replicated_servers(my_replicated_servers);

			System.out.println("Server ready");
		}catch(Exception e){
//			System.out.println("RMIServer error: "+e.getMessage());
			e.printStackTrace();
			System.exit(0);
		}
	}

}