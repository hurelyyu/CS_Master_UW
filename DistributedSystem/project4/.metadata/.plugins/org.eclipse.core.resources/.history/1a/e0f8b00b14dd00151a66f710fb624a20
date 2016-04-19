/*
 *Stephen Mosby
 *Kellen Han-Nin Cheng
 *Aqeel S Bin Rustum
 *Nai-Wei Chen
 *CSS558 Sp14 Project3
*/

import java.io.IOException;
import java.rmi.NotBoundException;

/**
 * This is a simple class for creating Client threads.  Use NUM_CLIENTS to 
 * specify the number of client threads you want to create on one machine.
 */
public class ClientMain {

	/**A constant variable for specifying the number of client threads 
	 * to create.
	 */
	private static int NUM_CLIENTS = 1;

	/**
	 * The main method, where the client threads are created.
	 * 
	 * @param args[0] specifies the server to connect to
	 * @throws NotBoundException
	 * @throws IOException
	 */
	public static void main(String[] args) 
			throws NotBoundException, IOException {
		if(args.length<1){
			System.out.println("Usage: java ClientMain <host>");
			System.exit(0);
		}

		for(int client_id = 1; client_id <= NUM_CLIENTS; client_id++) {
			(new Thread(new RMIClient(args[0],client_id))).start();
		}
	}
}