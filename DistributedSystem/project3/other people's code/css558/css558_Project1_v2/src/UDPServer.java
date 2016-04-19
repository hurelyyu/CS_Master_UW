import java.io.IOException;
import java.net.SocketException;


public class UDPServer {

	public static void main(String[] args) throws IOException {
		if(args.length != 1) {
			System.out.println("Usage: java UDPServer <port>");
		}
		new UDPServerThread().start();
	}

}
