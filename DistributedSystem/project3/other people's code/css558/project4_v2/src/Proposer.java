import java.rmi.RemoteException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.RunnableFuture;
import java.util.concurrent.TimeUnit;

public class Proposer implements Runnable {
	private static final int QUORUM_SIZE = 3;
	private List<IPaxos> my_replicated_server;
	private BlockingQueue<String> requests_queue;
	private int server_id;
	private float proposal_number = 0;
	private ConcurrentMap<Float, String> my_log;
	List<String> replies;

	public Proposer(int server_id, final List<IPaxos> my_replicated_server,
			BlockingQueue<String> requests_queue,
			ConcurrentMap<Float, String> the_log) {
		this.server_id = server_id;
		proposal_number = ((float)server_id) / 10;
		this.my_replicated_server = my_replicated_server;
		this.requests_queue = requests_queue;
		my_log = the_log;
		replies = new ArrayList<String>();
	}

	public enum Phase {
		PREPARE, ACCEPT
	}

	@Override
	public void run() {
		float max_accepted_proposal_number = 0;
		float max_returned_proposal_number = 0;
		String[] prepare_response = null;
		String accept_response = null;
		boolean is_peeked = false;
		int quorum_count;
		String request = null;
		while (true) {
			if (!requests_queue.isEmpty()) {
				// prepare phase
				quorum_count = 0;
				while (quorum_count < QUORUM_SIZE) {
					proposal_number++;
					quorum_count = 0;
					request = null;
					replies.clear();
					try {
						System.out.println("Preparing Task " + proposal_number);
						scheduleTask(Phase.PREPARE, null);
					} catch (Exception e) {
						e.printStackTrace();
					}
					System.out.println(replies.toString());
					for(String s : replies){
						prepare_response = s.trim().split(",");
						if(prepare_response.length > 1){
							System.out.print(prepare_response);
							if(max_accepted_proposal_number < Float
									.parseFloat(prepare_response[0])){
								System.out.println(max_accepted_proposal_number);
								max_accepted_proposal_number = Float
										.parseFloat(prepare_response[0]);
								if(!prepare_response[1].equals("NO_ACCEPTED_VALUE_YET")){
									request = prepare_response[1];
								}
							}
							quorum_count++;
						}
						else{
							// it will return the minimal promised proposal number
							max_returned_proposal_number = Math.max(max_returned_proposal_number ,
									Float.parseFloat(prepare_response[0]));
						}
					}
					System.out.println("quorum_count: " + quorum_count);
					if (request == null) {
						System.out.println("peeking at queue");
						request = requests_queue.peek();
						is_peeked = true;
					}
					if(quorum_count < QUORUM_SIZE && 
							max_returned_proposal_number > 0){
						proposal_number = max_returned_proposal_number;
					}
				}
				replies.clear();

				// accept phase
				try {
					System.out.println("Accepting: " + request);
					scheduleTask(Phase.ACCEPT, request);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} 
				quorum_count = replies.size();
				for(String s : replies){
					if(proposal_number < Float.parseFloat(s)){
						quorum_count--;
					}
				}
				if(quorum_count >= QUORUM_SIZE){
					if(is_peeked){
						String removed=requests_queue.remove();
						System.out.println("removed from queue: "+removed+ " "+request);
					}
					request = request + ",[chosen]";
					my_log.put(proposal_number, request);

					//learning phase
					for(final IPaxos p : my_replicated_server){
						try {
							System.out.println("Teaching " + p.getServer_id() + 
									proposal_number + "," + request);
							p.learn(proposal_number + "," + request);
						} catch (RemoteException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
					}
				}
			}
		}
	}


	private void scheduleTask(final Phase phase, final String value)
			throws InterruptedException, ExecutionException {
		String response = null;
		for (final IPaxos p : my_replicated_server) {
			@SuppressWarnings("unchecked")
			RunnableFuture f = new FutureTask(new Callable<String>(){
				// implement call
				public String call() throws RemoteException, InterruptedException {
					if (phase.equals(Phase.PREPARE)) {
						String prepReturn = p.prepare(proposal_number);
						System.out.println("We got "+ prepReturn+ "from acceptors");
						return prepReturn;
					}
					else{
						return p.accept(proposal_number, value);
					}
				}
			});
			// start the thread to execute it (you may also use an Executor)
			Thread t = new Thread(f);
			t.start();
			// get the result
			try {
				response = (String)f.get(3, TimeUnit.SECONDS);
				if(response != null){
					replies.add(response);
				}
			} catch (Exception e) {
//				e.printStackTrace();
//				e.getCause();
//				System.out.println("Timeout");
			}
			f.cancel(true);
		}
	}
}
