import java.util.NavigableSet;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ConcurrentSkipListMap;



public class LogProcessor implements Runnable {

	private ConcurrentSkipListMap<Float, String> my_log;
	
	private IKVProcessor my_kvs;

	private int earliestUnchosen;

	public LogProcessor(ConcurrentMap<Float, String> the_log, 
			final IKVProcessor the_kvs) {
		my_log = (ConcurrentSkipListMap<Float,String>)the_log;
		my_kvs = the_kvs;
		earliestUnchosen = 0;
	}

	@Override
	public void run() {

		String proposal;
		String[] tokens;
		String[] request_tokens;
		while(true) {
			//			for(int i = earliestUnchosen; i < my_log.size(); i++) {

			NavigableSet<Float> keys=(my_log.keySet());
//			System.out.println("my_log size: "+my_log.size());
			Object[] sortedKeys=keys.toArray();

//			System.out.println("sortedKey size: "+sortedKeys.length);
//			for(int i = 0; i < my_log.size(); i++){
			for(Float i:keys){
//				System.out.print("key: "+i);
				proposal = my_log.get(i);
//				System.out.println("my_log get("+i+")="+proposal);
				if(proposal != null) {
//					System.out.println("LogProcessor: proposal="+proposal);
					tokens = proposal.split(",");
					if(tokens[1].equals("[chosen]")) {
//						System.out.println("LogProcessor: tokens[1]==[chosen]");
						request_tokens = tokens[0].split("\\s+");
						if(request_tokens[0].equals("put")) {
//							System.out.println("LogProcessor: request_tokens[0]==put");
							my_kvs.kv_put(request_tokens[1], request_tokens[2]);
						} else if (request_tokens[0].equals("delete")) {
//							System.out.println("LogProcessor: request_tokens[0]==delete");
							my_kvs.kv_delete(request_tokens[1]);
						}
					} else {
//						earliestUnchosen = i;
//						break;
					}
				}
			}
		}
	}

}
