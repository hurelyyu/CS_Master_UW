import java.util.HashMap;
import java.util.Map;
/*
 * If an acceptor receives a prepare request with number n greater
 * than that of any prepare request to which it has already responded,
 * the it responds to the request with a promise not to accept any more
 * proposals numbered less than n and with the highest-numbered proposal
 * (if any) that it has accepted.
 */

public class Acceptor{
	private Map<Float, String> my_log;
	private float min_proposal_number;
	private String accepted_value;
	private float accepted_proposal_number;
	
	public Acceptor(final Map<Float, String> the_log){
		my_log = the_log;
		min_proposal_number = 0;
		accepted_value = "NO_ACCEPTED_VALUE_YET";
		accepted_proposal_number = 0;
		System.out.println("Initialized"); 
	}

	public String getAccepted_value() {
		return accepted_value;
	}

	public void setAccepted_value(String accepted_value) {
		this.accepted_value = accepted_value;
	}

	public float getAccepted_proposal_number() {
		return accepted_proposal_number;
	}

	public void setAccepted_proposal_number(float accepted_proposal_number) {
		this.accepted_proposal_number = accepted_proposal_number;
	}

	public synchronized String prepare(float n){
		System.out.println("Preparing "+ n);
		String v;
		if(n > min_proposal_number){
			min_proposal_number = n;
			v = accepted_proposal_number + "," + accepted_value;
		}
		else{
			v = min_proposal_number + "";
		}
		return v;
	}

	/*
	 * If an acceptor receives an accept request for a proposal numbered n, it
	 * accepts the proposal unless it has already responded to a prepare request
	 * have a number greater than n.
	 */
	public synchronized String accept(float n, String value){
		if(n >= min_proposal_number){
			min_proposal_number = n;
			accepted_proposal_number = n;
			accepted_value = value;
			my_log.put(n, value+",[accepted]");
		}

		return String.valueOf(min_proposal_number);
	}
	
	
}
