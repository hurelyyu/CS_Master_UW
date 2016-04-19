
public class ClassCaseException extends Exception {
	public ClassCaseException(String message){
		//use super to initialize the exception's error message
		super(message);
	}
	public ClassCaseException(String message,Throwable throwable){
		super(message,throwable);
	}
}

