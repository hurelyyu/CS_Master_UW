

public class ImplementHashmapYu implements HashmapYu{
		//an implementation of the HashMapYu interface
		

			public void put(Object key, Object value) {
				sets.put(key,value);
				System.out.println(key+", "+value+" sets is added");
			}
			
			public Object get(Object key) throws ClassCaseException {
				// TODO Auto-generated method stub
				if(!sets.containsKey(key))
					throw new ClassCaseException("Could not find key" + key + "from set");
				return sets.get(key);

			}

			
			public void delete(Object key) throws ClassCaseException {
				// TODO Auto-generated method stub
				if(!sets.containsKey(key)) {
					throw new ClassCaseException("Cannot find key" + key + "from set");}
				else {
					 sets.remove(key);
					System.out.println(key+" and its value is removed from set");
				}
			}
	
}

