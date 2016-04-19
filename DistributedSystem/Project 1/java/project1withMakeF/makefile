JFLAGS = -g
JC = javac
.SUFFIXES: .java .class
.java.class:
	$(JC) $(JFLAGS) $*.java

CLASSES = \
	CustomException.java \
	HashMapKV.java \
	PlainHashMapKV.java \
	TCPClientHM.java \
	TCPServerHM.java \
	UDPClientHM.java \
	UDPServerHM.java \
	UpdateClientLog.java 

default: classes

classes: $(CLASSES:.java=.class)

clean:
	$(RM) *.class
