all: src/**/*.java
	mkdir -p bin
	javac -d bin -cp src/cnn:bin/ src/**/*.java

test:
	java -cp bin cnn.driver.Main

clean:
	rm -rf bin