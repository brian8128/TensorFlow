# TensorFlow

## Running on EC2.  
Roughly following this blog
http://erikbern.com/2015/11/12/installing-tensorflow-on-aws/

1. Launch an EC2 instance.  Probably use the smaller of the ones with graphic cards.  Recommend to use spot instances since they're cheap.
2. Click the connect button in the EC2 console.  Use the ssh command they supply but change the user to 'ubuntu'.
3. Run the following commands.  Everything that you need to download takes a bit of time, so removing dependencies where possible is a good idea. 

```
$ sudo apt-get install python-sklearn python-pandas
$ git clone https://github.com/datamath28/neural-network.git
$ cd neural-network
$ python churn_nn.py
```

4. (Preliminary experiments suggest the EC2 is twice as fast as laptop.)
