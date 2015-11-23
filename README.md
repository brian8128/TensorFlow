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

4. (Preliminary experiment suggest the EC2 is twice as fast as laptop.  When we have 128 observations per batch this increases to 3x faster.  With 256 observations per batch the EC2 is 5-6x faster.  9x faster with 512 observations per batch.  15x faster with 1024 observations per batch, 16x faster with 2048 observations per batch.)
5. To see GPU utilization: (In our case it looks like it's using 80% of the (single) CPU and 90% of the GPU.)
```
nvidia-smi
```
