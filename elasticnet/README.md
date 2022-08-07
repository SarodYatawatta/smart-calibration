### Distributed reinforcement learning with pytorch RPC

Here is an example on how to run ```distributed_per_sac.py``` on two computers. The learner and one actor is run on node with IP 192.168.178.101, network interface eth0:

```
export GLOO_SOCKET_IFNAME=eth0 && export TP_SOCKET_IFNAME=eth0 && python distributed_per_sac.py --world-size 3 --learner-addr 192.168.178.101 --learner-port 8080 --rank 0
```

and

```
export GLOO_SOCKET_IFNAME=eth0 && export TP_SOCKET_IFNAME=eth0 && python distributed_per_sac.py --world-size 3 --learner-addr 192.168.178.101 --learner-port 8080 --rank 1
```

Another actor is run on another node, with network interface eth1:

```
export GLOO_SOCKET_IFNAME=eth1 && export TP_SOCKET_IFNAME=eth1 && python distributed_per_sac.py --world-size 3 --learner-addr 192.168.178.101 --learner-port 8080 --rank 2
```

This script will use soft actor-critic algorithm with distributed prioritized experience replay to learn.
