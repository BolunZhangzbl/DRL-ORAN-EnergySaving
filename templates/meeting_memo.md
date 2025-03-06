## Problems to be fixed

1. Use 5G TS 38.901 channel models (fast fading) + Path loss

   Reference: https://nvlabs.github.io/sionna/api/channel.wireless.html#gpp-38-901-channel-models

2. Revise 'get_sinr()' in oran.py

   **NOTE that a user's SINR should be computed as the SINR averaged over its allocated PRBs.**

   E.g., a user is allocated with RB 1-5, then we 1) compute interference powers separately for RB 1-5; 2) compute SINRs separately for RB 1-5; 3) get the average SINR for this user.

```python
power_interference_dbm = gNB.power_tx - path_loss
```

3. Revise PRB allocation

   Please **first read SPARC** to get a sense of interference formation and PRB allocation principles. May Find something useful in SPARC.

   The principles are: 

   * we must **specify a resource (PRB) reuse pattern** among multiple BSs, E.g, you can assume neighboring BSs must allocate orthogonal RBs to their users, but BSs at a distance can fully reuse all the PRBs.  Now in the current implementation, all the users have orthogonal PRBs, then there is no interference at all, so the interference computation now is **WRONG**. **Interference only occurs when two or multiple users reuse a PRB (in the same time & frequency block).**
   * Based on the BS PRB reuse pattern, PRBs are allocated (an algorithm is used). 
   * Baesd on PRB allocation, compute interference (related to get_sinr()): E.g., For PRB1 allocated to UE1, **first** figure out which BSs are using this PRB; **second**, get interfering channels between interferer BSs and target UE; **third**, get power_tx of interferer BSs (which should be the power allcoated to PRB1 only), and compute interference accordingly. Remember, there could be multiple interferers and interference can be superimposed. 

   

4. Revise transmission rates on each PRB based on MCSs (Pipeline SINR --> MCS index --> MCS + RB bandwidth --> rate)

   Reference: https://nvlabs.github.io/sionna/api/nr.html#id47

   *Table 1* *MCS Index Table 1 (Table 5.1.3.1-1 in [[3GPP38214\]](https://nvlabs.github.io/sionna/api/nr.html#gpp38214))*

   Sionna is an open-source implementation. We can use them if their codes are helpful.

5. Revise throughput computation given PRB allocation and transmission rates on each PRB

6. Debug the convergence of DQN: why loss and reward do not follow the same trend? why action is fixed at last?

7. Do visualization of BS deployment, user mobility, and activation states.

   * Develop render member function in env to visualize the BS, and UE info
   

## Questions:

1. Why '-30' when converting log10() to linear scale?

```python
       # Convert received power to linear scale watts
        power_rx = 10 ** ((power_rx_dbm - 30) / 10)
```



## Further work: 

1. Read two papers and understand their designs & implementation difficulties
2. Read ORAN-gym if there is spare time to see how their interfaces are defined.