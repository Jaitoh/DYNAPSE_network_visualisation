# DYNAPSE_network_visualisation

This function is built base on the output of `net_gen.print_network()` and conducting a further visualization of the network.

To use this function in your code, first import the functions in the network_visualisation.py with `from network_visualisation import *`

In your code, you can simply 
- **replace** `net_gen.print_network()`  with `draw_network(net_gen, title="Network", store_path = "./")`
- or **add** `draw_network(net_gen, title="Network", store_path = "./")` following `net_gen.print_network()` 

The `store_path` is where you store the network structure, and the `title` is the title you assigned to the stored picture.