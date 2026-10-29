# Board results

Per-inference CSVs written by the board test scripts and the nextpnr utilisation reports.

| File | Content |
|---|---|
| `mynet_mnist_board.csv` | MyNet on 500 MNIST test images (`gatemate_inference.py --dataset`) |
| `jet_fixed_M1.csv`, `jet_fixed_M16.csv` | Jet tagger fixed point, 1 and 16 multipliers (`jet_inference.py`) |
| `jet_float32.csv` | Jet tagger float32 from the SODA flow (`jetf_inference.py`) |
| `util_*.txt` | nextpnr "Device utilisation" and max-frequency lines per design |
