// jet_shim.v -- adapts Bambu's jet_top (ports named after the C arguments in_raw / out_raw)
// to the forward_kernel interface nn_core expects (P0 / P1). Same 2-channel memory bus.
module forward_kernel(clock, reset, start_port, done_port, P0, P1,
  M_Rdata_ram, M_DataRdy, Mout_oe_ram, Mout_we_ram, Mout_addr_ram, Mout_Wdata_ram, Mout_data_ram_size);
  input clock, reset, start_port;
  output done_port;
  input [31:0] P0, P1;
  input [127:0] M_Rdata_ram;
  input [1:0] M_DataRdy;
  output [1:0] Mout_oe_ram, Mout_we_ram;
  output [63:0] Mout_addr_ram;
  output [127:0] Mout_Wdata_ram;
  output [13:0] Mout_data_ram_size;
  jet_top k (.clock(clock), .reset(reset), .start_port(start_port), .done_port(done_port),
    .in_raw(P0), .out_raw(P1), .M_Rdata_ram(M_Rdata_ram), .M_DataRdy(M_DataRdy),
    .Mout_oe_ram(Mout_oe_ram), .Mout_we_ram(Mout_we_ram), .Mout_addr_ram(Mout_addr_ram),
    .Mout_Wdata_ram(Mout_Wdata_ram), .Mout_data_ram_size(Mout_data_ram_size));
endmodule
