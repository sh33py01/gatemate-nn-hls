`timescale 1ns/1ps
// Wrapper-level simulation of nn_core with the REAL Bambu forward_kernel.
// Run from output/sim_noheap so the kernel's $readmem .mem files resolve.
// Expected (float32 reference from 01_tosa.mlir weights):
//   ones : 0x3F08D68E 0x3EEE52E4 (Bambu RTL gave ...52E3, 1 ulp)
//   zeros: 0x3EF695FA 0x3F04B503
module tb;
`ifndef WACK
`define WACK 0
`endif
  reg clk = 0, rst_n = 0, start = 0, host_we = 0;
  reg [9:0] host_addr = 0; reg [63:0] host_wdata = 0;
  wire busy, done, bad; wire [31:0] cycles; wire [63:0] res;
  always #5 clk = ~clk;

  nn_core #(.WACK_COMB(`WACK)) dut(.clk(clk), .rst_n(rst_n), .start(start), .busy(busy),
    .done(done), .cycles(cycles), .bad_addr(bad), .host_we(host_we), .host_addr(host_addr),
    .host_wdata(host_wdata), .host_rdata(res));

  integer i, n, errors = 0;

  function integer near;   // |a-b| <= 8 ulps (same sign, positive floats)
    input [31:0] a, b;
    begin near = (a > b ? a - b : b - a) <= 8; end
  endfunction

  task run;
    input [63:0] fill; input [31:0] e0, e1; input [8*8-1:0] name;
    begin
      for (i = 0; i < 512; i = i + 1) begin
        @(negedge clk); host_we = 1; host_addr = i; host_wdata = fill;
      end
      @(negedge clk); host_addr = 512; host_wdata = 64'hDEADBEEF_DEADBEEF; // poison output
      @(negedge clk); host_we = 0; start = 1;
      @(negedge clk); start = 0;
      n = 0;
      while (!done && n < 2000000) begin @(posedge clk); n = n + 1; end
      @(posedge clk);
      if (!done && n >= 2000000) begin
        $display("[%0s] TIMEOUT: kernel never finished (busy=%0d bad_addr=%0d)", name, busy, bad);
        errors = errors + 1;
      end else begin
        $display("[%0s] out0=%h out1=%h cycles=%0d bad_addr=%0d", name, res[31:0], res[63:32], cycles, bad);
        if (near(res[31:0], e0) && near(res[63:32], e1) && !bad) $display("[%0s] PASS", name);
        else begin $display("[%0s] FAIL (expected %h %h)", name, e0, e1); errors = errors + 1; end
      end
    end
  endtask

  // expected outputs: defaults = untrained weights; override for other weights with
  // +e_ones0=<hex> +e_ones1=<hex> +e_zeros0=<hex> +e_zeros1=<hex>
  reg [31:0] eo0 = 32'h3F08D68E, eo1 = 32'h3EEE52E4, ez0 = 32'h3EF695FA, ez1 = 32'h3F04B503;
  initial begin
    if ($value$plusargs("e_ones0=%h", eo0)) ;
    if ($value$plusargs("e_ones1=%h", eo1)) ;
    if ($value$plusargs("e_zeros0=%h", ez0)) ;
    if ($value$plusargs("e_zeros1=%h", ez1)) ;
    repeat (10) @(posedge clk); rst_n = 1; repeat (2) @(posedge clk);
    run(64'h3F800000_3F800000, eo0, eo1, "ones");
    run(64'h00000000_00000000, ez0, ez1, "zeros");
    if (errors) $display("OVERALL: FAIL"); else $display("OVERALL: PASS");
    $finish;
  end
endmodule
