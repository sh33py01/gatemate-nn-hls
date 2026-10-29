`timescale 1ns/1ps
// Wrapper-level simulation: nn_core (OUT_WORDS=5) + real Bambu jet kernel (renamed forward_kernel).
// Inputs/expected outputs from jet_tb_data.py (hls4ml bit-accurate emulation).
module tb;
  parameter NJ = 50;
  reg clk = 0, rst_n = 0, start = 0, host_we = 0;
  reg [9:0] host_addr = 0; reg [63:0] host_wdata = 0;
  wire busy, done, bad; wire [31:0] cycles; wire [63:0] res;
  always #5 clk = ~clk;
  nn_core #(.OUT_WORDS(5)) dut(.clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done),
    .cycles(cycles), .bad_addr(bad), .host_we(host_we), .host_addr(host_addr),
    .host_wdata(host_wdata), .host_rdata(res));
  reg [63:0] xin [0:4*NJ-1];
  reg [63:0] yexp [0:5*NJ-1];
  integer j, i, n, errors = 0, cyc = 0;
  initial begin
    $readmemh("jet_in.hex", xin);
    $readmemh("jet_out.hex", yexp);
    repeat (10) @(posedge clk); rst_n = 1; repeat (2) @(posedge clk);
    for (j = 0; j < NJ; j = j + 1) begin
      for (i = 0; i < 4; i = i + 1) begin
        @(negedge clk); host_we = 1; host_addr = i; host_wdata = xin[4*j+i];
      end
      @(negedge clk); host_we = 0; start = 1; @(negedge clk); start = 0;
      n = 0; while (!done && n < 1000000) begin @(posedge clk); n = n + 1; end
      @(posedge clk);
      if (n >= 1000000) begin $display("jet %0d TIMEOUT", j); errors = errors + 1; end
      cyc = cycles;
      for (i = 0; i < 5; i = i + 1) begin
        @(negedge clk); host_addr = i; #1;
        if (res !== yexp[5*j+i]) begin
          errors = errors + 1;
          if (errors < 10) $display("jet %0d out%0d: got %h expected %h", j, i, res, yexp[5*j+i]);
        end
      end
      if (bad) begin $display("jet %0d: bad_addr", j); errors = errors + 1; end
    end
    $display("%0d jets, cycles=%0d, errors=%0d", NJ, cyc, errors);
    if (errors) $display("OVERALL: FAIL"); else $display("OVERALL: PASS");
    $finish;
  end
endmodule
