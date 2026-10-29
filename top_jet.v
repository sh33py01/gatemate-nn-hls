// top_jet.v -- UART top for the jet tagger: 32 input bytes (16 x int16 raw ap_fixed<16,6>),
// 5 result words (int64 raw ap_fixed<38,18>). Same pins as top_uart (top_uart.ccf).
module top_jet(input wire clk, input wire rx, output wire tx, output wire led);
    top_uart #(.IN_BYTES(32), .OUT_WORDS(5)) u (.clk(clk), .rx(rx), .tx(tx), .led(led));
endmodule

// top_jetf -- same jet tagger from the SODA float32 flow: 64 input bytes (16 x float32),
// 3 result words (5 x float32 + 4 padding bytes).
module top_jetf(input wire clk, input wire rx, output wire tx, output wire led);
    top_uart #(.IN_BYTES(64), .OUT_WORDS(3)) u (.clk(clk), .rx(rx), .tx(tx), .led(led));
endmodule
