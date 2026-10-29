// top_uart.v -- GateMate CCGM1A1 eval board: UART front-end for nn_core.
//
// Protocol (57600 8N1 at 10 MHz):
//   PC -> FPGA : IN_BYTES bytes, little-endian (MyNet: 1x1x32x32 float32, row-major).
//                If the line goes quiet for ~0.1 s mid-frame, the byte count resets.
//   FPGA       : runs the kernel as soon as the last byte arrives.
//   FPGA -> PC : AA 55 | OUT_WORDS*8 result bytes (LE) | cycles (4, LE u32) | flags (1)
//                (MyNet: 15 bytes, result = out0 f32 | out1 f32)
//                flags bit0 = bad_addr (kernel touched memory outside the window)
//   Then waits for the next input. D1 toggles after each inference.
//
// Pins (see top_uart.ccf): clk IO_SB_A8, tx IO_NB_A1 (grey, adapter RX),
//                          rx IO_NB_A2 (orange, adapter TX), led IO_EB_B1
module top_uart #(
    parameter BAUD_DIV  = 174,           // 10 MHz / 57600
    parameter QUIET     = 1_000_000,     // 0.1 s at 10 MHz
    parameter IN_BYTES  = 4096,          // MyNet 4096 (1x1x32x32 f32), jet tagger 32 (16 x int16)
    parameter OUT_WORDS = 1              // 64-bit result words: MyNet 1, jet tagger 5
)(
    input  wire clk,
    input  wire rx,
    output reg  tx  = 1'b1,
    output reg  led = 1'b1
);
    // ---------------- power-on reset (active low) ----------------
    reg [15:0] por = 0;
    wire rst_n = &por;
    always @(posedge clk) if (!rst_n) por <= por + 1;

    // ---------------- UART RX ----------------
    reg rx_s0 = 1, rx_s1 = 1;
    reg [1:0] rx_st = 0; reg [7:0] rx_cnt = 0; reg [2:0] rx_bit = 0;
    reg [7:0] rx_sh = 0, rx_data = 0; reg rx_valid = 0;
    always @(posedge clk) begin
        rx_s0 <= rx; rx_s1 <= rx_s0; rx_valid <= 0;
        case (rx_st)
            0: if (!rx_s1) begin rx_cnt <= BAUD_DIV/2; rx_st <= 1; end
            1: if (rx_cnt == 0) begin
                   if (rx_s1) rx_st <= 0;               // glitch, not a start bit
                   else begin rx_cnt <= BAUD_DIV-1; rx_bit <= 0; rx_st <= 2; end
               end else rx_cnt <= rx_cnt - 1;
            2: if (rx_cnt == 0) begin
                   rx_sh <= {rx_s1, rx_sh[7:1]}; rx_cnt <= BAUD_DIV-1;
                   if (rx_bit == 7) rx_st <= 3; else rx_bit <= rx_bit + 1;
               end else rx_cnt <= rx_cnt - 1;
            3: if (rx_cnt == 0) begin                   // middle of stop bit
                   rx_data <= rx_sh; rx_valid <= rx_s1; rx_st <= 0;
               end else rx_cnt <= rx_cnt - 1;
        endcase
    end

    // ---------------- UART TX ----------------
    reg [1:0] tx_st = 0; reg [7:0] tx_cnt = 0; reg [2:0] tx_bit = 0;
    reg [7:0] tx_sh = 0, tx_byte = 0; reg tx_go = 0;
    wire tx_busy = (tx_st != 0);
    always @(posedge clk) begin
        case (tx_st)
            0: begin tx <= 1; if (tx_go) begin tx_sh <= tx_byte; tx_cnt <= BAUD_DIV-1; tx_st <= 1; end end
            1: begin tx <= 0;
                 if (tx_cnt == 0) begin tx_cnt <= BAUD_DIV-1; tx_bit <= 0; tx_st <= 2; end
                 else tx_cnt <= tx_cnt - 1; end
            2: begin tx <= tx_sh[0];
                 if (tx_cnt == 0) begin tx_sh <= {1'b1, tx_sh[7:1]}; tx_cnt <= BAUD_DIV-1;
                     if (tx_bit == 7) tx_st <= 3; else tx_bit <= tx_bit + 1; end
                 else tx_cnt <= tx_cnt - 1; end
            3: begin tx <= 1;
                 if (tx_cnt == 0) tx_st <= 0; else tx_cnt <= tx_cnt - 1; end
        endcase
    end

    // ---------------- accelerator ----------------
    reg        k_start = 0, h_we = 0;
    reg  [9:0] h_addr = 0;
    reg [63:0] h_wdata = 0;
    wire       k_busy, k_done, k_bad;
    wire [31:0] k_cycles;
    wire [63:0] k_result;
    nn_core #(.OUT_WORDS(OUT_WORDS)) core (.clk(clk), .rst_n(rst_n), .start(k_start), .busy(k_busy), .done(k_done),
                  .cycles(k_cycles), .bad_addr(k_bad), .host_we(h_we), .host_addr(h_addr),
                  .host_wdata(h_wdata), .host_rdata(k_result));

    // ---------------- main FSM ----------------
    localparam S_RECV = 0, S_RUN = 1, S_SEND = 2;
    localparam NFRAME = 2 + 8 * OUT_WORDS + 5;
    reg [1:0]  st = S_RECV;
    reg [12:0] nbytes = 0;
    reg [20:0] quiet = 0;
    reg [15:0] tx_idx = 0;           // byte index within the reply frame
    reg [31:0] cyc_l = 0;
    reg        bad_l = 0;
    // byte tx_idx of the reply frame (result words read combinationally from nn_core)
    wire [15:0] rb = tx_idx - 16'd2;
    wire [15:0] tb = tx_idx - 16'd2 - 8 * OUT_WORDS;
    wire [7:0] frame_byte =
        (tx_idx == 0) ? 8'hAA :
        (tx_idx == 1) ? 8'h55 :
        (tx_idx < 2 + 8 * OUT_WORDS) ? k_result[8 * rb[2:0] +: 8] :
        (tb < 4) ? cyc_l[8 * tb[1:0] +: 8] : {7'd0, bad_l};

    always @(posedge clk) begin
        h_we <= 0; k_start <= 0; tx_go <= 0;
        if (!rst_n) begin
            st <= S_RECV; nbytes <= 0; quiet <= 0; h_addr <= 0;
        end else case (st)
            S_RECV: begin
                if (rx_valid) begin
                    quiet <= 0;
                    h_wdata <= {rx_data, h_wdata[63:8]};          // little-endian packing
                    nbytes  <= nbytes + 1;
                    if (nbytes[2:0] == 3'd7) begin
                        h_we <= 1; h_addr <= {1'b0, nbytes[11:3]};
                    end
                    if (nbytes == IN_BYTES - 1) begin nbytes <= 0; st <= S_RUN; k_start <= 1; end
                end else if (nbytes != 0) begin
                    if (quiet == QUIET) begin nbytes <= 0; quiet <= 0; end
                    else quiet <= quiet + 1;
                end
            end
            S_RUN: if (k_done) begin
                cyc_l <= k_cycles; bad_l <= k_bad;
                tx_idx <= 0; h_addr <= 0; led <= ~led; st <= S_SEND;
            end
            S_SEND: if (!tx_busy && !tx_go) begin
                if (tx_idx == NFRAME) st <= S_RECV;
                else begin
                    tx_byte <= frame_byte; tx_go <= 1;
                    tx_idx  <= tx_idx + 1;
                    // select the result word holding the NEXT byte
                    h_addr  <= (tx_idx + 1 >= 2 && tx_idx + 1 < 2 + 8 * OUT_WORDS) ? ((tx_idx - 1) >> 3) : 10'd0;
                end
            end
        endcase
    end
endmodule
