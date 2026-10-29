// nn_core.v
// Bambu forward_kernel (bare-ptr, ALL_BRAM build) + external memory for P0/P1.
// CH=1 for --channels-type=MEM_ACC_11 kernels (64-bit bus), CH=2 for MEM_ACC_NN (128-bit bus).
//
// Bus protocol, copied from Bambu's own testbench memory model (bambu_testbench.v):
//   * 2 channels, each: oe/we, 32-bit byte address, 7-bit access size in BITS,
//     64-bit data. Channel i uses slice [32*i +: 32] / [64*i +: 64] / [7*i +: 7].
//   * Every request (read OR write) must be answered with M_DataRdy[i].
//   * Read data is the 64 bits starting AT the byte address (low byte = byte at addr),
//     i.e. shifted down, not left in place within the 64-bit word.
//   * Model only serves addresses >= base_addr (0x4000_0000), so P0/P1 live there.
//   * Reset is ACTIVE LOW.
// Read latency: data+ready one cycle after the request (matches MEM_DELAY_READ=2).
// Write ack: registered, one cycle after the request (WACK_COMB=1 gives a same-cycle
//   ack like the testbench's negedge model; try it only if the default deadlocks).

module nn_core #(
    parameter [31:0] BASE      = 32'h4000_0000,
    parameter [31:0] OUT_OFF   = 32'h0000_1000,   // output buffer offset (8-aligned)
    parameter        WACK_COMB = 0,
    parameter        CH        = 2,     // Bambu memory channels: 1 (MEM_ACC_11) or 2 (MEM_ACC_NN)
    parameter        OUT_WORDS = 1      // output buffer size in 64-bit words (MyNet 1, jet tagger 5)
)(
    input             clk,
    input             rst_n,          // active low
    // control
    input             start,          // 1-cycle pulse, ignored while busy
    output reg        busy,
    output reg        done,           // 1-cycle pulse when kernel finishes
    output reg [31:0] cycles,         // start->done latency of the last run
    output reg        bad_addr,       // sticky: kernel accessed outside the window
    // host port (writes only while !busy): words 0..511 = input;
    // host_rdata = output word host_addr (0..OUT_WORDS-1), combinational
    input             host_we,
    input      [9:0]  host_addr,
    input      [63:0] host_wdata,
    output     [63:0] host_rdata
);
    localparam [31:0] P0 = BASE;
    localparam [31:0] P1 = BASE + OUT_OFF;

    // ---------------- kernel ----------------
    wire         k_done;
    reg          k_start;
    wire [CH-1:0]    k_oe, k_we, k_rdy;
    wire [32*CH-1:0] k_addr;
    wire [64*CH-1:0] k_wdata, k_rdata;
    wire [7*CH-1:0]  k_size;
    // internal view is always 2 channels; channel 1 is tied off when CH == 1
    wire [1:0]   oe, we;
    wire [63:0]  addr;
    wire [127:0] wdata;
    wire [13:0]  size;
    wire [127:0] rdata;
    wire [1:0]   rdy;
    generate if (CH == 2) begin : g2
        assign oe = k_oe; assign we = k_we; assign addr = k_addr;
        assign wdata = k_wdata; assign size = k_size;
        assign k_rdata = rdata; assign k_rdy = rdy;
    end else begin : g1
        assign oe = {1'b0, k_oe}; assign we = {1'b0, k_we}; assign addr = {32'd0, k_addr};
        assign wdata = {64'd0, k_wdata}; assign size = {7'd0, k_size};
        assign k_rdata = rdata[63:0]; assign k_rdy = rdy[0];
    end endgenerate

    forward_kernel fk (
        .clock(clk), .reset(rst_n), .start_port(k_start), .done_port(k_done),
        .P0(P0), .P1(P1),
        .M_Rdata_ram(k_rdata), .M_DataRdy(k_rdy),
        .Mout_oe_ram(k_oe), .Mout_we_ram(k_we), .Mout_addr_ram(k_addr),
        .Mout_Wdata_ram(k_wdata), .Mout_data_ram_size(k_size)
    );

    // ---------------- memory ----------------
    // Input: 4 KB BRAM, 512 x 64 (only the kernel reads it; host writes it while idle).
    // Output: 8 bytes in registers, not BRAM, so the two channels may write both
    // floats in the same cycle without a dual-port write collision.
    reg [63:0] mem [0:511];
    reg [63:0] result [0:OUT_WORDS-1];

    wire [31:0] off0 = addr[31:0]  - BASE;
    wire [31:0] off1 = addr[63:32] - BASE;
    wire [8:0]  ones0 = (9'd1 << size[6:3])   - 9'd1;   // size is in bits
    wire [8:0]  ones1 = (9'd1 << size[13:10]) - 9'd1;
    wire [15:0] be0w = {7'd0, ones0} << off0[2:0];
    wire [15:0] be1w = {7'd0, ones1} << off1[2:0];
    wire [63:0] wd0  = wdata[63:0]   << {off0[2:0], 3'b000};
    wire [63:0] wd1  = wdata[127:64] << {off1[2:0], 3'b000};
    wire        req0 = busy & (oe[0] | we[0]);
    wire        req1 = busy & (oe[1] | we[1]);
    wire        in0  = off0 < 32'd4096;
    wire        in1  = off1 < 32'd4096;
    wire        out0 = (off0 >= OUT_OFF) && (off0 < OUT_OFF + 8 * OUT_WORDS);
    wire        out1 = (off1 >= OUT_OFF) && (off1 < OUT_OFF + 8 * OUT_WORDS);
    wire [31:0] ow0  = (off0 - OUT_OFF) >> 3;      // output word index
    wire [31:0] ow1  = (off1 - OUT_OFF) >> 3;
    wire        oob0 = req0 & ((!in0 & !out0) | (be0w[15:8] != 0) | (in0 & we[0]));
    wire        oob1 = req1 & ((!in1 & !out1) | (be1w[15:8] != 0) | (in1 & we[1]));

    // port A: kernel ch0 reads while busy, host writes while idle
    wire        a_rd   = req0 & oe[0] & in0;
    wire        a_wr   = !busy & host_we & !host_addr[9];
    wire [8:0]  a_addr = busy ? off0[11:3] : host_addr[8:0];
    // port B: kernel ch1 reads
    wire        b_rd   = req1 & oe[1] & in1;
    wire [8:0]  b_addr = off1[11:3];

    reg [63:0] a_q, b_q;
    always @(posedge clk) begin
        if (a_wr)      mem[a_addr] <= host_wdata;
        else if (a_rd) a_q <= mem[a_addr];
    end
    always @(posedge clk)
        if (b_rd) b_q <= mem[b_addr];

    // output registers: byte-lane writes from either channel
    integer k;
    always @(posedge clk) begin
        for (k = 0; k < 8; k = k + 1) begin
            if (req0 & we[0] & out0 & be0w[k]) result[ow0][8*k +: 8] <= wd0[8*k +: 8];
            if (req1 & we[1] & out1 & be1w[k]) result[ow1][8*k +: 8] <= wd1[8*k +: 8];
        end
        if (!busy && host_we && host_addr == 10'd512) result[0] <= host_wdata; // test hook
    end
    assign host_rdata = (host_addr < OUT_WORDS) ? result[host_addr] : result[0];

    // ---------------- responses ----------------
    reg [1:0] rd_q, wa_q, ro_q;
    reg [31:0] ow0_q, ow1_q;
    reg [2:0] sh0_q, sh1_q;
    always @(posedge clk) begin
        if (!rst_n) begin
            rd_q <= 0; wa_q <= 0; ro_q <= 0;
        end else begin
            rd_q <= {req1 & oe[1], req0 & oe[0]};
            ro_q <= {out1, out0};
            ow0_q <= ow0; ow1_q <= ow1;
            wa_q <= WACK_COMB ? 2'b00 : {req1 & we[1], req0 & we[0]};
        end
        sh0_q <= off0[2:0];
        sh1_q <= off1[2:0];
    end
    wire [63:0] r0 = ro_q[0] ? result[ow0_q] : a_q;
    wire [63:0] r1 = ro_q[1] ? result[ow1_q] : b_q;
    assign rdata = { r1 >> {sh1_q, 3'b000}, r0 >> {sh0_q, 3'b000} };
    assign rdy   = rd_q | wa_q | (WACK_COMB ? {req1 & we[1], req0 & we[0]} : 2'b00);

    // ---------------- control ----------------
    always @(posedge clk) begin
        k_start <= 0;
        done    <= 0;
        if (!rst_n) begin
            busy <= 0; cycles <= 0; bad_addr <= 0;
        end else begin
            if (busy && (oob0 || oob1)) bad_addr <= 1;
            if (!busy && start) begin
                busy <= 1; k_start <= 1; cycles <= 0;
            end else if (busy) begin
                cycles <= cycles + 1;
                if (k_done) begin busy <= 0; done <= 1; end
            end
        end
    end
endmodule
