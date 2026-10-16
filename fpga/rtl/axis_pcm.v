`timescale 1ns / 1ps

module axis_pcm #(
    parameter CHANNELS = 14,
    parameter HOP = 512,
    parameter SAMPLE_RATE = 48000
) (
    input wire clk,
    input wire resetn,
    input wire enable,
    input wire [16*CHANNELS-1:0] samples,
    input wire valid,
    input wire clipped,
    output reg [31:0] m_axis_tdata,
    output wire [3:0] m_axis_tkeep,
    output wire m_axis_tvalid,
    input wire m_axis_tready,
    output wire m_axis_tlast,
    output reg [31:0] dropped
);
    localparam IDLE = 0, HEADER = 1, READ = 2, PAYLOAD = 3;
    (* ram_style = "block" *) reg [16*CHANNELS-1:0] memory [0:2*HOP-1];
    reg [16*CHANNELS-1:0] read_frame;
    reg [1:0] ready;
    reg write_bank, read_bank;
    reg writing;
    integer write_frame, frame_index, channel_index;
    reg [63:0] sample_count;
    reg [63:0] first_sample [0:1];
    reg [31:0] loss_count [0:1];
    reg clip_seen;
    reg [31:0] packet_flags;
    reg [63:0] packet_first;
    reg [31:0] packet_loss;
    reg [1:0] state;
    reg [2:0] header_index;
    wire last_sample = frame_index == HOP-1 && channel_index == CHANNELS-1;
    wire write_enable = valid && ((write_frame == 0 && enable && !ready[write_bank]) ||
                                  (write_frame != 0 && writing));
    wire signed [15:0] pcm = read_frame[16*channel_index +: 16];

    assign m_axis_tkeep = 4'hf;
    assign m_axis_tvalid = state == HEADER || state == PAYLOAD;
    assign m_axis_tlast = state == PAYLOAD && last_sample;

    always @* begin
        m_axis_tdata = 0;
        if (state == PAYLOAD)
            m_axis_tdata = {{16{pcm[15]}}, pcm};
        else if (state == HEADER) begin
            case (header_index)
                0: m_axis_tdata = 32'h314d4450;
                1: m_axis_tdata = 32'h00010800 | CHANNELS;
                2: m_axis_tdata = packet_first[31:0];
                3: m_axis_tdata = packet_first[63:32];
                4: m_axis_tdata = packet_loss;
                5: m_axis_tdata = HOP;
                6: m_axis_tdata = SAMPLE_RATE;
                7: m_axis_tdata = packet_flags;
            endcase
        end
    end

    always @(posedge clk) begin
        if (resetn && write_enable)
            memory[write_bank*HOP + write_frame] <= samples;
        if (resetn && state == READ)
            read_frame <= memory[read_bank*HOP + frame_index];
    end

    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            ready <= 0;
            write_bank <= 0;
            read_bank <= 0;
            writing <= 0;
            write_frame <= 0;
            frame_index <= 0;
            channel_index <= 0;
            sample_count <= 0;
            first_sample[0] <= 0; first_sample[1] <= 0;
            loss_count[0] <= 0; loss_count[1] <= 0;
            dropped <= 0;
            clip_seen <= 0;
            packet_flags <= 0;
            packet_first <= 0;
            packet_loss <= 0;
            state <= IDLE;
            header_index <= 0;
        end else begin
            if (clipped)
                clip_seen <= 1;
            if (valid) begin
                sample_count <= sample_count + 1'b1;
                if (write_frame == 0) begin
                    writing <= enable && !ready[write_bank];
                    if (enable && !ready[write_bank]) begin
                        first_sample[write_bank] <= sample_count;
                        loss_count[write_bank] <= dropped;
                    end else if (enable) begin
                        dropped <= dropped + 1'b1;
                    end
                end
                if (write_frame == HOP-1) begin
                    write_frame <= 0;
                    if (writing) begin
                        ready[write_bank] <= 1;
                        write_bank <= !write_bank;
                    end
                    writing <= 0;
                end else begin
                    write_frame <= write_frame + 1;
                end
            end
            case (state)
                IDLE: if (ready[read_bank]) begin
                    packet_first <= first_sample[read_bank];
                    packet_loss <= loss_count[read_bank];
                    packet_flags <= {31'd0, clip_seen};
                    header_index <= 0;
                    state <= HEADER;
                end
                HEADER: if (m_axis_tready) begin
                    if (header_index == 7) begin
                        frame_index <= 0;
                        channel_index <= 0;
                        state <= READ;
                    end else begin
                        header_index <= header_index + 1'b1;
                    end
                end
                READ: state <= PAYLOAD;
                PAYLOAD: if (m_axis_tready) begin
                    if (channel_index == CHANNELS-1) begin
                        channel_index <= 0;
                        if (frame_index == HOP-1) begin
                            ready[read_bank] <= 0;
                            read_bank <= !read_bank;
                            state <= IDLE;
                        end else begin
                            frame_index <= frame_index + 1;
                            state <= READ;
                        end
                    end else begin
                        channel_index <= channel_index + 1;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
