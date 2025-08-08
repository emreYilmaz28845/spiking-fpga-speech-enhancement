library IEEE;
use     IEEE.STD_LOGIC_1164.ALL;
use     IEEE.NUMERIC_STD.ALL;

entity snn_axi_stream_wrapper is
    Port (
        -- Global clock & reset (active-low, matches Zynq FCLK_RESETn)
        aclk            : in  std_logic;
        aresetn         : in  std_logic;

        -- AXI-Stream slave  (PS → PL)
        s_axis_tdata    : in  std_logic_vector(31 downto 0);
        s_axis_tvalid   : in  std_logic;
        s_axis_tready   : out std_logic;
        s_axis_tlast    : in  std_logic;

        -- AXI-Stream master (PL → PS)
        m_axis_tdata    : out std_logic_vector(31 downto 0);
        m_axis_tvalid   : out std_logic;
        m_axis_tready   : in  std_logic;
        m_axis_tlast    : out std_logic
    );
end snn_axi_stream_wrapper;

architecture RTL of snn_axi_stream_wrapper is

    constant WORDS_PER_FRAME : integer := 9;

    type word_array  is array (0 to WORDS_PER_FRAME-1)
                          of std_logic_vector(31 downto 0);

    type state_type  is (IDLE_RX, CAPTURE, WAIT_CORE, TX, GAP);

    -- handshake + data registers
    signal s_axis_tready_i : std_logic := '0';
    signal m_axis_tvalid_i : std_logic := '0';
    signal m_axis_tdata_i  : std_logic_vector(31 downto 0) := (others => '0');
    signal m_axis_tlast_i  : std_logic := '0';

    -- frame buffers
    signal rx_buf     : word_array := (others => (others => '0'));
    signal tx_buf     : word_array := (others => (others => '0'));

    signal rx_index   : natural range 0 to WORDS_PER_FRAME := 0;
    signal tx_index   : natural range 0 to WORDS_PER_FRAME := 0;

    signal state      : state_type := IDLE_RX;

    -- legacy core handshake (internal only)
    signal start_i, sample_ready_i : std_logic := '0';
    signal ready_i                 : std_logic;
    signal in_spikes_i, out_spikes_i : std_logic_vector(256 downto 0);

    -- helper signals
    signal frame_done : std_logic := '0';      -- asserted for 1 clk after word-8
    signal gap_cnt    : std_logic := '0';

begin

    s_axis_tready <= s_axis_tready_i;
    m_axis_tdata  <= m_axis_tdata_i;
    m_axis_tvalid <= m_axis_tvalid_i;
    m_axis_tlast  <= m_axis_tlast_i;

    network_inst : entity work.network
        port map (
            clk          => aclk,
            rst_n        => aresetn,
            start        => start_i,
            sample_ready => sample_ready_i,
            ready        => ready_i,
            sample       => open,
            in_spikes    => in_spikes_i,
            out_spikes   => out_spikes_i
        );

    process(aclk)
    begin
        if rising_edge(aclk) then
        
            -- asynchronous reset (active low)
            if aresetn = '0' then
                s_axis_tready_i <= '0';
                m_axis_tvalid_i <= '0';
                m_axis_tlast_i  <= '0';
                start_i         <= '0';
                sample_ready_i  <= '0';
                rx_index        <= 0;
                tx_index        <= 0;
                frame_done      <= '0';
                state           <= IDLE_RX;

            else

                -- defaults
                start_i         <= '0';
                sample_ready_i  <= '0';
                frame_done      <= '0';

                case state is

                -- IDLE_RX : wait for first beat of a new frame
                when IDLE_RX =>
                    s_axis_tready_i <= '1';
                    if s_axis_tvalid = '1' then
                        rx_buf(0) <= s_axis_tdata;
                        rx_index  <= 1;
                        state     <= CAPTURE;
                    end if;


                -- CAPTURE : store beats 1 … 8
                when CAPTURE =>
                    if s_axis_tvalid = '1' and s_axis_tready_i = '1' then
                        rx_buf(rx_index) <= s_axis_tdata;

                        if rx_index = WORDS_PER_FRAME-1 then     -- received word-8
                            s_axis_tready_i <= '0';              -- stop reception
                            frame_done      <= '1';              -- mark completion
                            state           <= WAIT_CORE;
                        else
                            rx_index <= rx_index + 1;
                        end if;
                    end if;


                -- WAIT_CORE : assert start/sample_ready until core ready
                when WAIT_CORE =>
                    if frame_done = '1' then
                        in_spikes_i <= rx_buf(8)(0) & rx_buf(7) & rx_buf(6) &
                                       rx_buf(5) & rx_buf(4) & rx_buf(3) &
                                       rx_buf(2) & rx_buf(1) & rx_buf(0);
                    end if;

                    start_i        <= '1';       
                    sample_ready_i <= '1';

                    if ready_i = '1' then
                        tx_buf(0) <= out_spikes_i( 31 downto   0);
                        tx_buf(1) <= out_spikes_i( 63 downto  32);
                        tx_buf(2) <= out_spikes_i( 95 downto  64);
                        tx_buf(3) <= out_spikes_i(127 downto  96);
                        tx_buf(4) <= out_spikes_i(159 downto 128);
                        tx_buf(5) <= out_spikes_i(191 downto 160);
                        tx_buf(6) <= out_spikes_i(223 downto 192);
                        tx_buf(7) <= out_spikes_i(255 downto 224);
                        tx_buf(8) <= (others => '0');
                        tx_buf(8)(0) <= out_spikes_i(256);

                        tx_index        <= 0;
                        m_axis_tdata_i  <= tx_buf(0);
                        m_axis_tvalid_i <= '1';
                        m_axis_tlast_i  <= '0';
                        state           <= TX;
                    end if;


                -- TX : stream 9 beats back to PS
                when TX =>
                    if m_axis_tvalid_i = '1' and m_axis_tready = '1' then
                        if tx_index = WORDS_PER_FRAME-1 then         
                            m_axis_tvalid_i <= '0';
                            m_axis_tlast_i  <= '0';                  
                            state           <= GAP;

                        else                                         
                            tx_index        <= tx_index + 1;
                            m_axis_tdata_i  <= tx_buf(tx_index + 1);

                            if tx_index = WORDS_PER_FRAME-2 then
                                m_axis_tlast_i <= '1';                
                            else
                                m_axis_tlast_i <= '0';
                            end if;
                        end if;
                    end if;


                -- GAP : two idle clocks between successive frames
                when GAP =>
                    s_axis_tready_i <= '0';          

                    if gap_cnt = '0' then
                        gap_cnt <= '1';             
                    else
                        gap_cnt        <= '0';       
                        s_axis_tready_i <= '1';      
                        rx_index        <= 0;
                        state           <= IDLE_RX;
                    end if;
                end case;
            end if;
        end if;
    end process;
end RTL;