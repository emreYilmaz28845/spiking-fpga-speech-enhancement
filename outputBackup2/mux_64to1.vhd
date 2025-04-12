---------------------------------------------------------------------------------
-- This is free and unencumbered software released into the public domain.
--
-- Anyone is free to copy, modify, publish, use, compile, sell, or
-- distribute this software, either in source code form or as a compiled
-- binary, for any purpose, commercial or non-commercial, and by any
-- means.
--
-- In jurisdictions that recognize copyright laws, the author or authors
-- of this software dedicate any and all copyright interest in the
-- software to the public domain. We make this dedication for the benefit
-- of the public at large and to the detriment of our heirs and
-- successors. We intend this dedication to be an overt act of
-- relinquishment in perpetuity of all present and future rights to this
-- software under copyright law.
--
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
-- EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
-- MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
-- IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
-- OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
-- ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
-- OTHER DEALINGS IN THE SOFTWARE.
--
-- For more information, please refer to <http://unlicense.org/>
---------------------------------------------------------------------------------


library ieee;
use ieee.std_logic_1164.all;


entity mux_64to1 is
    port (
        mux_sel : in std_logic_vector(5 downto 0);
        in0 : in std_logic;
        in1 : in std_logic;
        in2 : in std_logic;
        in3 : in std_logic;
        in4 : in std_logic;
        in5 : in std_logic;
        in6 : in std_logic;
        in7 : in std_logic;
        in8 : in std_logic;
        in9 : in std_logic;
        in10 : in std_logic;
        in11 : in std_logic;
        in12 : in std_logic;
        in13 : in std_logic;
        in14 : in std_logic;
        in15 : in std_logic;
        in16 : in std_logic;
        in17 : in std_logic;
        in18 : in std_logic;
        in19 : in std_logic;
        in20 : in std_logic;
        in21 : in std_logic;
        in22 : in std_logic;
        in23 : in std_logic;
        in24 : in std_logic;
        in25 : in std_logic;
        in26 : in std_logic;
        in27 : in std_logic;
        in28 : in std_logic;
        in29 : in std_logic;
        in30 : in std_logic;
        in31 : in std_logic;
        in32 : in std_logic;
        in33 : in std_logic;
        in34 : in std_logic;
        in35 : in std_logic;
        in36 : in std_logic;
        in37 : in std_logic;
        in38 : in std_logic;
        in39 : in std_logic;
        in40 : in std_logic;
        in41 : in std_logic;
        in42 : in std_logic;
        in43 : in std_logic;
        in44 : in std_logic;
        in45 : in std_logic;
        in46 : in std_logic;
        in47 : in std_logic;
        in48 : in std_logic;
        in49 : in std_logic;
        in50 : in std_logic;
        in51 : in std_logic;
        in52 : in std_logic;
        in53 : in std_logic;
        in54 : in std_logic;
        in55 : in std_logic;
        in56 : in std_logic;
        in57 : in std_logic;
        in58 : in std_logic;
        in59 : in std_logic;
        in60 : in std_logic;
        in61 : in std_logic;
        in62 : in std_logic;
        in63 : in std_logic;
        mux_out : out std_logic
    );
end entity mux_64to1;

architecture behavior of mux_64to1 is


begin

    selection : process(mux_sel, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15, in16, in17, in18, in19, in20, in21, in22, in23, in24, in25, in26, in27, in28, in29, in30, in31, in32, in33, in34, in35, in36, in37, in38, in39, in40, in41, in42, in43, in44, in45, in46, in47, in48, in49, in50, in51, in52, in53, in54, in55, in56, in57, in58, in59, in60, in61, in62, in63 )
    begin

        case mux_sel is

            when "000000" =>
                mux_out <= in0;


            when "000001" =>
                mux_out <= in1;


            when "000010" =>
                mux_out <= in2;


            when "000011" =>
                mux_out <= in3;


            when "000100" =>
                mux_out <= in4;


            when "000101" =>
                mux_out <= in5;


            when "000110" =>
                mux_out <= in6;


            when "000111" =>
                mux_out <= in7;


            when "001000" =>
                mux_out <= in8;


            when "001001" =>
                mux_out <= in9;


            when "001010" =>
                mux_out <= in10;


            when "001011" =>
                mux_out <= in11;


            when "001100" =>
                mux_out <= in12;


            when "001101" =>
                mux_out <= in13;


            when "001110" =>
                mux_out <= in14;


            when "001111" =>
                mux_out <= in15;


            when "010000" =>
                mux_out <= in16;


            when "010001" =>
                mux_out <= in17;


            when "010010" =>
                mux_out <= in18;


            when "010011" =>
                mux_out <= in19;


            when "010100" =>
                mux_out <= in20;


            when "010101" =>
                mux_out <= in21;


            when "010110" =>
                mux_out <= in22;


            when "010111" =>
                mux_out <= in23;


            when "011000" =>
                mux_out <= in24;


            when "011001" =>
                mux_out <= in25;


            when "011010" =>
                mux_out <= in26;


            when "011011" =>
                mux_out <= in27;


            when "011100" =>
                mux_out <= in28;


            when "011101" =>
                mux_out <= in29;


            when "011110" =>
                mux_out <= in30;


            when "011111" =>
                mux_out <= in31;


            when "100000" =>
                mux_out <= in32;


            when "100001" =>
                mux_out <= in33;


            when "100010" =>
                mux_out <= in34;


            when "100011" =>
                mux_out <= in35;


            when "100100" =>
                mux_out <= in36;


            when "100101" =>
                mux_out <= in37;


            when "100110" =>
                mux_out <= in38;


            when "100111" =>
                mux_out <= in39;


            when "101000" =>
                mux_out <= in40;


            when "101001" =>
                mux_out <= in41;


            when "101010" =>
                mux_out <= in42;


            when "101011" =>
                mux_out <= in43;


            when "101100" =>
                mux_out <= in44;


            when "101101" =>
                mux_out <= in45;


            when "101110" =>
                mux_out <= in46;


            when "101111" =>
                mux_out <= in47;


            when "110000" =>
                mux_out <= in48;


            when "110001" =>
                mux_out <= in49;


            when "110010" =>
                mux_out <= in50;


            when "110011" =>
                mux_out <= in51;


            when "110100" =>
                mux_out <= in52;


            when "110101" =>
                mux_out <= in53;


            when "110110" =>
                mux_out <= in54;


            when "110111" =>
                mux_out <= in55;


            when "111000" =>
                mux_out <= in56;


            when "111001" =>
                mux_out <= in57;


            when "111010" =>
                mux_out <= in58;


            when "111011" =>
                mux_out <= in59;


            when "111100" =>
                mux_out <= in60;


            when "111101" =>
                mux_out <= in61;


            when "111110" =>
                mux_out <= in62;


            when others =>
                mux_out <= in63;


        end case;

    end process selection;


end architecture behavior;

