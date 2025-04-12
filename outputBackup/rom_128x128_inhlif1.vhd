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


entity rom_128x128_inhlif1 is
    port (
        clka : in std_logic;
        addra : in std_logic_vector(6 downto 0);
        dout_0 : out std_logic_vector(5 downto 0);
        dout_1 : out std_logic_vector(5 downto 0);
        dout_2 : out std_logic_vector(5 downto 0);
        dout_3 : out std_logic_vector(5 downto 0);
        dout_4 : out std_logic_vector(5 downto 0);
        dout_5 : out std_logic_vector(5 downto 0);
        dout_6 : out std_logic_vector(5 downto 0);
        dout_7 : out std_logic_vector(5 downto 0);
        dout_8 : out std_logic_vector(5 downto 0);
        dout_9 : out std_logic_vector(5 downto 0);
        dout_a : out std_logic_vector(5 downto 0);
        dout_b : out std_logic_vector(5 downto 0);
        dout_c : out std_logic_vector(5 downto 0);
        dout_d : out std_logic_vector(5 downto 0);
        dout_e : out std_logic_vector(5 downto 0);
        dout_f : out std_logic_vector(5 downto 0);
        dout_10 : out std_logic_vector(5 downto 0);
        dout_11 : out std_logic_vector(5 downto 0);
        dout_12 : out std_logic_vector(5 downto 0);
        dout_13 : out std_logic_vector(5 downto 0);
        dout_14 : out std_logic_vector(5 downto 0);
        dout_15 : out std_logic_vector(5 downto 0);
        dout_16 : out std_logic_vector(5 downto 0);
        dout_17 : out std_logic_vector(5 downto 0);
        dout_18 : out std_logic_vector(5 downto 0);
        dout_19 : out std_logic_vector(5 downto 0);
        dout_1a : out std_logic_vector(5 downto 0);
        dout_1b : out std_logic_vector(5 downto 0);
        dout_1c : out std_logic_vector(5 downto 0);
        dout_1d : out std_logic_vector(5 downto 0);
        dout_1e : out std_logic_vector(5 downto 0);
        dout_1f : out std_logic_vector(5 downto 0);
        dout_20 : out std_logic_vector(5 downto 0);
        dout_21 : out std_logic_vector(5 downto 0);
        dout_22 : out std_logic_vector(5 downto 0);
        dout_23 : out std_logic_vector(5 downto 0);
        dout_24 : out std_logic_vector(5 downto 0);
        dout_25 : out std_logic_vector(5 downto 0);
        dout_26 : out std_logic_vector(5 downto 0);
        dout_27 : out std_logic_vector(5 downto 0);
        dout_28 : out std_logic_vector(5 downto 0);
        dout_29 : out std_logic_vector(5 downto 0);
        dout_2a : out std_logic_vector(5 downto 0);
        dout_2b : out std_logic_vector(5 downto 0);
        dout_2c : out std_logic_vector(5 downto 0);
        dout_2d : out std_logic_vector(5 downto 0);
        dout_2e : out std_logic_vector(5 downto 0);
        dout_2f : out std_logic_vector(5 downto 0);
        dout_30 : out std_logic_vector(5 downto 0);
        dout_31 : out std_logic_vector(5 downto 0);
        dout_32 : out std_logic_vector(5 downto 0);
        dout_33 : out std_logic_vector(5 downto 0);
        dout_34 : out std_logic_vector(5 downto 0);
        dout_35 : out std_logic_vector(5 downto 0);
        dout_36 : out std_logic_vector(5 downto 0);
        dout_37 : out std_logic_vector(5 downto 0);
        dout_38 : out std_logic_vector(5 downto 0);
        dout_39 : out std_logic_vector(5 downto 0);
        dout_3a : out std_logic_vector(5 downto 0);
        dout_3b : out std_logic_vector(5 downto 0);
        dout_3c : out std_logic_vector(5 downto 0);
        dout_3d : out std_logic_vector(5 downto 0);
        dout_3e : out std_logic_vector(5 downto 0);
        dout_3f : out std_logic_vector(5 downto 0);
        dout_40 : out std_logic_vector(5 downto 0);
        dout_41 : out std_logic_vector(5 downto 0);
        dout_42 : out std_logic_vector(5 downto 0);
        dout_43 : out std_logic_vector(5 downto 0);
        dout_44 : out std_logic_vector(5 downto 0);
        dout_45 : out std_logic_vector(5 downto 0);
        dout_46 : out std_logic_vector(5 downto 0);
        dout_47 : out std_logic_vector(5 downto 0);
        dout_48 : out std_logic_vector(5 downto 0);
        dout_49 : out std_logic_vector(5 downto 0);
        dout_4a : out std_logic_vector(5 downto 0);
        dout_4b : out std_logic_vector(5 downto 0);
        dout_4c : out std_logic_vector(5 downto 0);
        dout_4d : out std_logic_vector(5 downto 0);
        dout_4e : out std_logic_vector(5 downto 0);
        dout_4f : out std_logic_vector(5 downto 0);
        dout_50 : out std_logic_vector(5 downto 0);
        dout_51 : out std_logic_vector(5 downto 0);
        dout_52 : out std_logic_vector(5 downto 0);
        dout_53 : out std_logic_vector(5 downto 0);
        dout_54 : out std_logic_vector(5 downto 0);
        dout_55 : out std_logic_vector(5 downto 0);
        dout_56 : out std_logic_vector(5 downto 0);
        dout_57 : out std_logic_vector(5 downto 0);
        dout_58 : out std_logic_vector(5 downto 0);
        dout_59 : out std_logic_vector(5 downto 0);
        dout_5a : out std_logic_vector(5 downto 0);
        dout_5b : out std_logic_vector(5 downto 0);
        dout_5c : out std_logic_vector(5 downto 0);
        dout_5d : out std_logic_vector(5 downto 0);
        dout_5e : out std_logic_vector(5 downto 0);
        dout_5f : out std_logic_vector(5 downto 0);
        dout_60 : out std_logic_vector(5 downto 0);
        dout_61 : out std_logic_vector(5 downto 0);
        dout_62 : out std_logic_vector(5 downto 0);
        dout_63 : out std_logic_vector(5 downto 0);
        dout_64 : out std_logic_vector(5 downto 0);
        dout_65 : out std_logic_vector(5 downto 0);
        dout_66 : out std_logic_vector(5 downto 0);
        dout_67 : out std_logic_vector(5 downto 0);
        dout_68 : out std_logic_vector(5 downto 0);
        dout_69 : out std_logic_vector(5 downto 0);
        dout_6a : out std_logic_vector(5 downto 0);
        dout_6b : out std_logic_vector(5 downto 0);
        dout_6c : out std_logic_vector(5 downto 0);
        dout_6d : out std_logic_vector(5 downto 0);
        dout_6e : out std_logic_vector(5 downto 0);
        dout_6f : out std_logic_vector(5 downto 0);
        dout_70 : out std_logic_vector(5 downto 0);
        dout_71 : out std_logic_vector(5 downto 0);
        dout_72 : out std_logic_vector(5 downto 0);
        dout_73 : out std_logic_vector(5 downto 0);
        dout_74 : out std_logic_vector(5 downto 0);
        dout_75 : out std_logic_vector(5 downto 0);
        dout_76 : out std_logic_vector(5 downto 0);
        dout_77 : out std_logic_vector(5 downto 0);
        dout_78 : out std_logic_vector(5 downto 0);
        dout_79 : out std_logic_vector(5 downto 0);
        dout_7a : out std_logic_vector(5 downto 0);
        dout_7b : out std_logic_vector(5 downto 0);
        dout_7c : out std_logic_vector(5 downto 0);
        dout_7d : out std_logic_vector(5 downto 0);
        dout_7e : out std_logic_vector(5 downto 0);
        dout_7f : out std_logic_vector(5 downto 0)
    );
end entity rom_128x128_inhlif1;

architecture behavior of rom_128x128_inhlif1 is


    component rom_128x128_inhlif1_ip is
        port (
            clka : in std_logic;
            addra : in std_logic_vector(6 downto 0);
            douta : out std_logic_vector(767 downto 0)
        );
    end component;


    signal douta : std_logic_vector(767 downto 0);

begin

    dout_0 <= douta(5 downto 0);
    dout_1 <= douta(11 downto 6);
    dout_2 <= douta(17 downto 12);
    dout_3 <= douta(23 downto 18);
    dout_4 <= douta(29 downto 24);
    dout_5 <= douta(35 downto 30);
    dout_6 <= douta(41 downto 36);
    dout_7 <= douta(47 downto 42);
    dout_8 <= douta(53 downto 48);
    dout_9 <= douta(59 downto 54);
    dout_a <= douta(65 downto 60);
    dout_b <= douta(71 downto 66);
    dout_c <= douta(77 downto 72);
    dout_d <= douta(83 downto 78);
    dout_e <= douta(89 downto 84);
    dout_f <= douta(95 downto 90);
    dout_10 <= douta(101 downto 96);
    dout_11 <= douta(107 downto 102);
    dout_12 <= douta(113 downto 108);
    dout_13 <= douta(119 downto 114);
    dout_14 <= douta(125 downto 120);
    dout_15 <= douta(131 downto 126);
    dout_16 <= douta(137 downto 132);
    dout_17 <= douta(143 downto 138);
    dout_18 <= douta(149 downto 144);
    dout_19 <= douta(155 downto 150);
    dout_1a <= douta(161 downto 156);
    dout_1b <= douta(167 downto 162);
    dout_1c <= douta(173 downto 168);
    dout_1d <= douta(179 downto 174);
    dout_1e <= douta(185 downto 180);
    dout_1f <= douta(191 downto 186);
    dout_20 <= douta(197 downto 192);
    dout_21 <= douta(203 downto 198);
    dout_22 <= douta(209 downto 204);
    dout_23 <= douta(215 downto 210);
    dout_24 <= douta(221 downto 216);
    dout_25 <= douta(227 downto 222);
    dout_26 <= douta(233 downto 228);
    dout_27 <= douta(239 downto 234);
    dout_28 <= douta(245 downto 240);
    dout_29 <= douta(251 downto 246);
    dout_2a <= douta(257 downto 252);
    dout_2b <= douta(263 downto 258);
    dout_2c <= douta(269 downto 264);
    dout_2d <= douta(275 downto 270);
    dout_2e <= douta(281 downto 276);
    dout_2f <= douta(287 downto 282);
    dout_30 <= douta(293 downto 288);
    dout_31 <= douta(299 downto 294);
    dout_32 <= douta(305 downto 300);
    dout_33 <= douta(311 downto 306);
    dout_34 <= douta(317 downto 312);
    dout_35 <= douta(323 downto 318);
    dout_36 <= douta(329 downto 324);
    dout_37 <= douta(335 downto 330);
    dout_38 <= douta(341 downto 336);
    dout_39 <= douta(347 downto 342);
    dout_3a <= douta(353 downto 348);
    dout_3b <= douta(359 downto 354);
    dout_3c <= douta(365 downto 360);
    dout_3d <= douta(371 downto 366);
    dout_3e <= douta(377 downto 372);
    dout_3f <= douta(383 downto 378);
    dout_40 <= douta(389 downto 384);
    dout_41 <= douta(395 downto 390);
    dout_42 <= douta(401 downto 396);
    dout_43 <= douta(407 downto 402);
    dout_44 <= douta(413 downto 408);
    dout_45 <= douta(419 downto 414);
    dout_46 <= douta(425 downto 420);
    dout_47 <= douta(431 downto 426);
    dout_48 <= douta(437 downto 432);
    dout_49 <= douta(443 downto 438);
    dout_4a <= douta(449 downto 444);
    dout_4b <= douta(455 downto 450);
    dout_4c <= douta(461 downto 456);
    dout_4d <= douta(467 downto 462);
    dout_4e <= douta(473 downto 468);
    dout_4f <= douta(479 downto 474);
    dout_50 <= douta(485 downto 480);
    dout_51 <= douta(491 downto 486);
    dout_52 <= douta(497 downto 492);
    dout_53 <= douta(503 downto 498);
    dout_54 <= douta(509 downto 504);
    dout_55 <= douta(515 downto 510);
    dout_56 <= douta(521 downto 516);
    dout_57 <= douta(527 downto 522);
    dout_58 <= douta(533 downto 528);
    dout_59 <= douta(539 downto 534);
    dout_5a <= douta(545 downto 540);
    dout_5b <= douta(551 downto 546);
    dout_5c <= douta(557 downto 552);
    dout_5d <= douta(563 downto 558);
    dout_5e <= douta(569 downto 564);
    dout_5f <= douta(575 downto 570);
    dout_60 <= douta(581 downto 576);
    dout_61 <= douta(587 downto 582);
    dout_62 <= douta(593 downto 588);
    dout_63 <= douta(599 downto 594);
    dout_64 <= douta(605 downto 600);
    dout_65 <= douta(611 downto 606);
    dout_66 <= douta(617 downto 612);
    dout_67 <= douta(623 downto 618);
    dout_68 <= douta(629 downto 624);
    dout_69 <= douta(635 downto 630);
    dout_6a <= douta(641 downto 636);
    dout_6b <= douta(647 downto 642);
    dout_6c <= douta(653 downto 648);
    dout_6d <= douta(659 downto 654);
    dout_6e <= douta(665 downto 660);
    dout_6f <= douta(671 downto 666);
    dout_70 <= douta(677 downto 672);
    dout_71 <= douta(683 downto 678);
    dout_72 <= douta(689 downto 684);
    dout_73 <= douta(695 downto 690);
    dout_74 <= douta(701 downto 696);
    dout_75 <= douta(707 downto 702);
    dout_76 <= douta(713 downto 708);
    dout_77 <= douta(719 downto 714);
    dout_78 <= douta(725 downto 720);
    dout_79 <= douta(731 downto 726);
    dout_7a <= douta(737 downto 732);
    dout_7b <= douta(743 downto 738);
    dout_7c <= douta(749 downto 744);
    dout_7d <= douta(755 downto 750);
    dout_7e <= douta(761 downto 756);
    dout_7f <= douta(767 downto 762);


    rom_128x128_inhlif1_ip_instance : rom_128x128_inhlif1_ip
        port map(
            clka => clka,
            addra => addra,
            douta => douta
        );


end architecture behavior;

