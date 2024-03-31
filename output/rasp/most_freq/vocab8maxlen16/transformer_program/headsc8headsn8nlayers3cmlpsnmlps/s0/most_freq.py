import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys] for q in queries]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/rasp/most_freq/vocab8maxlen16/transformer_program/headsc8headsn8nlayers3cmlpsnmlps/s0/most_freq_weights.csv",
        index_col=[0, 1],
        dtype={"feature": str},
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(q_position, k_position):
        if q_position in {0, 3}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2, 7}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 12
        elif q_position in {5}:
            return k_position == 2
        elif q_position in {6, 8, 10, 11, 12}:
            return k_position == 3
        elif q_position in {9}:
            return k_position == 7
        elif q_position in {13}:
            return k_position == 9
        elif q_position in {14, 15}:
            return k_position == 14

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0, 4, 5, 6, 8, 10}:
            return token == "4"
        elif position in {1, 7}:
            return token == "2"
        elif position in {2, 3, 13}:
            return token == "1"
        elif position in {9, 11}:
            return token == "0"
        elif position in {12, 14}:
            return token == "3"
        elif position in {15}:
            return token == "<s>"

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0, 3, 6}:
            return k_position == 2
        elif q_position in {1, 9}:
            return k_position == 4
        elif q_position in {2, 4}:
            return k_position == 1
        elif q_position in {5}:
            return k_position == 3
        elif q_position in {7}:
            return k_position == 5
        elif q_position in {8}:
            return k_position == 6
        elif q_position in {10, 13}:
            return k_position == 12
        elif q_position in {11}:
            return k_position == 9
        elif q_position in {12}:
            return k_position == 0
        elif q_position in {14}:
            return k_position == 15
        elif q_position in {15}:
            return k_position == 10

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 9}:
            return k_position == 3
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 11
        elif q_position in {8, 7}:
            return k_position == 4
        elif q_position in {10, 11}:
            return k_position == 6
        elif q_position in {12, 13}:
            return k_position == 12
        elif q_position in {14, 15}:
            return k_position == 0

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(position, token):
        if position in {0, 4, 5, 6, 9, 10, 12, 14, 15}:
            return token == "3"
        elif position in {1}:
            return token == "1"
        elif position in {2, 3}:
            return token == "0"
        elif position in {11, 7}:
            return token == "2"
        elif position in {8}:
            return token == "5"
        elif position in {13}:
            return token == ""

    attn_0_4_pattern = select_closest(tokens, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {0, 11, 14, 7}:
            return token == "2"
        elif position in {1}:
            return token == "4"
        elif position in {2, 8, 12, 13, 15}:
            return token == "5"
        elif position in {3, 4, 5, 6}:
            return token == "1"
        elif position in {9}:
            return token == "0"
        elif position in {10}:
            return token == "3"

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 11}:
            return k_position == 8
        elif q_position in {1}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3, 5}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 2
        elif q_position in {13, 6}:
            return k_position == 3
        elif q_position in {7}:
            return k_position == 7
        elif q_position in {8}:
            return k_position == 15
        elif q_position in {9}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 0
        elif q_position in {12, 15}:
            return k_position == 12
        elif q_position in {14}:
            return k_position == 11

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 4
        elif q_position in {1, 12}:
            return k_position == 13
        elif q_position in {2, 14, 7}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 3
        elif q_position in {10, 5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 10
        elif q_position in {8}:
            return k_position == 5
        elif q_position in {9, 15}:
            return k_position == 11
        elif q_position in {11}:
            return k_position == 2
        elif q_position in {13}:
            return k_position == 14

    attn_0_7_pattern = select_closest(positions, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 4, 5, 6, 8, 11}:
            return token == "5"
        elif position in {1}:
            return token == "4"
        elif position in {10, 2, 3, 12}:
            return token == "<s>"
        elif position in {9, 13, 7}:
            return token == "2"
        elif position in {14}:
            return token == ""
        elif position in {15}:
            return token == "1"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 5, 6, 8, 9, 10, 11}:
            return token == "<s>"
        elif position in {1}:
            return token == "1"
        elif position in {2, 3, 4, 14}:
            return token == ""
        elif position in {12, 15, 7}:
            return token == "3"
        elif position in {13}:
            return token == "5"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 2, 3, 4, 5, 7, 8}:
            return token == "<s>"
        elif position in {1}:
            return token == "4"
        elif position in {6, 9, 10, 11, 12, 13, 15}:
            return token == ""
        elif position in {14}:
            return token == "<pad>"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0, 2, 3, 4, 5, 8, 9, 13}:
            return token == "<s>"
        elif position in {1, 11, 7}:
            return token == "1"
        elif position in {6, 10, 12, 14, 15}:
            return token == ""

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0, 2}:
            return token == ""
        elif position in {1, 10, 12, 15}:
            return token == "0"
        elif position in {3, 4}:
            return token == "<pad>"
        elif position in {5, 6}:
            return token == "<s>"
        elif position in {8, 11, 14, 7}:
            return token == "1"
        elif position in {9}:
            return token == "5"
        elif position in {13}:
            return token == "3"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0}:
            return token == "<pad>"
        elif position in {8, 1, 12}:
            return token == "5"
        elif position in {2, 3, 4}:
            return token == ""
        elif position in {5, 6, 14}:
            return token == "4"
        elif position in {7}:
            return token == "0"
        elif position in {9, 10, 13}:
            return token == "2"
        elif position in {11, 15}:
            return token == "1"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 2, 3, 4}:
            return token == "<s>"
        elif position in {1}:
            return token == "2"
        elif position in {5, 6}:
            return token == "4"
        elif position in {7, 8, 9, 10, 11, 12, 14, 15}:
            return token == ""
        elif position in {13}:
            return token == "<pad>"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 2, 3}:
            return token == ""
        elif position in {1, 10}:
            return token == "3"
        elif position in {4, 5}:
            return token == "<pad>"
        elif position in {6}:
            return token == "<s>"
        elif position in {7, 8, 9, 12, 14, 15}:
            return token == "5"
        elif position in {11}:
            return token == "1"
        elif position in {13}:
            return token == "0"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, position):
        key = (attn_0_1_output, position)
        if key in {
            ("4", 0),
            ("4", 1),
            ("4", 2),
            ("4", 3),
            ("4", 4),
            ("4", 5),
            ("4", 6),
            ("4", 7),
            ("4", 8),
            ("4", 9),
            ("4", 10),
            ("4", 11),
            ("4", 12),
            ("4", 13),
            ("4", 14),
            ("4", 15),
        }:
            return 7
        elif key in {("0", 1), ("1", 1), ("2", 1), ("3", 1), ("5", 1), ("<s>", 1)}:
            return 13
        return 2

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(position, attn_0_5_output):
        key = (position, attn_0_5_output)
        if key in {
            (5, "0"),
            (5, "1"),
            (5, "3"),
            (5, "4"),
            (10, "0"),
            (10, "1"),
            (10, "2"),
            (10, "3"),
            (10, "4"),
            (10, "<s>"),
            (14, "0"),
            (14, "1"),
            (14, "2"),
            (14, "3"),
            (14, "4"),
            (14, "<s>"),
            (15, "0"),
            (15, "1"),
            (15, "3"),
            (15, "4"),
            (15, "<s>"),
        }:
            return 1
        elif key in {
            (0, "5"),
            (1, "5"),
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "5"),
            (3, "5"),
            (4, "5"),
            (5, "5"),
            (6, "2"),
            (6, "5"),
            (7, "5"),
            (8, "5"),
            (9, "5"),
            (10, "5"),
            (11, "5"),
            (12, "5"),
            (13, "2"),
            (13, "5"),
            (14, "5"),
            (15, "2"),
            (15, "5"),
        }:
            return 10
        elif key in {
            (1, "2"),
            (2, "3"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "<s>"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
        }:
            return 15
        elif key in {
            (0, "2"),
            (6, "1"),
            (7, "2"),
            (11, "2"),
            (13, "0"),
            (13, "1"),
            (13, "3"),
            (13, "4"),
        }:
            return 6
        elif key in {(2, "4"), (2, "<s>"), (4, "0"), (4, "4"), (4, "<s>")}:
            return 14
        elif key in {(5, "2"), (5, "<s>")}:
            return 8
        return 12

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(positions, attn_0_5_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(attn_0_0_output, attn_0_5_output):
        key = (attn_0_0_output, attn_0_5_output)
        if key in {
            ("0", "2"),
            ("0", "3"),
            ("0", "<s>"),
            ("1", "2"),
            ("2", "2"),
            ("2", "3"),
            ("2", "<s>"),
            ("3", "2"),
            ("3", "3"),
            ("3", "<s>"),
            ("4", "0"),
            ("4", "2"),
            ("4", "3"),
            ("4", "5"),
            ("4", "<s>"),
            ("5", "2"),
            ("<s>", "2"),
        }:
            return 11
        return 10

    mlp_0_2_outputs = [
        mlp_0_2(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_5_outputs)
    ]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_5_output, attn_0_4_output):
        key = (attn_0_5_output, attn_0_4_output)
        if key in {
            ("1", "1"),
            ("1", "2"),
            ("1", "4"),
            ("1", "<s>"),
            ("2", "1"),
            ("2", "2"),
            ("2", "4"),
            ("2", "<s>"),
            ("4", "1"),
            ("4", "2"),
            ("4", "4"),
            ("4", "<s>"),
            ("5", "1"),
            ("5", "2"),
            ("5", "4"),
            ("5", "<s>"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "4"),
            ("<s>", "5"),
            ("<s>", "<s>"),
        }:
            return 0
        elif key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("0", "4"),
            ("0", "5"),
            ("0", "<s>"),
            ("1", "0"),
            ("1", "5"),
            ("2", "0"),
            ("2", "5"),
            ("4", "0"),
            ("4", "5"),
            ("5", "0"),
            ("5", "5"),
            ("<s>", "0"),
        }:
            return 2
        elif key in {("0", "3"), ("3", "0"), ("3", "5")}:
            return 6
        return 14

    mlp_0_3_outputs = [
        mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_5_outputs, attn_0_4_outputs)
    ]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output, num_attn_0_6_output):
        key = (num_attn_0_2_output, num_attn_0_6_output)
        return 11

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_7_output, num_attn_0_1_output):
        key = (num_attn_0_7_output, num_attn_0_1_output)
        return 10

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_5_output, num_attn_0_4_output):
        key = (num_attn_0_5_output, num_attn_0_4_output)
        return 15

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_2_output, num_attn_0_0_output):
        key = (num_attn_0_2_output, num_attn_0_0_output)
        if key in {
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 15),
            (2, 6),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (3, 7),
            (3, 8),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
            (3, 14),
            (3, 15),
            (4, 9),
            (4, 10),
            (4, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
            (5, 10),
            (5, 11),
            (5, 12),
            (5, 13),
            (5, 14),
            (5, 15),
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 14),
            (6, 15),
            (7, 13),
            (7, 14),
            (7, 15),
            (8, 14),
            (8, 15),
        }:
            return 9
        return 8

    num_mlp_0_3_outputs = [
        num_mlp_0_3(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 5, 6}:
            return k_position == 8
        elif q_position in {1}:
            return k_position == 2
        elif q_position in {2}:
            return k_position == 11
        elif q_position in {3, 4}:
            return k_position == 5
        elif q_position in {8, 9, 7}:
            return k_position == 1
        elif q_position in {10, 13}:
            return k_position == 10
        elif q_position in {11, 12, 14, 15}:
            return k_position == 14

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, attn_0_3_output):
        if position in {0, 13, 5, 6}:
            return attn_0_3_output == "3"
        elif position in {1, 2, 3, 4}:
            return attn_0_3_output == "0"
        elif position in {7}:
            return attn_0_3_output == ""
        elif position in {8, 9, 10, 11, 12, 14, 15}:
            return attn_0_3_output == "<s>"

    attn_1_1_pattern = select_closest(attn_0_3_outputs, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_3_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(token, position):
        if token in {"<s>", "5", "0", "2", "4", "3", "1"}:
            return position == 7

    attn_1_2_pattern = select_closest(positions, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, mlp_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "4"
        elif q_token in {"1"}:
            return k_token == "2"
        elif q_token in {"5", "2"}:
            return k_token == ""
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == "0"

    attn_1_3_pattern = select_closest(tokens, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(q_token, k_token):
        if q_token in {"0", "2", "4", "3", "1"}:
            return k_token == "5"
        elif q_token in {"5"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_1_4_pattern = select_closest(tokens, tokens, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(position, mlp_0_1_output):
        if position in {0, 2, 3}:
            return mlp_0_1_output == 4
        elif position in {1}:
            return mlp_0_1_output == 10
        elif position in {4, 5, 6}:
            return mlp_0_1_output == 6
        elif position in {7, 8, 9, 10, 11, 12, 13, 14, 15}:
            return mlp_0_1_output == 14

    attn_1_5_pattern = select_closest(mlp_0_1_outputs, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, mlp_0_0_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(attn_0_4_output, position):
        if attn_0_4_output in {"<s>", "0"}:
            return position == 5
        elif attn_0_4_output in {"1"}:
            return position == 3
        elif attn_0_4_output in {"2"}:
            return position == 6
        elif attn_0_4_output in {"3"}:
            return position == 10
        elif attn_0_4_output in {"4"}:
            return position == 12
        elif attn_0_4_output in {"5"}:
            return position == 9

    attn_1_6_pattern = select_closest(positions, attn_0_4_outputs, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, mlp_0_3_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(token, attn_0_7_output):
        if token in {"0"}:
            return attn_0_7_output == ""
        elif token in {"1", "2"}:
            return attn_0_7_output == "0"
        elif token in {"3"}:
            return attn_0_7_output == "3"
        elif token in {"5", "4"}:
            return attn_0_7_output == "5"
        elif token in {"<s>"}:
            return attn_0_7_output == "2"

    attn_1_7_pattern = select_closest(attn_0_7_outputs, tokens, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_6_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, token):
        if position in {0, 2}:
            return token == ""
        elif position in {1}:
            return token == "2"
        elif position in {3}:
            return token == "<pad>"
        elif position in {4, 5, 6, 10, 15}:
            return token == "0"
        elif position in {14, 7}:
            return token == "1"
        elif position in {8, 11}:
            return token == "5"
        elif position in {9, 12, 13}:
            return token == "4"

    num_attn_1_0_pattern = select(tokens, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(position, attn_0_2_output):
        if position in {0, 2, 3, 4}:
            return attn_0_2_output == "<s>"
        elif position in {1}:
            return attn_0_2_output == "5"
        elif position in {5, 6, 7, 10, 12, 14}:
            return attn_0_2_output == ""
        elif position in {8, 9, 11, 13, 15}:
            return attn_0_2_output == "<pad>"

    num_attn_1_1_pattern = select(attn_0_2_outputs, positions, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_0_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_7_output, attn_0_4_output):
        if attn_0_7_output in {"0"}:
            return attn_0_4_output == "0"
        elif attn_0_7_output in {"1"}:
            return attn_0_4_output == "1"
        elif attn_0_7_output in {"<s>", "5", "2", "4"}:
            return attn_0_4_output == ""
        elif attn_0_7_output in {"3"}:
            return attn_0_4_output == "<pad>"

    num_attn_1_2_pattern = select(attn_0_4_outputs, attn_0_7_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_7_output, token):
        if attn_0_7_output in {"5", "0", "2", "4"}:
            return token == ""
        elif attn_0_7_output in {"1"}:
            return token == "1"
        elif attn_0_7_output in {"3"}:
            return token == "3"
        elif attn_0_7_output in {"<s>"}:
            return token == "<pad>"

    num_attn_1_3_pattern = select(tokens, attn_0_7_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(position, token):
        if position in {0, 2, 3}:
            return token == "<s>"
        elif position in {1}:
            return token == "4"
        elif position in {4, 5, 6}:
            return token == "2"
        elif position in {7, 10, 12, 13, 15}:
            return token == "<pad>"
        elif position in {8, 9, 11, 14}:
            return token == ""

    num_attn_1_4_pattern = select(tokens, positions, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, ones)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(position, attn_0_3_output):
        if position in {0, 2}:
            return attn_0_3_output == "<s>"
        elif position in {1, 14}:
            return attn_0_3_output == "4"
        elif position in {3}:
            return attn_0_3_output == "<pad>"
        elif position in {4, 5}:
            return attn_0_3_output == ""
        elif position in {12, 6}:
            return attn_0_3_output == "0"
        elif position in {15, 13, 7}:
            return attn_0_3_output == "5"
        elif position in {8, 9, 11}:
            return attn_0_3_output == "3"
        elif position in {10}:
            return attn_0_3_output == "2"

    num_attn_1_5_pattern = select(attn_0_3_outputs, positions, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_2_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(position, token):
        if position in {0}:
            return token == "3"
        elif position in {1}:
            return token == "5"
        elif position in {2}:
            return token == "<s>"
        elif position in {3, 4, 5, 6}:
            return token == "1"
        elif position in {8, 7}:
            return token == ""
        elif position in {9, 10, 11, 12, 13, 14, 15}:
            return token == "<pad>"

    num_attn_1_6_pattern = select(tokens, positions, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_3_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(position, attn_0_2_output):
        if position in {0, 3}:
            return attn_0_2_output == "<s>"
        elif position in {1, 2}:
            return attn_0_2_output == "3"
        elif position in {4, 5, 6}:
            return attn_0_2_output == "0"
        elif position in {8, 11, 14, 7}:
            return attn_0_2_output == ""
        elif position in {9, 10, 12, 13, 15}:
            return attn_0_2_output == "<pad>"

    num_attn_1_7_pattern = select(attn_0_2_outputs, positions, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_2_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(position):
        key = position
        if key in {1, 2, 3}:
            return 8
        return 4

    mlp_1_0_outputs = [mlp_1_0(k0) for k0 in positions]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_7_output, attn_1_5_output):
        key = (attn_1_7_output, attn_1_5_output)
        return 13

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_7_outputs, attn_1_5_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(mlp_0_2_output, mlp_0_0_output):
        key = (mlp_0_2_output, mlp_0_0_output)
        if key in {
            (1, 6),
            (1, 8),
            (9, 6),
            (9, 8),
            (11, 6),
            (11, 8),
            (14, 6),
            (14, 8),
            (15, 6),
            (15, 8),
        }:
            return 8
        return 2

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(mlp_0_2_outputs, mlp_0_0_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(attn_1_7_output, attn_0_5_output):
        key = (attn_1_7_output, attn_0_5_output)
        if key in {
            ("0", "1"),
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "5"),
            ("1", "<s>"),
            ("2", "1"),
            ("3", "1"),
            ("4", "1"),
            ("5", "1"),
            ("<s>", "1"),
        }:
            return 4
        return 2

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(attn_1_7_outputs, attn_0_5_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_1_output, num_attn_1_1_output):
        key = (num_attn_0_1_output, num_attn_1_1_output)
        if key in {
            (2, 0),
            (3, 0),
            (4, 0),
            (4, 1),
            (5, 0),
            (5, 1),
            (6, 0),
            (6, 1),
            (7, 0),
            (7, 1),
            (7, 2),
            (8, 0),
            (8, 1),
            (8, 2),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (12, 4),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (14, 4),
            (14, 5),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 5),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (16, 4),
            (16, 5),
            (16, 6),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 4),
            (17, 5),
            (17, 6),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (18, 4),
            (18, 5),
            (18, 6),
            (18, 7),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 4),
            (19, 5),
            (19, 6),
            (19, 7),
            (20, 0),
            (20, 1),
            (20, 2),
            (20, 3),
            (20, 4),
            (20, 5),
            (20, 6),
            (20, 7),
            (20, 8),
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (21, 4),
            (21, 5),
            (21, 6),
            (21, 7),
            (21, 8),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (22, 4),
            (22, 5),
            (22, 6),
            (22, 7),
            (22, 8),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (23, 4),
            (23, 5),
            (23, 6),
            (23, 7),
            (23, 8),
            (23, 9),
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (24, 4),
            (24, 5),
            (24, 6),
            (24, 7),
            (24, 8),
            (24, 9),
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
            (25, 4),
            (25, 5),
            (25, 6),
            (25, 7),
            (25, 8),
            (25, 9),
            (25, 10),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (26, 4),
            (26, 5),
            (26, 6),
            (26, 7),
            (26, 8),
            (26, 9),
            (26, 10),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (27, 4),
            (27, 5),
            (27, 6),
            (27, 7),
            (27, 8),
            (27, 9),
            (27, 10),
            (27, 11),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
            (28, 5),
            (28, 6),
            (28, 7),
            (28, 8),
            (28, 9),
            (28, 10),
            (28, 11),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (29, 4),
            (29, 5),
            (29, 6),
            (29, 7),
            (29, 8),
            (29, 9),
            (29, 10),
            (29, 11),
            (29, 12),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (30, 4),
            (30, 5),
            (30, 6),
            (30, 7),
            (30, 8),
            (30, 9),
            (30, 10),
            (30, 11),
            (30, 12),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (31, 4),
            (31, 5),
            (31, 6),
            (31, 7),
            (31, 8),
            (31, 9),
            (31, 10),
            (31, 11),
            (31, 12),
        }:
            return 14
        elif key in {(0, 0)}:
            return 13
        return 7

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_7_output, num_attn_0_4_output):
        key = (num_attn_0_7_output, num_attn_0_4_output)
        return 6

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_0_4_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_0_5_output, num_attn_1_4_output):
        key = (num_attn_0_5_output, num_attn_1_4_output)
        return 7

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_0_4_output, num_attn_1_5_output):
        key = (num_attn_0_4_output, num_attn_1_5_output)
        return 15

    num_mlp_1_3_outputs = [
        num_mlp_1_3(k0, k1)
        for k0, k1 in zip(num_attn_0_4_outputs, num_attn_1_5_outputs)
    ]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_3_output, mlp_0_1_output):
        if attn_0_3_output in {"0"}:
            return mlp_0_1_output == 3
        elif attn_0_3_output in {"3", "1", "4"}:
            return mlp_0_1_output == 12
        elif attn_0_3_output in {"5", "2"}:
            return mlp_0_1_output == 8
        elif attn_0_3_output in {"<s>"}:
            return mlp_0_1_output == 15

    attn_2_0_pattern = select_closest(mlp_0_1_outputs, attn_0_3_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 9
        elif q_position in {1, 10}:
            return k_position == 11
        elif q_position in {2, 5}:
            return k_position == 4
        elif q_position in {3, 15}:
            return k_position == 8
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {6}:
            return k_position == 10
        elif q_position in {7}:
            return k_position == 5
        elif q_position in {8, 9}:
            return k_position == 1
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {12}:
            return k_position == 7
        elif q_position in {13}:
            return k_position == 0
        elif q_position in {14}:
            return k_position == 12

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, mlp_0_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"3", "0", "4"}:
            return k_token == "<s>"
        elif q_token in {"1", "2"}:
            return k_token == "5"
        elif q_token in {"5"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, mlp_0_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(position, mlp_0_0_output):
        if position in {0, 1, 4, 5, 6}:
            return mlp_0_0_output == 7
        elif position in {2, 3}:
            return mlp_0_0_output == 10
        elif position in {7, 8, 9, 10, 11, 12, 13, 14, 15}:
            return mlp_0_0_output == 2

    attn_2_3_pattern = select_closest(mlp_0_0_outputs, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, mlp_0_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(q_attn_0_4_output, k_attn_0_4_output):
        if q_attn_0_4_output in {"3", "0"}:
            return k_attn_0_4_output == "0"
        elif q_attn_0_4_output in {"1"}:
            return k_attn_0_4_output == "4"
        elif q_attn_0_4_output in {"2"}:
            return k_attn_0_4_output == "2"
        elif q_attn_0_4_output in {"4"}:
            return k_attn_0_4_output == "3"
        elif q_attn_0_4_output in {"<s>", "5"}:
            return k_attn_0_4_output == "<s>"

    attn_2_4_pattern = select_closest(attn_0_4_outputs, attn_0_4_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, mlp_0_3_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_4_output, attn_1_6_output):
        if attn_0_4_output in {"0"}:
            return attn_1_6_output == 4
        elif attn_0_4_output in {"1"}:
            return attn_1_6_output == 14
        elif attn_0_4_output in {"3", "2", "4"}:
            return attn_1_6_output == 6
        elif attn_0_4_output in {"5"}:
            return attn_1_6_output == 15
        elif attn_0_4_output in {"<s>"}:
            return attn_1_6_output == 2

    attn_2_5_pattern = select_closest(attn_1_6_outputs, attn_0_4_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_6_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(position, mlp_0_1_output):
        if position in {0}:
            return mlp_0_1_output == 8
        elif position in {1}:
            return mlp_0_1_output == 4
        elif position in {2}:
            return mlp_0_1_output == 1
        elif position in {3, 4, 5, 6}:
            return mlp_0_1_output == 6
        elif position in {7}:
            return mlp_0_1_output == 0
        elif position in {8}:
            return mlp_0_1_output == 14
        elif position in {9, 10, 13}:
            return mlp_0_1_output == 3
        elif position in {11, 12, 14, 15}:
            return mlp_0_1_output == 12

    attn_2_6_pattern = select_closest(mlp_0_1_outputs, positions, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, mlp_0_1_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_3_output, mlp_0_1_output):
        if attn_0_3_output in {"0"}:
            return mlp_0_1_output == 9
        elif attn_0_3_output in {"1"}:
            return mlp_0_1_output == 2
        elif attn_0_3_output in {"2"}:
            return mlp_0_1_output == 11
        elif attn_0_3_output in {"3", "4"}:
            return mlp_0_1_output == 1
        elif attn_0_3_output in {"5"}:
            return mlp_0_1_output == 7
        elif attn_0_3_output in {"<s>"}:
            return mlp_0_1_output == 14

    attn_2_7_pattern = select_closest(mlp_0_1_outputs, attn_0_3_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, mlp_1_3_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_7_output, attn_1_3_output):
        if attn_0_7_output in {"3", "0", "1", "4"}:
            return attn_1_3_output == ""
        elif attn_0_7_output in {"2"}:
            return attn_1_3_output == "<pad>"
        elif attn_0_7_output in {"5"}:
            return attn_1_3_output == "5"
        elif attn_0_7_output in {"<s>"}:
            return attn_1_3_output == "<s>"

    num_attn_2_0_pattern = select(attn_1_3_outputs, attn_0_7_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_1_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_7_output, attn_0_1_output):
        if attn_0_7_output in {"<s>", "0", "4", "3", "1"}:
            return attn_0_1_output == ""
        elif attn_0_7_output in {"2"}:
            return attn_0_1_output == "2"
        elif attn_0_7_output in {"5"}:
            return attn_0_1_output == "<pad>"

    num_attn_2_1_pattern = select(attn_0_1_outputs, attn_0_7_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_6_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_2_output, attn_0_5_output):
        if attn_0_2_output in {"5", "0", "2", "4", "3"}:
            return attn_0_5_output == "1"
        elif attn_0_2_output in {"1"}:
            return attn_0_5_output == "4"
        elif attn_0_2_output in {"<s>"}:
            return attn_0_5_output == "<s>"

    num_attn_2_2_pattern = select(attn_0_5_outputs, attn_0_2_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_7_output, attn_1_3_output):
        if attn_0_7_output in {"<s>", "5", "0", "4", "1"}:
            return attn_1_3_output == ""
        elif attn_0_7_output in {"2"}:
            return attn_1_3_output == "<s>"
        elif attn_0_7_output in {"3"}:
            return attn_1_3_output == "3"

    num_attn_2_3_pattern = select(attn_1_3_outputs, attn_0_7_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_7_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_0_1_output, attn_0_2_output):
        if attn_0_1_output in {"3", "5", "0", "1"}:
            return attn_0_2_output == ""
        elif attn_0_1_output in {"2"}:
            return attn_0_2_output == "<s>"
        elif attn_0_1_output in {"4"}:
            return attn_0_2_output == "4"
        elif attn_0_1_output in {"<s>"}:
            return attn_0_2_output == "<pad>"

    num_attn_2_4_pattern = select(attn_0_2_outputs, attn_0_1_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_2_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_0_7_output, attn_0_5_output):
        if attn_0_7_output in {"<s>", "0"}:
            return attn_0_5_output == "<s>"
        elif attn_0_7_output in {"1"}:
            return attn_0_5_output == "1"
        elif attn_0_7_output in {"2"}:
            return attn_0_5_output == "<pad>"
        elif attn_0_7_output in {"3"}:
            return attn_0_5_output == ""
        elif attn_0_7_output in {"4"}:
            return attn_0_5_output == "4"
        elif attn_0_7_output in {"5"}:
            return attn_0_5_output == "5"

    num_attn_2_5_pattern = select(attn_0_5_outputs, attn_0_7_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_2_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(position, token):
        if position in {0, 2, 3, 7, 9, 11, 14, 15}:
            return token == ""
        elif position in {1}:
            return token == "2"
        elif position in {10, 4, 5, 6}:
            return token == "3"
        elif position in {8}:
            return token == "<pad>"
        elif position in {12}:
            return token == "<s>"
        elif position in {13}:
            return token == "4"

    num_attn_2_6_pattern = select(tokens, positions, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, ones)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_0_7_output, attn_0_4_output):
        if attn_0_7_output in {"0"}:
            return attn_0_4_output == "0"
        elif attn_0_7_output in {"1", "2"}:
            return attn_0_4_output == "<pad>"
        elif attn_0_7_output in {"3"}:
            return attn_0_4_output == "3"
        elif attn_0_7_output in {"4"}:
            return attn_0_4_output == "4"
        elif attn_0_7_output in {"<s>", "5"}:
            return attn_0_4_output == ""

    num_attn_2_7_pattern = select(attn_0_4_outputs, attn_0_7_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_3_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(mlp_0_3_output, num_mlp_1_1_output):
        key = (mlp_0_3_output, num_mlp_1_1_output)
        return 12

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(mlp_0_3_outputs, num_mlp_1_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_1_output, attn_1_7_output):
        key = (attn_2_1_output, attn_1_7_output)
        if key in {
            (0, "1"),
            (0, "3"),
            (0, "4"),
            (3, "4"),
            (9, "1"),
            (9, "2"),
            (9, "3"),
            (9, "4"),
            (9, "5"),
            (9, "<s>"),
            (11, "3"),
            (11, "4"),
            (12, "3"),
            (12, "4"),
            (14, "1"),
            (14, "3"),
            (14, "4"),
            (15, "1"),
            (15, "3"),
            (15, "4"),
        }:
            return 8
        return 15

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_1_7_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(attn_2_1_output, num_mlp_1_0_output):
        key = (attn_2_1_output, num_mlp_1_0_output)
        return 2

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(attn_2_1_outputs, num_mlp_1_0_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(attn_2_7_output, mlp_0_2_output):
        key = (attn_2_7_output, mlp_0_2_output)
        return 13

    mlp_2_3_outputs = [
        mlp_2_3(k0, k1) for k0, k1 in zip(attn_2_7_outputs, mlp_0_2_outputs)
    ]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_1_output, num_attn_1_4_output):
        key = (num_attn_0_1_output, num_attn_1_4_output)
        if key in {(0, 0), (0, 1), (1, 0)}:
            return 13
        elif key in {(0, 2)}:
            return 9
        return 12

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_1_output, num_attn_1_2_output):
        key = (num_attn_1_1_output, num_attn_1_2_output)
        return 0

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_1_4_output, num_attn_0_0_output):
        key = (num_attn_1_4_output, num_attn_0_0_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 16),
            (0, 17),
            (0, 18),
            (0, 19),
            (0, 20),
            (0, 21),
            (0, 22),
            (0, 23),
            (0, 24),
            (0, 25),
            (0, 26),
            (0, 27),
            (0, 28),
            (0, 29),
            (0, 30),
            (0, 31),
            (0, 32),
            (0, 33),
            (0, 34),
            (0, 35),
            (0, 36),
            (0, 37),
            (0, 38),
            (0, 39),
            (0, 40),
            (0, 41),
            (0, 42),
            (0, 43),
            (0, 44),
            (0, 45),
            (0, 46),
            (0, 47),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (1, 21),
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
            (1, 30),
            (1, 31),
            (1, 32),
            (1, 33),
            (1, 34),
            (1, 35),
            (1, 36),
            (1, 37),
            (1, 38),
            (1, 39),
            (1, 40),
            (1, 41),
            (1, 42),
            (1, 43),
            (1, 44),
            (1, 45),
            (1, 46),
            (1, 47),
            (2, 41),
            (2, 42),
            (2, 43),
            (2, 44),
            (2, 45),
            (2, 46),
            (2, 47),
            (3, 0),
            (4, 0),
            (5, 0),
            (5, 1),
            (6, 0),
            (6, 1),
            (7, 0),
            (7, 1),
            (7, 2),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 3),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (12, 4),
            (12, 5),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (13, 4),
            (13, 5),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (14, 4),
            (14, 5),
            (14, 6),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 5),
            (15, 6),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (16, 4),
            (16, 5),
            (16, 6),
            (16, 7),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 4),
            (17, 5),
            (17, 6),
            (17, 7),
            (17, 8),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (18, 4),
            (18, 5),
            (18, 6),
            (18, 7),
            (18, 8),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 4),
            (19, 5),
            (19, 6),
            (19, 7),
            (19, 8),
            (19, 9),
            (20, 0),
            (20, 1),
            (20, 2),
            (20, 3),
            (20, 4),
            (20, 5),
            (20, 6),
            (20, 7),
            (20, 8),
            (20, 9),
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (21, 4),
            (21, 5),
            (21, 6),
            (21, 7),
            (21, 8),
            (21, 9),
            (21, 10),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (22, 4),
            (22, 5),
            (22, 6),
            (22, 7),
            (22, 8),
            (22, 9),
            (22, 10),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (23, 4),
            (23, 5),
            (23, 6),
            (23, 7),
            (23, 8),
            (23, 9),
            (23, 10),
            (23, 11),
            (24, 0),
            (24, 1),
            (24, 2),
            (24, 3),
            (24, 4),
            (24, 5),
            (24, 6),
            (24, 7),
            (24, 8),
            (24, 9),
            (24, 10),
            (24, 11),
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
            (25, 4),
            (25, 5),
            (25, 6),
            (25, 7),
            (25, 8),
            (25, 9),
            (25, 10),
            (25, 11),
            (25, 12),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (26, 4),
            (26, 5),
            (26, 6),
            (26, 7),
            (26, 8),
            (26, 9),
            (26, 10),
            (26, 11),
            (26, 12),
            (26, 13),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (27, 4),
            (27, 5),
            (27, 6),
            (27, 7),
            (27, 8),
            (27, 9),
            (27, 10),
            (27, 11),
            (27, 12),
            (27, 13),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (28, 4),
            (28, 5),
            (28, 6),
            (28, 7),
            (28, 8),
            (28, 9),
            (28, 10),
            (28, 11),
            (28, 12),
            (28, 13),
            (28, 14),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (29, 4),
            (29, 5),
            (29, 6),
            (29, 7),
            (29, 8),
            (29, 9),
            (29, 10),
            (29, 11),
            (29, 12),
            (29, 13),
            (29, 14),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (30, 4),
            (30, 5),
            (30, 6),
            (30, 7),
            (30, 8),
            (30, 9),
            (30, 10),
            (30, 11),
            (30, 12),
            (30, 13),
            (30, 14),
            (30, 15),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (31, 4),
            (31, 5),
            (31, 6),
            (31, 7),
            (31, 8),
            (31, 9),
            (31, 10),
            (31, 11),
            (31, 12),
            (31, 13),
            (31, 14),
            (31, 15),
            (32, 0),
            (32, 1),
            (32, 2),
            (32, 3),
            (32, 4),
            (32, 5),
            (32, 6),
            (32, 7),
            (32, 8),
            (32, 9),
            (32, 10),
            (32, 11),
            (32, 12),
            (32, 13),
            (32, 14),
            (32, 15),
            (32, 16),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (33, 4),
            (33, 5),
            (33, 6),
            (33, 7),
            (33, 8),
            (33, 9),
            (33, 10),
            (33, 11),
            (33, 12),
            (33, 13),
            (33, 14),
            (33, 15),
            (33, 16),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (34, 4),
            (34, 5),
            (34, 6),
            (34, 7),
            (34, 8),
            (34, 9),
            (34, 10),
            (34, 11),
            (34, 12),
            (34, 13),
            (34, 14),
            (34, 15),
            (34, 16),
            (34, 17),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (35, 4),
            (35, 5),
            (35, 6),
            (35, 7),
            (35, 8),
            (35, 9),
            (35, 10),
            (35, 11),
            (35, 12),
            (35, 13),
            (35, 14),
            (35, 15),
            (35, 16),
            (35, 17),
            (35, 18),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (36, 4),
            (36, 5),
            (36, 6),
            (36, 7),
            (36, 8),
            (36, 9),
            (36, 10),
            (36, 11),
            (36, 12),
            (36, 13),
            (36, 14),
            (36, 15),
            (36, 16),
            (36, 17),
            (36, 18),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (37, 4),
            (37, 5),
            (37, 6),
            (37, 7),
            (37, 8),
            (37, 9),
            (37, 10),
            (37, 11),
            (37, 12),
            (37, 13),
            (37, 14),
            (37, 15),
            (37, 16),
            (37, 17),
            (37, 18),
            (37, 19),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (38, 4),
            (38, 5),
            (38, 6),
            (38, 7),
            (38, 8),
            (38, 9),
            (38, 10),
            (38, 11),
            (38, 12),
            (38, 13),
            (38, 14),
            (38, 15),
            (38, 16),
            (38, 17),
            (38, 18),
            (38, 19),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (39, 5),
            (39, 6),
            (39, 7),
            (39, 8),
            (39, 9),
            (39, 10),
            (39, 11),
            (39, 12),
            (39, 13),
            (39, 14),
            (39, 15),
            (39, 16),
            (39, 17),
            (39, 18),
            (39, 19),
            (39, 20),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (40, 4),
            (40, 5),
            (40, 6),
            (40, 7),
            (40, 8),
            (40, 9),
            (40, 10),
            (40, 11),
            (40, 12),
            (40, 13),
            (40, 14),
            (40, 15),
            (40, 16),
            (40, 17),
            (40, 18),
            (40, 19),
            (40, 20),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (41, 4),
            (41, 5),
            (41, 6),
            (41, 7),
            (41, 8),
            (41, 9),
            (41, 10),
            (41, 11),
            (41, 12),
            (41, 13),
            (41, 14),
            (41, 15),
            (41, 16),
            (41, 17),
            (41, 18),
            (41, 19),
            (41, 20),
            (41, 21),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (42, 4),
            (42, 5),
            (42, 6),
            (42, 7),
            (42, 8),
            (42, 9),
            (42, 10),
            (42, 11),
            (42, 12),
            (42, 13),
            (42, 14),
            (42, 15),
            (42, 16),
            (42, 17),
            (42, 18),
            (42, 19),
            (42, 20),
            (42, 21),
            (43, 0),
            (43, 1),
            (43, 2),
            (43, 3),
            (43, 4),
            (43, 5),
            (43, 6),
            (43, 7),
            (43, 8),
            (43, 9),
            (43, 10),
            (43, 11),
            (43, 12),
            (43, 13),
            (43, 14),
            (43, 15),
            (43, 16),
            (43, 17),
            (43, 18),
            (43, 19),
            (43, 20),
            (43, 21),
            (43, 22),
            (44, 0),
            (44, 1),
            (44, 2),
            (44, 3),
            (44, 4),
            (44, 5),
            (44, 6),
            (44, 7),
            (44, 8),
            (44, 9),
            (44, 10),
            (44, 11),
            (44, 12),
            (44, 13),
            (44, 14),
            (44, 15),
            (44, 16),
            (44, 17),
            (44, 18),
            (44, 19),
            (44, 20),
            (44, 21),
            (44, 22),
            (44, 23),
            (45, 0),
            (45, 1),
            (45, 2),
            (45, 3),
            (45, 4),
            (45, 5),
            (45, 6),
            (45, 7),
            (45, 8),
            (45, 9),
            (45, 10),
            (45, 11),
            (45, 12),
            (45, 13),
            (45, 14),
            (45, 15),
            (45, 16),
            (45, 17),
            (45, 18),
            (45, 19),
            (45, 20),
            (45, 21),
            (45, 22),
            (45, 23),
            (46, 0),
            (46, 1),
            (46, 2),
            (46, 3),
            (46, 4),
            (46, 5),
            (46, 6),
            (46, 7),
            (46, 8),
            (46, 9),
            (46, 10),
            (46, 11),
            (46, 12),
            (46, 13),
            (46, 14),
            (46, 15),
            (46, 16),
            (46, 17),
            (46, 18),
            (46, 19),
            (46, 20),
            (46, 21),
            (46, 22),
            (46, 23),
            (46, 24),
            (47, 0),
            (47, 1),
            (47, 2),
            (47, 3),
            (47, 4),
            (47, 5),
            (47, 6),
            (47, 7),
            (47, 8),
            (47, 9),
            (47, 10),
            (47, 11),
            (47, 12),
            (47, 13),
            (47, 14),
            (47, 15),
            (47, 16),
            (47, 17),
            (47, 18),
            (47, 19),
            (47, 20),
            (47, 21),
            (47, 22),
            (47, 23),
            (47, 24),
        }:
            return 11
        return 3

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_6_output, num_attn_1_1_output):
        key = (num_attn_1_6_output, num_attn_1_1_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 16),
            (0, 17),
            (0, 18),
            (0, 19),
            (0, 20),
            (0, 21),
            (0, 22),
            (0, 23),
            (0, 24),
            (0, 25),
            (0, 26),
            (0, 27),
            (0, 28),
            (0, 29),
            (0, 30),
            (0, 31),
            (0, 32),
            (0, 33),
            (0, 34),
            (0, 35),
            (0, 36),
            (0, 37),
            (0, 38),
            (0, 39),
            (0, 40),
            (0, 41),
            (0, 42),
            (0, 43),
            (0, 44),
            (0, 45),
            (0, 46),
            (0, 47),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (1, 21),
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
            (1, 30),
            (1, 31),
            (1, 32),
            (1, 33),
            (1, 34),
            (1, 35),
            (1, 36),
            (1, 37),
            (1, 38),
            (1, 39),
            (1, 40),
            (1, 41),
            (1, 42),
            (1, 43),
            (1, 44),
            (1, 45),
            (1, 46),
            (1, 47),
            (2, 24),
            (2, 25),
            (2, 26),
            (2, 27),
            (2, 28),
            (2, 29),
            (2, 30),
            (2, 31),
            (2, 32),
            (2, 33),
            (2, 34),
            (2, 35),
            (2, 36),
            (2, 37),
            (2, 38),
            (2, 39),
            (2, 40),
            (2, 41),
            (2, 42),
            (2, 43),
            (2, 44),
            (2, 45),
            (2, 46),
            (2, 47),
            (3, 36),
            (3, 37),
            (3, 38),
            (3, 39),
            (3, 40),
            (3, 41),
            (3, 42),
            (3, 43),
            (3, 44),
            (3, 45),
            (3, 46),
            (3, 47),
        }:
            return 9
        elif key in {
            (31, 0),
            (32, 0),
            (33, 0),
            (34, 0),
            (35, 0),
            (36, 0),
            (37, 0),
            (38, 0),
            (39, 0),
            (40, 0),
            (41, 0),
            (42, 0),
            (43, 0),
            (44, 0),
            (45, 0),
            (46, 0),
            (47, 0),
        }:
            return 13
        return 14

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_1_6_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_2_3_output_scores = classifier_weights.loc[
        [("num_mlp_2_3_outputs", str(v)) for v in num_mlp_2_3_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                attn_0_4_output_scores,
                attn_0_5_output_scores,
                attn_0_6_output_scores,
                attn_0_7_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                mlp_0_2_output_scores,
                mlp_0_3_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                num_mlp_0_2_output_scores,
                num_mlp_0_3_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                attn_1_4_output_scores,
                attn_1_5_output_scores,
                attn_1_6_output_scores,
                attn_1_7_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                mlp_1_2_output_scores,
                mlp_1_3_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                num_mlp_1_2_output_scores,
                num_mlp_1_3_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                attn_2_4_output_scores,
                attn_2_5_output_scores,
                attn_2_6_output_scores,
                attn_2_7_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                mlp_2_2_output_scores,
                mlp_2_3_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                num_mlp_2_2_output_scores,
                num_mlp_2_3_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_0_4_output_scores,
                num_attn_0_5_output_scores,
                num_attn_0_6_output_scores,
                num_attn_0_7_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_1_4_output_scores,
                num_attn_1_5_output_scores,
                num_attn_1_6_output_scores,
                num_attn_1_7_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
                num_attn_2_4_output_scores,
                num_attn_2_5_output_scores,
                num_attn_2_6_output_scores,
                num_attn_2_7_output_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    predictions = classes[logits.argmax(-1)]
    if tokens[0] == "<s>":
        predictions[0] = "<s>"
    if tokens[-1] == "</s>":
        predictions[-1] = "</s>"
    return predictions.tolist()


print(run(["<s>", "0", "2", "5", "2", "4", "3", "5", "4", "5", "4", "5", "0", "5"]))
