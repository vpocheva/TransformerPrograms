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
        "output/rasp/double_hist/vocab16maxlen16/transformer_program/headsc4headsn4nlayers3cmlpsnmlps/s0/double_hist_weights.csv",
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
        if q_position in {0, 12, 13, 14, 15}:
            return k_position == 12
        elif q_position in {1, 2, 3, 4, 8}:
            return k_position == 4
        elif q_position in {5, 6}:
            return k_position == 6
        elif q_position in {9, 10, 7}:
            return k_position == 7
        elif q_position in {11}:
            return k_position == 11

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 7}:
            return k_position == 7
        elif q_position in {1, 3}:
            return k_position == 4
        elif q_position in {2, 10, 5, 15}:
            return k_position == 14
        elif q_position in {8, 4}:
            return k_position == 9
        elif q_position in {6}:
            return k_position == 6
        elif q_position in {9, 11, 12, 13, 14}:
            return k_position == 15

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, positions)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 7
        elif q_position in {1, 2, 3}:
            return k_position == 3
        elif q_position in {4, 15, 7}:
            return k_position == 14
        elif q_position in {5}:
            return k_position == 5
        elif q_position in {10, 6}:
            return k_position == 15
        elif q_position in {8}:
            return k_position == 8
        elif q_position in {9, 11}:
            return k_position == 9
        elif q_position in {12, 14}:
            return k_position == 12
        elif q_position in {13}:
            return k_position == 13

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 7}:
            return k_position == 7
        elif q_position in {1, 2, 3, 4, 5}:
            return k_position == 5
        elif q_position in {6}:
            return k_position == 6
        elif q_position in {8, 9, 11}:
            return k_position == 9
        elif q_position in {10}:
            return k_position == 10
        elif q_position in {12, 13}:
            return k_position == 13
        elif q_position in {14}:
            return k_position == 12
        elif q_position in {15}:
            return k_position == 14

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 14
        elif q_position in {1, 11}:
            return k_position == 10
        elif q_position in {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14}:
            return k_position == 11
        elif q_position in {15}:
            return k_position == 13

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 14
        elif q_position in {1, 2, 5, 7}:
            return k_position == 8
        elif q_position in {3, 4, 6, 11, 12}:
            return k_position == 10
        elif q_position in {8}:
            return k_position == 6
        elif q_position in {9}:
            return k_position == 5
        elif q_position in {10}:
            return k_position == 7
        elif q_position in {13, 14}:
            return k_position == 11
        elif q_position in {15}:
            return k_position == 13

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"10"}:
            return k_token == "10"
        elif q_token in {"11"}:
            return k_token == "11"
        elif q_token in {"12"}:
            return k_token == "12"
        elif q_token in {"13"}:
            return k_token == "13"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"6"}:
            return k_token == "6"
        elif q_token in {"7"}:
            return k_token == "7"
        elif q_token in {"8"}:
            return k_token == "8"
        elif q_token in {"9"}:
            return k_token == "9"
        elif q_token in {"<s>"}:
            return k_token == "<pad>"

    num_attn_0_2_pattern = select(tokens, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0, 15}:
            return k_position == 15
        elif q_position in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}:
            return k_position == 6
        elif q_position in {14}:
            return k_position == 5

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_2_output, attn_0_3_output):
        key = (attn_0_2_output, attn_0_3_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 4),
            (0, 15),
            (1, 0),
            (1, 2),
            (1, 4),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 4),
            (2, 15),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 15),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 4),
        }:
            return 0
        elif key in {
            (3, 5),
            (3, 6),
            (3, 7),
            (3, 8),
            (3, 9),
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
            (3, 14),
            (9, 3),
            (10, 3),
            (11, 3),
            (14, 3),
        }:
            return 9
        elif key in {
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
        }:
            return 12
        elif key in {
            (0, 3),
            (1, 1),
            (1, 3),
            (2, 3),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 15),
            (15, 3),
        }:
            return 11
        return 7

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_3_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_0_output, attn_0_1_output):
        key = (attn_0_0_output, attn_0_1_output)
        if key in {
            (0, 5),
            (0, 6),
            (0, 8),
            (1, 1),
            (1, 5),
            (1, 6),
            (1, 8),
            (2, 5),
            (2, 6),
            (2, 8),
            (3, 5),
            (3, 6),
            (3, 8),
            (4, 0),
            (4, 1),
            (4, 3),
            (4, 5),
            (4, 6),
            (4, 8),
            (5, 0),
            (5, 1),
            (5, 3),
            (5, 5),
            (5, 6),
            (5, 8),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (6, 6),
            (6, 8),
            (7, 8),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
        }:
            return 4
        elif key in {
            (0, 4),
            (0, 7),
            (1, 2),
            (1, 4),
            (1, 7),
            (2, 4),
            (2, 7),
            (3, 4),
            (3, 7),
            (4, 2),
            (4, 4),
            (4, 7),
            (5, 2),
            (5, 4),
            (5, 7),
            (6, 7),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 6),
            (7, 7),
        }:
            return 7
        elif key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (15, 0),
        }:
            return 1
        return 15

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_1_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output):
        key = num_attn_0_2_output
        if key in {0, 1}:
            return 8
        return 3

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 0

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 7
        elif q_position in {1, 2, 3, 4, 5, 6, 7, 11}:
            return k_position == 8
        elif q_position in {8, 9}:
            return k_position == 10
        elif q_position in {10, 12}:
            return k_position == 9
        elif q_position in {13}:
            return k_position == 2
        elif q_position in {14}:
            return k_position == 4
        elif q_position in {15}:
            return k_position == 3

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_3_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 4
        elif q_position in {3, 12, 6}:
            return k_position == 1
        elif q_position in {10, 4}:
            return k_position == 9
        elif q_position in {11, 5, 15}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8, 14}:
            return k_position == 11
        elif q_position in {9, 13}:
            return k_position == 15

    attn_1_1_pattern = select_closest(positions, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, num_mlp_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 5
        elif q_position in {1, 2, 7}:
            return k_position == 3
        elif q_position in {3, 4, 12, 6}:
            return k_position == 2
        elif q_position in {8, 9, 5, 15}:
            return k_position == 7
        elif q_position in {10}:
            return k_position == 1
        elif q_position in {11}:
            return k_position == 6
        elif q_position in {13, 14}:
            return k_position == 9

    attn_1_2_pattern = select_closest(positions, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, num_mlp_0_0_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 6
        elif q_position in {1, 11, 6}:
            return k_position == 9
        elif q_position in {2, 5, 7}:
            return k_position == 8
        elif q_position in {3, 12}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {8}:
            return k_position == 10
        elif q_position in {9}:
            return k_position == 13
        elif q_position in {10}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 7
        elif q_position in {14}:
            return k_position == 1
        elif q_position in {15}:
            return k_position == 3

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, num_mlp_0_0_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_0_output, num_mlp_0_0_output):
        if attn_0_0_output in {0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 14}:
            return num_mlp_0_0_output == 11
        elif attn_0_0_output in {5, 11, 12, 13, 15}:
            return num_mlp_0_0_output == 8

    num_attn_1_0_pattern = select(
        num_mlp_0_0_outputs, attn_0_0_outputs, num_predicate_1_0
    )
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_3_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_0_output, num_mlp_0_0_output):
        if attn_0_0_output in {0, 1, 2, 3, 4, 6, 7, 8}:
            return num_mlp_0_0_output == 8
        elif attn_0_0_output in {5}:
            return num_mlp_0_0_output == 11
        elif attn_0_0_output in {9, 13}:
            return num_mlp_0_0_output == 12
        elif attn_0_0_output in {10, 11, 12, 14}:
            return num_mlp_0_0_output == 13
        elif attn_0_0_output in {15}:
            return num_mlp_0_0_output == 14

    num_attn_1_1_pattern = select(
        num_mlp_0_0_outputs, attn_0_0_outputs, num_predicate_1_1
    )
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_3_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_0_output, num_mlp_0_0_output):
        if attn_0_0_output in {0, 1, 2, 3, 4, 5, 6, 7, 8}:
            return num_mlp_0_0_output == 3
        elif attn_0_0_output in {9, 10, 12, 14}:
            return num_mlp_0_0_output == 7
        elif attn_0_0_output in {11}:
            return num_mlp_0_0_output == 5
        elif attn_0_0_output in {13, 15}:
            return num_mlp_0_0_output == 4

    num_attn_1_2_pattern = select(
        num_mlp_0_0_outputs, attn_0_0_outputs, num_predicate_1_2
    )
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_2_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_0_output, num_mlp_0_0_output):
        if attn_0_0_output in {0, 5, 11, 12, 13, 15}:
            return num_mlp_0_0_output == 11
        elif attn_0_0_output in {1, 2, 3, 4, 6, 7, 9, 10, 14}:
            return num_mlp_0_0_output == 8
        elif attn_0_0_output in {8}:
            return num_mlp_0_0_output == 13

    num_attn_1_3_pattern = select(
        num_mlp_0_0_outputs, attn_0_0_outputs, num_predicate_1_3
    )
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_1_output, num_mlp_0_0_output):
        key = (attn_1_1_output, num_mlp_0_0_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 7),
            (0, 9),
            (0, 10),
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 14),
            (2, 15),
            (4, 3),
            (5, 2),
            (5, 3),
            (5, 4),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (6, 6),
            (6, 7),
            (6, 9),
            (6, 10),
            (6, 14),
            (6, 15),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 4),
            (7, 5),
            (7, 7),
            (7, 9),
            (7, 10),
            (7, 12),
            (7, 14),
            (7, 15),
            (8, 3),
            (9, 3),
            (9, 4),
            (10, 3),
            (10, 4),
            (11, 3),
            (14, 3),
            (14, 4),
        }:
            return 14
        elif key in {
            (0, 6),
            (0, 11),
            (4, 4),
            (7, 6),
            (7, 11),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 4),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
            (8, 9),
            (8, 10),
            (8, 11),
            (8, 12),
            (8, 13),
            (8, 14),
            (8, 15),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 4),
            (11, 5),
            (11, 6),
            (11, 7),
            (11, 8),
            (11, 9),
            (11, 10),
            (11, 11),
            (11, 12),
            (11, 13),
            (11, 14),
            (11, 15),
        }:
            return 8
        elif key in {
            (0, 8),
            (1, 6),
            (1, 8),
            (1, 11),
            (2, 11),
            (4, 1),
            (4, 2),
            (4, 5),
            (4, 6),
            (4, 7),
            (4, 8),
            (4, 11),
            (6, 8),
            (6, 11),
            (7, 8),
        }:
            return 1
        return 7

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_1_outputs, num_mlp_0_0_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_2_output, attn_1_3_output):
        key = (attn_1_2_output, attn_1_3_output)
        if key in {(5, 5)}:
            return 0
        return 15

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_3_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
        return 8

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_3_output, num_attn_0_1_output):
        key = (num_attn_0_3_output, num_attn_0_1_output)
        return 11

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0, 12}:
            return k_position == 10
        elif q_position in {8, 1, 3}:
            return k_position == 7
        elif q_position in {2, 10, 7}:
            return k_position == 5
        elif q_position in {4}:
            return k_position == 6
        elif q_position in {5}:
            return k_position == 13
        elif q_position in {13, 6}:
            return k_position == 8
        elif q_position in {9, 15}:
            return k_position == 2
        elif q_position in {11}:
            return k_position == 3
        elif q_position in {14}:
            return k_position == 11

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_1_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 7
        elif q_position in {1, 3, 5, 6, 8, 9, 10}:
            return k_position == 13
        elif q_position in {2, 13, 14}:
            return k_position == 12
        elif q_position in {4, 12, 15}:
            return k_position == 11
        elif q_position in {11, 7}:
            return k_position == 9

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_0_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(position, num_mlp_1_1_output):
        if position in {0, 14, 15}:
            return num_mlp_1_1_output == 5
        elif position in {1, 5, 9}:
            return num_mlp_1_1_output == 11
        elif position in {2, 10}:
            return num_mlp_1_1_output == 12
        elif position in {3, 12}:
            return num_mlp_1_1_output == 14
        elif position in {4, 7}:
            return num_mlp_1_1_output == 13
        elif position in {13, 6}:
            return num_mlp_1_1_output == 10
        elif position in {8}:
            return num_mlp_1_1_output == 7
        elif position in {11}:
            return num_mlp_1_1_output == 15

    attn_2_2_pattern = select_closest(num_mlp_1_1_outputs, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_2_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0, 8, 15}:
            return k_position == 10
        elif q_position in {1, 4, 13}:
            return k_position == 9
        elif q_position in {2, 7}:
            return k_position == 12
        elif q_position in {9, 3}:
            return k_position == 14
        elif q_position in {5}:
            return k_position == 1
        elif q_position in {10, 6, 14}:
            return k_position == 5
        elif q_position in {11}:
            return k_position == 2
        elif q_position in {12}:
            return k_position == 6

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_0_output, position):
        if attn_1_0_output in {0}:
            return position == 14
        elif attn_1_0_output in {1, 5, 7}:
            return position == 10
        elif attn_1_0_output in {2, 3, 4, 6}:
            return position == 9
        elif attn_1_0_output in {8, 9, 10, 14}:
            return position == 5
        elif attn_1_0_output in {11}:
            return position == 1
        elif attn_1_0_output in {12}:
            return position == 7
        elif attn_1_0_output in {13}:
            return position == 13
        elif attn_1_0_output in {15}:
            return position == 2

    num_attn_2_0_pattern = select(positions, attn_1_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(num_mlp_0_1_output, mlp_1_1_output):
        if num_mlp_0_1_output in {0, 15}:
            return mlp_1_1_output == 2
        elif num_mlp_0_1_output in {1, 5, 9}:
            return mlp_1_1_output == 12
        elif num_mlp_0_1_output in {2, 13, 7}:
            return mlp_1_1_output == 3
        elif num_mlp_0_1_output in {3, 4}:
            return mlp_1_1_output == 11
        elif num_mlp_0_1_output in {10, 11, 6}:
            return mlp_1_1_output == 14
        elif num_mlp_0_1_output in {8}:
            return mlp_1_1_output == 13
        elif num_mlp_0_1_output in {12}:
            return mlp_1_1_output == 6
        elif num_mlp_0_1_output in {14}:
            return mlp_1_1_output == 1

    num_attn_2_1_pattern = select(
        mlp_1_1_outputs, num_mlp_0_1_outputs, num_predicate_2_1
    )
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(q_attn_0_0_output, k_attn_0_0_output):
        if q_attn_0_0_output in {0}:
            return k_attn_0_0_output == 11
        elif q_attn_0_0_output in {1, 2, 3, 4, 5, 6, 7, 8}:
            return k_attn_0_0_output == 12
        elif q_attn_0_0_output in {9, 10}:
            return k_attn_0_0_output == 13
        elif q_attn_0_0_output in {11}:
            return k_attn_0_0_output == 14
        elif q_attn_0_0_output in {12, 14}:
            return k_attn_0_0_output == 2
        elif q_attn_0_0_output in {13}:
            return k_attn_0_0_output == 8
        elif q_attn_0_0_output in {15}:
            return k_attn_0_0_output == 9

    num_attn_2_2_pattern = select(attn_0_0_outputs, attn_0_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_0_output, position):
        if attn_1_0_output in {0, 2}:
            return position == 1
        elif attn_1_0_output in {1, 14, 6, 7}:
            return position == 2
        elif attn_1_0_output in {3, 4, 9, 10, 15}:
            return position == 4
        elif attn_1_0_output in {13, 5}:
            return position == 3
        elif attn_1_0_output in {8}:
            return position == 9
        elif attn_1_0_output in {11}:
            return position == 11
        elif attn_1_0_output in {12}:
            return position == 12

    num_attn_2_3_pattern = select(positions, attn_1_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_3_output, attn_2_1_output):
        key = (attn_2_3_output, attn_2_1_output)
        if key in {
            (0, 2),
            (0, 4),
            (0, 6),
            (0, 7),
            (0, 8),
            (1, 0),
            (1, 2),
            (1, 4),
            (1, 6),
            (1, 7),
            (1, 8),
            (2, 0),
            (2, 2),
            (2, 4),
            (2, 6),
            (2, 7),
            (2, 8),
            (6, 0),
            (6, 2),
            (6, 4),
            (6, 6),
            (6, 7),
            (6, 8),
            (7, 0),
            (7, 2),
            (7, 4),
            (7, 6),
            (7, 7),
            (7, 8),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
            (8, 9),
            (8, 10),
            (8, 11),
            (8, 12),
            (8, 13),
            (8, 14),
            (8, 15),
            (9, 0),
            (9, 1),
            (9, 2),
            (9, 4),
            (9, 5),
            (9, 6),
            (9, 7),
            (9, 8),
            (9, 13),
            (9, 14),
            (9, 15),
            (10, 0),
            (10, 2),
            (10, 4),
            (10, 6),
            (10, 7),
            (10, 8),
            (11, 0),
            (11, 2),
            (11, 4),
            (11, 6),
            (11, 7),
            (11, 8),
            (12, 0),
            (12, 2),
            (12, 4),
            (12, 6),
            (12, 7),
            (12, 8),
            (13, 2),
            (13, 4),
            (13, 6),
            (13, 7),
            (13, 8),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 4),
            (14, 6),
            (14, 7),
            (14, 8),
            (14, 14),
            (14, 15),
            (15, 0),
            (15, 2),
            (15, 4),
            (15, 6),
            (15, 7),
            (15, 8),
            (15, 15),
        }:
            return 3
        elif key in {(9, 11), (13, 0), (14, 13), (15, 1), (15, 13), (15, 14)}:
            return 12
        return 7

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_3_outputs, attn_2_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_1_output, attn_2_2_output):
        key = (attn_2_1_output, attn_2_2_output)
        if key in {
            (3, 12),
            (5, 0),
            (5, 3),
            (5, 10),
            (5, 12),
            (9, 0),
            (9, 3),
            (9, 4),
            (9, 6),
            (9, 10),
            (9, 12),
            (9, 15),
            (10, 0),
            (10, 3),
            (10, 4),
            (10, 6),
            (10, 10),
            (10, 12),
            (10, 15),
            (11, 3),
            (11, 12),
            (12, 3),
            (12, 10),
            (12, 12),
            (13, 0),
            (13, 3),
            (13, 10),
            (13, 12),
            (14, 0),
            (14, 3),
            (14, 4),
            (14, 6),
            (14, 10),
            (14, 12),
            (15, 0),
            (15, 3),
            (15, 4),
            (15, 6),
            (15, 10),
            (15, 12),
            (15, 15),
        }:
            return 15
        elif key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 8),
            (0, 11),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 8),
            (4, 11),
        }:
            return 3
        elif key in {(3, 3), (3, 4), (3, 6), (3, 10), (10, 1), (10, 2)}:
            return 6
        elif key in {(3, 1), (3, 2), (3, 5), (3, 7), (3, 9)}:
            return 0
        elif key in {(3, 0)}:
            return 14
        return 8

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_2_2_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_1_output, num_attn_1_2_output):
        key = (num_attn_0_1_output, num_attn_1_2_output)
        if key in {(0, 0)}:
            return 10
        return 9

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_3_output, num_attn_2_2_output):
        key = (num_attn_2_3_output, num_attn_2_2_output)
        if key in {
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
            (1, 45),
            (1, 46),
            (1, 47),
        }:
            return 13
        elif key in {
            (0, 39),
            (0, 40),
            (0, 41),
            (0, 42),
            (0, 43),
            (0, 44),
            (0, 45),
            (0, 46),
            (0, 47),
        }:
            return 7
        return 6

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_2_2_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
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
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
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


print(run(["<s>", "1", "8", "1", "2", "9", "0", "5", "8", "2", "11", "10", "9"]))
