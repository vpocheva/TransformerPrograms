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
        "output/rasp/most_freq/vocab8maxlen8/transformer_program/headsc8headsn8nlayers3cmlpsnmlps/s0/most_freq_weights.csv",
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
    def predicate_0_0(position, token):
        if position in {0, 6, 7}:
            return token == "5"
        elif position in {1, 3, 4, 5}:
            return token == "1"
        elif position in {2}:
            return token == "2"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 3}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 3

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"0", "4"}:
            return k_token == "4"
        elif q_token in {"1", "<s>"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"5"}:
            return k_token == "5"

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 3}:
            return k_position == 4
        elif q_position in {1, 2}:
            return k_position == 1
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {5, 7}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 3, 6}:
            return k_position == 5
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2, 4}:
            return k_position == 3
        elif q_position in {5, 7}:
            return k_position == 4

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {0, 1, 6, 7}:
            return token == "5"
        elif position in {2}:
            return token == "4"
        elif position in {3, 4, 5}:
            return token == "2"

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1}:
            return k_position == 0
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {3}:
            return k_position == 1
        elif q_position in {4, 7}:
            return k_position == 5
        elif q_position in {5, 6}:
            return k_position == 7

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(position, token):
        if position in {0, 1, 3}:
            return token == "1"
        elif position in {2, 4, 5}:
            return token == "2"
        elif position in {6, 7}:
            return token == "5"

    attn_0_7_pattern = select_closest(tokens, positions, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 2, 6}:
            return token == "<s>"
        elif position in {1}:
            return token == "0"
        elif position in {3, 4, 5}:
            return token == "3"
        elif position in {7}:
            return token == "<pad>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(position, token):
        if position in {0, 1}:
            return token == "1"
        elif position in {2}:
            return token == "<s>"
        elif position in {3, 4, 5, 6}:
            return token == "4"
        elif position in {7}:
            return token == "<pad>"

    num_attn_0_1_pattern = select(tokens, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(position, token):
        if position in {0, 3, 4, 5, 6, 7}:
            return token == "<s>"
        elif position in {1, 2}:
            return token == "1"

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1}:
            return token == "4"
        elif position in {2}:
            return token == "<s>"
        elif position in {3, 4, 5, 6}:
            return token == "2"
        elif position in {7}:
            return token == "5"

    num_attn_0_3_pattern = select(tokens, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(position, token):
        if position in {0, 2, 3, 4, 5, 6, 7}:
            return token == "<s>"
        elif position in {1}:
            return token == "5"

    num_attn_0_4_pattern = select(tokens, positions, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(position, token):
        if position in {0}:
            return token == "5"
        elif position in {1}:
            return token == "3"
        elif position in {2}:
            return token == "<s>"
        elif position in {3, 4, 5, 6}:
            return token == "0"
        elif position in {7}:
            return token == "2"

    num_attn_0_5_pattern = select(tokens, positions, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(position, token):
        if position in {0, 1}:
            return token == "1"
        elif position in {2}:
            return token == "<s>"
        elif position in {3, 4, 5, 6}:
            return token == "0"
        elif position in {7}:
            return token == "2"

    num_attn_0_6_pattern = select(tokens, positions, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
        if position in {0, 7}:
            return token == "1"
        elif position in {1}:
            return token == "2"
        elif position in {2}:
            return token == "<s>"
        elif position in {3, 4, 5, 6}:
            return token == "3"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_5_output):
        key = attn_0_5_output
        if key in {"2", "3"}:
            return 6
        elif key in {"<s>"}:
            return 2
        return 7

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in attn_0_5_outputs]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_2_output, position):
        key = (attn_0_2_output, position)
        if key in {
            ("0", 3),
            ("2", 0),
            ("2", 1),
            ("2", 2),
            ("2", 3),
            ("2", 5),
            ("2", 6),
            ("2", 7),
            ("5", 3),
        }:
            return 4
        elif key in {("2", 4), ("5", 5), ("5", 6), ("5", 7)}:
            return 5
        elif key in {("0", 6), ("0", 7), ("5", 0)}:
            return 7
        elif key in {("1", 3), ("3", 3), ("4", 3)}:
            return 0
        return 2

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, positions)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # mlp_0_2 #####################################################
    def mlp_0_2(position, attn_0_4_output):
        key = (position, attn_0_4_output)
        if key in {
            (2, "0"),
            (2, "1"),
            (2, "2"),
            (2, "3"),
            (2, "5"),
            (2, "<s>"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "5"),
            (3, "<s>"),
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (4, "<s>"),
        }:
            return 3
        elif key in {(1, "0"), (1, "1"), (1, "2"), (1, "3"), (1, "5"), (1, "<s>")}:
            return 5
        elif key in {(0, "4"), (1, "4"), (2, "4")}:
            return 6
        return 7

    mlp_0_2_outputs = [mlp_0_2(k0, k1) for k0, k1 in zip(positions, attn_0_4_outputs)]
    mlp_0_2_output_scores = classifier_weights.loc[
        [("mlp_0_2_outputs", str(v)) for v in mlp_0_2_outputs]
    ]

    # mlp_0_3 #####################################################
    def mlp_0_3(attn_0_7_output, position):
        key = (attn_0_7_output, position)
        if key in {
            ("0", 6),
            ("0", 7),
            ("1", 6),
            ("2", 6),
            ("2", 7),
            ("3", 3),
            ("3", 6),
            ("3", 7),
            ("4", 6),
            ("4", 7),
            ("5", 6),
            ("5", 7),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 5),
            ("<s>", 6),
            ("<s>", 7),
        }:
            return 5
        elif key in {
            ("0", 2),
            ("2", 1),
            ("2", 2),
            ("4", 1),
            ("4", 2),
            ("5", 1),
            ("5", 2),
        }:
            return 6
        elif key in {("1", 0), ("1", 3), ("1", 4), ("1", 5), ("1", 7)}:
            return 4
        elif key in {("0", 1), ("1", 1), ("3", 1), ("<s>", 1)}:
            return 2
        elif key in {("3", 2), ("<s>", 2)}:
            return 0
        elif key in {("1", 2)}:
            return 1
        return 3

    mlp_0_3_outputs = [mlp_0_3(k0, k1) for k0, k1 in zip(attn_0_7_outputs, positions)]
    mlp_0_3_output_scores = classifier_weights.loc[
        [("mlp_0_3_outputs", str(v)) for v in mlp_0_3_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
        if key in {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)}:
            return 1
        return 6

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output):
        key = num_attn_0_1_output
        if key in {0}:
            return 0
        return 2

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # num_mlp_0_2 #################################################
    def num_mlp_0_2(num_attn_0_4_output, num_attn_0_3_output):
        key = (num_attn_0_4_output, num_attn_0_3_output)
        if key in {
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (5, 2),
            (6, 2),
            (7, 2),
        }:
            return 6
        return 0

    num_mlp_0_2_outputs = [
        num_mlp_0_2(k0, k1)
        for k0, k1 in zip(num_attn_0_4_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_2_output_scores = classifier_weights.loc[
        [("num_mlp_0_2_outputs", str(v)) for v in num_mlp_0_2_outputs]
    ]

    # num_mlp_0_3 #################################################
    def num_mlp_0_3(num_attn_0_6_output):
        key = num_attn_0_6_output
        if key in {0}:
            return 7
        return 6

    num_mlp_0_3_outputs = [num_mlp_0_3(k0) for k0 in num_attn_0_6_outputs]
    num_mlp_0_3_output_scores = classifier_weights.loc[
        [("num_mlp_0_3_outputs", str(v)) for v in num_mlp_0_3_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 4}:
            return k_position == 1
        elif q_position in {1, 3, 5}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 0
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 6

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, mlp_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, mlp_0_3_output):
        if token in {"<s>", "5", "4", "0", "3"}:
            return mlp_0_3_output == 0
        elif token in {"1"}:
            return mlp_0_3_output == 6
        elif token in {"2"}:
            return mlp_0_3_output == 4

    attn_1_1_pattern = select_closest(mlp_0_3_outputs, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_3_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_1_output, position):
        if attn_0_1_output in {"5", "4", "1", "2", "0", "3"}:
            return position == 3
        elif attn_0_1_output in {"<s>"}:
            return position == 1

    attn_1_2_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1}:
            return k_position == 3
        elif q_position in {2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 0
        elif q_position in {4, 6}:
            return k_position == 5
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {7}:
            return k_position == 7

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(position, token):
        if position in {0, 2, 3, 4, 6}:
            return token == "5"
        elif position in {1}:
            return token == "2"
        elif position in {5}:
            return token == "<s>"
        elif position in {7}:
            return token == "<pad>"

    attn_1_4_pattern = select_closest(tokens, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, mlp_0_0_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(position, token):
        if position in {0, 4, 5, 6}:
            return token == "5"
        elif position in {1, 3}:
            return token == "4"
        elif position in {2, 7}:
            return token == "<s>"

    attn_1_5_pattern = select_closest(tokens, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, mlp_0_1_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(q_position, k_position):
        if q_position in {0, 1, 2, 7}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 0
        elif q_position in {4, 5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 7

    attn_1_6_pattern = select_closest(positions, positions, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, mlp_0_0_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(token, position):
        if token in {"0", "5"}:
            return position == 4
        elif token in {"4", "1", "<s>"}:
            return position == 7
        elif token in {"2"}:
            return position == 5
        elif token in {"3"}:
            return position == 6

    attn_1_7_pattern = select_closest(positions, tokens, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, positions)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_1_output, token):
        if attn_0_1_output in {"5", "1", "2", "0", "3"}:
            return token == "<s>"
        elif attn_0_1_output in {"4"}:
            return token == "4"
        elif attn_0_1_output in {"<s>"}:
            return token == "2"

    num_attn_1_0_pattern = select(tokens, attn_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_2_output, token):
        if attn_0_2_output in {"0"}:
            return token == "0"
        elif attn_0_2_output in {"1"}:
            return token == "1"
        elif attn_0_2_output in {"2"}:
            return token == "2"
        elif attn_0_2_output in {"3"}:
            return token == "3"
        elif attn_0_2_output in {"4"}:
            return token == "4"
        elif attn_0_2_output in {"<s>", "5"}:
            return token == "<s>"

    num_attn_1_1_pattern = select(tokens, attn_0_2_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_6_output, token):
        if attn_0_6_output in {"0", "1", "5", "4"}:
            return token == "3"
        elif attn_0_6_output in {"2"}:
            return token == "2"
        elif attn_0_6_output in {"<s>", "3"}:
            return token == "0"

    num_attn_1_2_pattern = select(tokens, attn_0_6_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_1_output, token):
        if attn_0_1_output in {"0"}:
            return token == "<s>"
        elif attn_0_1_output in {"<s>", "5", "4", "1", "2"}:
            return token == "0"
        elif attn_0_1_output in {"3"}:
            return token == "3"

    num_attn_1_3_pattern = select(tokens, attn_0_1_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_1_output, token):
        if attn_0_1_output in {"0", "5"}:
            return token == "1"
        elif attn_0_1_output in {"1"}:
            return token == "0"
        elif attn_0_1_output in {"4", "2"}:
            return token == "<s>"
        elif attn_0_1_output in {"3"}:
            return token == "3"
        elif attn_0_1_output in {"<s>"}:
            return token == "5"

    num_attn_1_4_pattern = select(tokens, attn_0_1_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, ones)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_2_output, token):
        if attn_0_2_output in {"0"}:
            return token == "<s>"
        elif attn_0_2_output in {"<s>", "5", "4", "1", "2", "3"}:
            return token == "0"

    num_attn_1_5_pattern = select(tokens, attn_0_2_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, ones)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(attn_0_1_output, token):
        if attn_0_1_output in {"0", "5"}:
            return token == "1"
        elif attn_0_1_output in {"1"}:
            return token == "0"
        elif attn_0_1_output in {"2"}:
            return token == "2"
        elif attn_0_1_output in {"4", "3"}:
            return token == "<s>"
        elif attn_0_1_output in {"<s>"}:
            return token == "3"

    num_attn_1_6_pattern = select(tokens, attn_0_1_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, ones)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(position, token):
        if position in {0}:
            return token == "1"
        elif position in {1}:
            return token == "4"
        elif position in {2}:
            return token == "<s>"
        elif position in {3, 4, 5, 6}:
            return token == "5"
        elif position in {7}:
            return token == "0"

    num_attn_1_7_pattern = select(tokens, positions, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, ones)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_4_output, attn_0_6_output):
        key = (attn_0_4_output, attn_0_6_output)
        if key in {
            ("0", "4"),
            ("0", "5"),
            ("0", "<s>"),
            ("2", "0"),
            ("2", "2"),
            ("2", "4"),
            ("2", "5"),
            ("2", "<s>"),
            ("3", "0"),
            ("3", "4"),
            ("3", "5"),
            ("3", "<s>"),
            ("4", "0"),
            ("4", "4"),
            ("4", "5"),
            ("4", "<s>"),
            ("5", "0"),
            ("5", "<s>"),
            ("<s>", "0"),
            ("<s>", "2"),
            ("<s>", "4"),
            ("<s>", "5"),
            ("<s>", "<s>"),
        }:
            return 0
        elif key in {("3", "2"), ("3", "3"), ("<s>", "3")}:
            return 4
        elif key in {("0", "0")}:
            return 6
        return 2

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_4_outputs, attn_0_6_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(position, attn_0_2_output):
        key = (position, attn_0_2_output)
        if key in {
            (0, "4"),
            (1, "0"),
            (1, "1"),
            (1, "2"),
            (1, "3"),
            (1, "4"),
            (1, "5"),
            (1, "<s>"),
            (2, "4"),
            (3, "0"),
            (3, "1"),
            (3, "2"),
            (3, "3"),
            (3, "4"),
            (3, "5"),
            (3, "<s>"),
        }:
            return 0
        elif key in {
            (0, "3"),
            (0, "<s>"),
            (2, "0"),
            (2, "2"),
            (2, "3"),
            (2, "<s>"),
            (4, "0"),
            (4, "1"),
            (4, "2"),
            (4, "3"),
            (4, "4"),
            (4, "5"),
            (4, "<s>"),
        }:
            return 1
        elif key in {(0, "1"), (0, "2"), (2, "1")}:
            return 6
        return 4

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(positions, attn_0_2_outputs)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # mlp_1_2 #####################################################
    def mlp_1_2(attn_1_7_output, attn_1_0_output):
        key = (attn_1_7_output, attn_1_0_output)
        return 7

    mlp_1_2_outputs = [
        mlp_1_2(k0, k1) for k0, k1 in zip(attn_1_7_outputs, attn_1_0_outputs)
    ]
    mlp_1_2_output_scores = classifier_weights.loc[
        [("mlp_1_2_outputs", str(v)) for v in mlp_1_2_outputs]
    ]

    # mlp_1_3 #####################################################
    def mlp_1_3(position, num_mlp_0_1_output):
        key = (position, num_mlp_0_1_output)
        if key in {
            (0, 2),
            (0, 6),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (2, 2),
            (2, 6),
            (2, 7),
            (3, 2),
            (3, 6),
            (4, 2),
            (4, 6),
            (5, 2),
            (5, 6),
            (6, 2),
            (6, 6),
            (7, 2),
            (7, 6),
        }:
            return 6
        return 3

    mlp_1_3_outputs = [
        mlp_1_3(k0, k1) for k0, k1 in zip(positions, num_mlp_0_1_outputs)
    ]
    mlp_1_3_output_scores = classifier_weights.loc[
        [("mlp_1_3_outputs", str(v)) for v in mlp_1_3_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_3_output):
        key = num_attn_1_3_output
        return 0

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_1_3_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_1_output, num_attn_0_6_output):
        key = (num_attn_0_1_output, num_attn_0_6_output)
        return 6

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_6_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # num_mlp_1_2 #################################################
    def num_mlp_1_2(num_attn_0_6_output, num_attn_0_2_output):
        key = (num_attn_0_6_output, num_attn_0_2_output)
        if key in {
            (0, 0),
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (3, 2),
            (4, 0),
            (4, 1),
            (4, 2),
            (5, 0),
            (5, 1),
            (5, 2),
            (6, 0),
            (6, 1),
            (6, 2),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
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
        }:
            return 5
        elif key in {
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
            (1, 3),
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
            (3, 10),
            (3, 11),
            (3, 12),
            (3, 13),
            (3, 14),
            (3, 15),
            (4, 13),
            (4, 14),
            (4, 15),
        }:
            return 7
        return 6

    num_mlp_1_2_outputs = [
        num_mlp_1_2(k0, k1)
        for k0, k1 in zip(num_attn_0_6_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_1_2_output_scores = classifier_weights.loc[
        [("num_mlp_1_2_outputs", str(v)) for v in num_mlp_1_2_outputs]
    ]

    # num_mlp_1_3 #################################################
    def num_mlp_1_3(num_attn_0_7_output):
        key = num_attn_0_7_output
        if key in {0}:
            return 7
        elif key in {1}:
            return 6
        return 2

    num_mlp_1_3_outputs = [num_mlp_1_3(k0) for k0 in num_attn_0_7_outputs]
    num_mlp_1_3_output_scores = classifier_weights.loc[
        [("num_mlp_1_3_outputs", str(v)) for v in num_mlp_1_3_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(position, token):
        if position in {0, 2, 4, 5, 6, 7}:
            return token == "<s>"
        elif position in {1}:
            return token == "4"
        elif position in {3}:
            return token == "<pad>"

    attn_2_0_pattern = select_closest(tokens, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, num_mlp_0_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(mlp_0_3_output, num_mlp_1_3_output):
        if mlp_0_3_output in {0, 1, 2, 6}:
            return num_mlp_1_3_output == 5
        elif mlp_0_3_output in {3}:
            return num_mlp_1_3_output == 0
        elif mlp_0_3_output in {4}:
            return num_mlp_1_3_output == 1
        elif mlp_0_3_output in {5, 7}:
            return num_mlp_1_3_output == 7

    attn_2_1_pattern = select_closest(
        num_mlp_1_3_outputs, mlp_0_3_outputs, predicate_2_1
    )
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_6_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_position, k_position):
        if q_position in {0, 2, 3, 4}:
            return k_position == 0
        elif q_position in {1, 7}:
            return k_position == 5
        elif q_position in {5, 6}:
            return k_position == 7

    attn_2_2_pattern = select_closest(positions, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, mlp_0_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_token, k_token):
        if q_token in {"0", "<s>"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"

    attn_2_3_pattern = select_closest(tokens, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_0_4_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(num_mlp_1_3_output, mlp_0_1_output):
        if num_mlp_1_3_output in {0}:
            return mlp_0_1_output == 4
        elif num_mlp_1_3_output in {1, 3, 5, 7}:
            return mlp_0_1_output == 7
        elif num_mlp_1_3_output in {2, 4}:
            return mlp_0_1_output == 2
        elif num_mlp_1_3_output in {6}:
            return mlp_0_1_output == 5

    attn_2_4_pattern = select_closest(
        mlp_0_1_outputs, num_mlp_1_3_outputs, predicate_2_4
    )
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_3_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_1_output, position):
        if attn_0_1_output in {"0", "1", "5", "<s>"}:
            return position == 3
        elif attn_0_1_output in {"4", "2"}:
            return position == 6
        elif attn_0_1_output in {"3"}:
            return position == 7

    attn_2_5_pattern = select_closest(positions, attn_0_1_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, num_mlp_1_3_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(q_attn_0_1_output, k_attn_0_1_output):
        if q_attn_0_1_output in {"0"}:
            return k_attn_0_1_output == "5"
        elif q_attn_0_1_output in {"4", "1", "5", "2"}:
            return k_attn_0_1_output == "3"
        elif q_attn_0_1_output in {"3"}:
            return k_attn_0_1_output == "<s>"
        elif q_attn_0_1_output in {"<s>"}:
            return k_attn_0_1_output == "2"

    attn_2_6_pattern = select_closest(attn_0_1_outputs, attn_0_1_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, num_mlp_1_3_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_0_1_output, token):
        if attn_0_1_output in {"0"}:
            return token == "5"
        elif attn_0_1_output in {"1", "5"}:
            return token == "4"
        elif attn_0_1_output in {"4", "2"}:
            return token == "1"
        elif attn_0_1_output in {"3"}:
            return token == "<s>"
        elif attn_0_1_output in {"<s>"}:
            return token == "2"

    attn_2_7_pattern = select_closest(tokens, attn_0_1_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, tokens)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_1_output, token):
        if attn_0_1_output in {"4", "1", "2", "0", "3"}:
            return token == "<s>"
        elif attn_0_1_output in {"5"}:
            return token == "5"
        elif attn_0_1_output in {"<s>"}:
            return token == "2"

    num_attn_2_0_pattern = select(tokens, attn_0_1_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_4_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_1_output, token):
        if attn_0_1_output in {"0"}:
            return token == "0"
        elif attn_0_1_output in {"<s>", "5", "4", "1", "3"}:
            return token == "2"
        elif attn_0_1_output in {"2"}:
            return token == "4"

    num_attn_2_1_pattern = select(tokens, attn_0_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, ones)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_1_output, token):
        if attn_0_1_output in {"4", "0", "<s>", "3"}:
            return token == "5"
        elif attn_0_1_output in {"1"}:
            return token == "1"
        elif attn_0_1_output in {"2"}:
            return token == "<s>"
        elif attn_0_1_output in {"5"}:
            return token == "0"

    num_attn_2_2_pattern = select(tokens, attn_0_1_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, ones)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_6_output, token):
        if attn_0_6_output in {"<s>", "5", "1", "2", "0", "3"}:
            return token == "4"
        elif attn_0_6_output in {"4"}:
            return token == "<s>"

    num_attn_2_3_pattern = select(tokens, attn_0_6_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, ones)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_0_1_output, token):
        if attn_0_1_output in {"5", "4", "2", "0", "3"}:
            return token == "<s>"
        elif attn_0_1_output in {"1"}:
            return token == "1"
        elif attn_0_1_output in {"<s>"}:
            return token == "2"

    num_attn_2_4_pattern = select(tokens, attn_0_1_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_2_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_0_1_output, token):
        if attn_0_1_output in {"0"}:
            return token == "0"
        elif attn_0_1_output in {"4", "1", "5", "2"}:
            return token == "3"
        elif attn_0_1_output in {"3"}:
            return token == "<s>"
        elif attn_0_1_output in {"<s>"}:
            return token == "4"

    num_attn_2_5_pattern = select(tokens, attn_0_1_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, ones)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_0_6_output, token):
        if attn_0_6_output in {"5", "4", "2", "0", "3"}:
            return token == "<s>"
        elif attn_0_6_output in {"1"}:
            return token == "1"
        elif attn_0_6_output in {"<s>"}:
            return token == "4"

    num_attn_2_6_pattern = select(tokens, attn_0_6_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_1_1_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_0_1_output, token):
        if attn_0_1_output in {"0"}:
            return token == "0"
        elif attn_0_1_output in {"4", "1", "5", "3"}:
            return token == "<s>"
        elif attn_0_1_output in {"2"}:
            return token == "2"
        elif attn_0_1_output in {"<s>"}:
            return token == "5"

    num_attn_2_7_pattern = select(tokens, attn_0_1_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, ones)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_0_5_output, position):
        key = (attn_0_5_output, position)
        return 0

    mlp_2_0_outputs = [mlp_2_0(k0, k1) for k0, k1 in zip(attn_0_5_outputs, positions)]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(position, num_mlp_1_3_output):
        key = (position, num_mlp_1_3_output)
        if key in {
            (0, 7),
            (2, 7),
            (3, 7),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 7),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 7),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 7),
            (7, 2),
            (7, 7),
        }:
            return 0
        return 3

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(positions, num_mlp_1_3_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # mlp_2_2 #####################################################
    def mlp_2_2(mlp_0_2_output, attn_0_1_output):
        key = (mlp_0_2_output, attn_0_1_output)
        return 1

    mlp_2_2_outputs = [
        mlp_2_2(k0, k1) for k0, k1 in zip(mlp_0_2_outputs, attn_0_1_outputs)
    ]
    mlp_2_2_output_scores = classifier_weights.loc[
        [("mlp_2_2_outputs", str(v)) for v in mlp_2_2_outputs]
    ]

    # mlp_2_3 #####################################################
    def mlp_2_3(token, attn_0_4_output):
        key = (token, attn_0_4_output)
        if key in {
            ("3", "1"),
            ("3", "3"),
            ("3", "4"),
            ("3", "5"),
            ("4", "1"),
            ("4", "3"),
            ("4", "4"),
            ("4", "5"),
            ("5", "1"),
            ("5", "3"),
            ("5", "4"),
            ("5", "5"),
            ("<s>", "1"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "5"),
        }:
            return 7
        return 2

    mlp_2_3_outputs = [mlp_2_3(k0, k1) for k0, k1 in zip(tokens, attn_0_4_outputs)]
    mlp_2_3_output_scores = classifier_weights.loc[
        [("mlp_2_3_outputs", str(v)) for v in mlp_2_3_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_7_output, num_attn_2_3_output):
        key = (num_attn_1_7_output, num_attn_2_3_output)
        return 1

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_2_3_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_5_output):
        key = num_attn_1_5_output
        if key in {0}:
            return 0
        elif key in {1}:
            return 7
        return 5

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_1_5_outputs]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    # num_mlp_2_2 #################################################
    def num_mlp_2_2(num_attn_0_3_output, num_attn_1_2_output):
        key = (num_attn_0_3_output, num_attn_1_2_output)
        return 3

    num_mlp_2_2_outputs = [
        num_mlp_2_2(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_2_2_output_scores = classifier_weights.loc[
        [("num_mlp_2_2_outputs", str(v)) for v in num_mlp_2_2_outputs]
    ]

    # num_mlp_2_3 #################################################
    def num_mlp_2_3(num_attn_1_0_output, num_attn_1_5_output):
        key = (num_attn_1_0_output, num_attn_1_5_output)
        return 1

    num_mlp_2_3_outputs = [
        num_mlp_2_3(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_5_outputs)
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


print(run(["<s>", "1", "5", "1", "2", "0", "3"]))
