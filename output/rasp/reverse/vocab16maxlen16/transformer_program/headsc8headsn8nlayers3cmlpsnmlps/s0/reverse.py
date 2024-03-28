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
        "output/rasp/reverse/vocab16maxlen16/transformer_program/headsc8headsn8nlayers3cmlpsnmlps/s0/reverse_weights.csv",
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
        if q_position in {0}:
            return k_position == 8
        elif q_position in {1}:
            return k_position == 14
        elif q_position in {2}:
            return k_position == 13
        elif q_position in {3}:
            return k_position == 12
        elif q_position in {4}:
            return k_position == 11
        elif q_position in {5}:
            return k_position == 10
        elif q_position in {13, 6}:
            return k_position == 2
        elif q_position in {14, 7}:
            return k_position == 1
        elif q_position in {8}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 15
        elif q_position in {10}:
            return k_position == 5
        elif q_position in {11}:
            return k_position == 4
        elif q_position in {12}:
            return k_position == 3
        elif q_position in {15}:
            return k_position == 0

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 2}:
            return k_position == 10
        elif q_position in {1}:
            return k_position == 13
        elif q_position in {10, 3}:
            return k_position == 5
        elif q_position in {4, 5}:
            return k_position == 9
        elif q_position in {8, 11, 6}:
            return k_position == 3
        elif q_position in {7}:
            return k_position == 4
        elif q_position in {9}:
            return k_position == 6
        elif q_position in {12}:
            return k_position == 15
        elif q_position in {13, 14}:
            return k_position == 1
        elif q_position in {15}:
            return k_position == 0

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 10
        elif q_position in {1}:
            return k_position == 12
        elif q_position in {2, 5}:
            return k_position == 4
        elif q_position in {3, 13}:
            return k_position == 2
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {8, 6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 6
        elif q_position in {9, 10}:
            return k_position == 15
        elif q_position in {11, 14}:
            return k_position == 1
        elif q_position in {12}:
            return k_position == 3
        elif q_position in {15}:
            return k_position == 0

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 10
        elif q_position in {1}:
            return k_position == 9
        elif q_position in {2}:
            return k_position == 5
        elif q_position in {3}:
            return k_position == 7
        elif q_position in {4, 5, 6, 9, 10, 12, 14}:
            return k_position == 1
        elif q_position in {7}:
            return k_position == 3
        elif q_position in {8, 13}:
            return k_position == 2
        elif q_position in {11}:
            return k_position == 4
        elif q_position in {15}:
            return k_position == 0

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_token, k_token):
        if q_token in {
            "8",
            "10",
            "0",
            "2",
            "3",
            "4",
            "5",
            "11",
            "9",
            "6",
            "1",
            "7",
            "12",
        }:
            return k_token == "4"
        elif q_token in {"<s>", "</s>"}:
            return k_token == "0"

    attn_0_4_pattern = select_closest(tokens, tokens, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0, 3, 13, 15}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 14
        elif q_position in {2, 4}:
            return k_position == 11
        elif q_position in {10, 12, 5}:
            return k_position == 2
        elif q_position in {6}:
            return k_position == 6
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8, 11, 14}:
            return k_position == 1
        elif q_position in {9}:
            return k_position == 5

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0, 2, 3}:
            return k_position == 8
        elif q_position in {1}:
            return k_position == 10
        elif q_position in {4, 5, 7, 11, 12}:
            return k_position == 15
        elif q_position in {6, 15}:
            return k_position == 0
        elif q_position in {8}:
            return k_position == 6
        elif q_position in {9}:
            return k_position == 2
        elif q_position in {10}:
            return k_position == 4
        elif q_position in {13, 14}:
            return k_position == 1

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(q_token, k_token):
        if q_token in {"10", "0", "3", "11", "1", "</s>"}:
            return k_token == "6"
        elif q_token in {"<s>", "12"}:
            return k_token == "0"
        elif q_token in {"2"}:
            return k_token == "11"
        elif q_token in {"4", "5", "6", "8", "7", "9"}:
            return k_token == "1"

    attn_0_7_pattern = select_closest(tokens, tokens, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(token, position):
        if token in {"10", "2", "0", "5", "1", "</s>"}:
            return position == 15
        elif token in {"11"}:
            return position == 14
        elif token in {"6", "7", "12", "3"}:
            return position == 8
        elif token in {"4"}:
            return position == 5
        elif token in {"8"}:
            return position == 7
        elif token in {"9"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 1

    num_attn_0_0_pattern = select(positions, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"4", "7", "0", "</s>"}:
            return position == 5
        elif token in {"10", "2", "3", "6", "1", "12"}:
            return position == 6
        elif token in {"8", "11"}:
            return position == 3
        elif token in {"5", "9"}:
            return position == 0
        elif token in {"<s>"}:
            return position == 1

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"4", "0", "12", "5"}:
            return position == 5
        elif token in {"1", "11"}:
            return position == 0
        elif token in {"7", "10", "</s>", "3"}:
            return position == 6
        elif token in {"2", "6"}:
            return position == 3
        elif token in {"8"}:
            return position == 1
        elif token in {"<s>", "9"}:
            return position == 4

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"0", "3"}:
            return position == 6
        elif token in {"1", "4", "5"}:
            return position == 0
        elif token in {"10", "2", "12"}:
            return position == 3
        elif token in {"7", "11", "</s>"}:
            return position == 5
        elif token in {"8", "6"}:
            return position == 8
        elif token in {"<s>", "9"}:
            return position == 1

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(q_token, k_token):
        if q_token in {"<s>", "11", "0"}:
            return k_token == "<s>"
        elif q_token in {"1", "10", "9", "6"}:
            return k_token == "5"
        elif q_token in {"12"}:
            return k_token == "8"
        elif q_token in {"4", "2", "5"}:
            return k_token == "12"
        elif q_token in {"3"}:
            return k_token == "6"
        elif q_token in {"7"}:
            return k_token == "<pad>"
        elif q_token in {"8", "</s>"}:
            return k_token == "</s>"

    num_attn_0_4_pattern = select(tokens, tokens, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(token, position):
        if token in {"</s>", "0", "12", "6"}:
            return position == 6
        elif token in {"1", "2", "3"}:
            return position == 5
        elif token in {"10", "9"}:
            return position == 8
        elif token in {"4", "11", "5"}:
            return position == 15
        elif token in {"7"}:
            return position == 10
        elif token in {"8"}:
            return position == 7
        elif token in {"<s>"}:
            return position == 3

    num_attn_0_5_pattern = select(positions, tokens, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(token, position):
        if token in {"</s>", "0", "12"}:
            return position == 3
        elif token in {"1", "2", "6"}:
            return position == 8
        elif token in {"10"}:
            return position == 2
        elif token in {"11"}:
            return position == 7
        elif token in {"3"}:
            return position == 0
        elif token in {"4", "8"}:
            return position == 5
        elif token in {"5", "7", "9"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 1

    num_attn_0_6_pattern = select(positions, tokens, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(token, position):
        if token in {"8", "0", "6"}:
            return position == 3
        elif token in {"1", "12", "3"}:
            return position == 4
        elif token in {"4", "7", "10", "2"}:
            return position == 0
        elif token in {"5", "11"}:
            return position == 6
        elif token in {"</s>", "9"}:
            return position == 5
        elif token in {"<s>"}:
            return position == 1

    num_attn_0_7_pattern = select(positions, tokens, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, attn_0_4_output):
        key = (attn_0_0_output, attn_0_4_output)
        if key in {
            ("0", "1"),
            ("0", "11"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "6"),
            ("0", "7"),
            ("0", "8"),
            ("0", "9"),
            ("0", "</s>"),
            ("1", "0"),
            ("1", "10"),
            ("1", "12"),
            ("1", "4"),
            ("1", "7"),
            ("1", "8"),
            ("1", "</s>"),
            ("10", "4"),
            ("11", "4"),
            ("12", "1"),
            ("2", "4"),
            ("3", "4"),
            ("4", "0"),
            ("4", "1"),
            ("4", "10"),
            ("4", "11"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "5"),
            ("4", "6"),
            ("4", "7"),
            ("4", "8"),
            ("4", "9"),
            ("4", "</s>"),
            ("5", "1"),
            ("5", "2"),
            ("5", "4"),
            ("6", "1"),
            ("6", "2"),
            ("6", "4"),
            ("6", "6"),
            ("6", "</s>"),
            ("7", "4"),
            ("8", "1"),
            ("8", "2"),
            ("8", "4"),
            ("8", "6"),
            ("8", "</s>"),
            ("9", "1"),
            ("9", "2"),
            ("9", "4"),
            ("9", "6"),
            ("</s>", "4"),
            ("<s>", "4"),
            ("<s>", "</s>"),
        }:
            return 10
        elif key in {
            ("10", "0"),
            ("10", "10"),
            ("10", "12"),
            ("10", "<s>"),
            ("11", "10"),
            ("11", "11"),
            ("11", "3"),
            ("11", "5"),
            ("11", "7"),
            ("11", "8"),
            ("11", "9"),
            ("11", "</s>"),
            ("11", "<s>"),
            ("2", "10"),
            ("2", "11"),
            ("2", "3"),
            ("2", "5"),
            ("2", "6"),
            ("2", "8"),
            ("2", "9"),
            ("2", "</s>"),
            ("2", "<s>"),
            ("3", "0"),
            ("3", "12"),
            ("3", "<s>"),
            ("7", "0"),
            ("7", "10"),
            ("7", "5"),
            ("7", "7"),
            ("7", "9"),
            ("7", "<s>"),
            ("</s>", "0"),
            ("</s>", "10"),
            ("</s>", "11"),
            ("</s>", "12"),
            ("</s>", "3"),
            ("</s>", "5"),
            ("</s>", "7"),
            ("</s>", "8"),
            ("</s>", "9"),
            ("</s>", "<s>"),
            ("<s>", "11"),
            ("<s>", "3"),
            ("<s>", "5"),
            ("<s>", "8"),
            ("<s>", "9"),
            ("<s>", "<s>"),
        }:
            return 6
        elif key in {
            ("1", "1"),
            ("1", "11"),
            ("1", "2"),
            ("1", "3"),
            ("1", "5"),
            ("1", "6"),
            ("1", "9"),
            ("1", "<s>"),
            ("10", "1"),
            ("10", "11"),
            ("10", "2"),
            ("10", "3"),
            ("10", "5"),
            ("10", "6"),
            ("10", "7"),
            ("10", "8"),
            ("10", "9"),
            ("10", "</s>"),
            ("11", "1"),
            ("11", "2"),
            ("11", "6"),
            ("2", "1"),
            ("2", "2"),
            ("3", "1"),
            ("3", "10"),
            ("3", "11"),
            ("3", "2"),
            ("3", "3"),
            ("3", "5"),
            ("3", "6"),
            ("3", "7"),
            ("3", "8"),
            ("3", "9"),
            ("3", "</s>"),
            ("7", "1"),
            ("7", "11"),
            ("7", "2"),
            ("7", "3"),
            ("7", "6"),
            ("7", "8"),
            ("7", "</s>"),
            ("</s>", "1"),
            ("</s>", "2"),
            ("</s>", "6"),
            ("</s>", "</s>"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "6"),
        }:
            return 9
        elif key in {("2", "7")}:
            return 7
        return 11

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_4_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_2_output, attn_0_1_output):
        key = (attn_0_2_output, attn_0_1_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "10"),
            ("0", "11"),
            ("0", "12"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "5"),
            ("0", "6"),
            ("0", "7"),
            ("0", "9"),
            ("0", "<s>"),
            ("2", "0"),
            ("2", "1"),
            ("2", "10"),
            ("2", "11"),
            ("2", "12"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("2", "5"),
            ("2", "6"),
            ("2", "7"),
            ("2", "9"),
            ("2", "<s>"),
            ("4", "0"),
            ("4", "1"),
            ("4", "10"),
            ("4", "11"),
            ("4", "12"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "5"),
            ("4", "6"),
            ("4", "7"),
            ("4", "9"),
            ("4", "<s>"),
            ("</s>", "7"),
        }:
            return 14
        elif key in {
            ("0", "</s>"),
            ("1", "</s>"),
            ("10", "</s>"),
            ("11", "</s>"),
            ("12", "</s>"),
            ("2", "</s>"),
            ("4", "</s>"),
            ("5", "</s>"),
            ("9", "</s>"),
            ("<s>", "</s>"),
        }:
            return 3
        elif key in {
            ("</s>", "1"),
            ("</s>", "3"),
            ("</s>", "6"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
        }:
            return 11
        elif key in {
            ("</s>", "0"),
            ("</s>", "10"),
            ("</s>", "4"),
            ("</s>", "5"),
            ("</s>", "9"),
        }:
            return 0
        elif key in {("</s>", "11"), ("</s>", "12"), ("</s>", "2")}:
            return 9
        elif key in {("3", "</s>"), ("6", "</s>")}:
            return 10
        elif key in {("8", "</s>")}:
            return 4
        return 12

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_1_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_5_output, num_attn_0_0_output):
        key = (num_attn_0_5_output, num_attn_0_0_output)
        return 6

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_5_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output, num_attn_0_5_output):
        key = (num_attn_0_0_output, num_attn_0_5_output)
        return 1

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_5_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0, 8, 9, 7}:
            return k_position == 15
        elif q_position in {1, 3}:
            return k_position == 11
        elif q_position in {2}:
            return k_position == 12
        elif q_position in {4}:
            return k_position == 10
        elif q_position in {5}:
            return k_position == 7
        elif q_position in {6}:
            return k_position == 8
        elif q_position in {10}:
            return k_position == 3
        elif q_position in {11}:
            return k_position == 14
        elif q_position in {12, 14, 15}:
            return k_position == 1
        elif q_position in {13}:
            return k_position == 2

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_1_output, token):
        if attn_0_1_output in {"0"}:
            return token == "5"
        elif attn_0_1_output in {
            "8",
            "</s>",
            "5",
            "11",
            "9",
            "6",
            "1",
            "7",
            "<s>",
            "12",
        }:
            return token == "10"
        elif attn_0_1_output in {"4", "10"}:
            return token == "3"
        elif attn_0_1_output in {"2"}:
            return token == "</s>"
        elif attn_0_1_output in {"3"}:
            return token == "9"

    attn_1_1_pattern = select_closest(tokens, attn_0_1_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_0_output, token):
        if attn_0_0_output in {"8", "0", "6"}:
            return token == "11"
        elif attn_0_0_output in {"1"}:
            return token == "2"
        elif attn_0_0_output in {"10", "12"}:
            return token == "3"
        elif attn_0_0_output in {"4", "<s>", "11", "2"}:
            return token == "5"
        elif attn_0_0_output in {"3"}:
            return token == "</s>"
        elif attn_0_0_output in {"5"}:
            return token == "<s>"
        elif attn_0_0_output in {"7", "</s>"}:
            return token == "10"
        elif attn_0_0_output in {"9"}:
            return token == "0"

    attn_1_2_pattern = select_closest(tokens, attn_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_7_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, token):
        if position in {0, 5, 6, 9, 12, 13, 14, 15}:
            return token == "6"
        elif position in {1, 2}:
            return token == "7"
        elif position in {3}:
            return token == "9"
        elif position in {10, 4}:
            return token == "11"
        elif position in {8, 7}:
            return token == "0"
        elif position in {11}:
            return token == "2"

    attn_1_3_pattern = select_closest(tokens, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_2_output, token):
        if attn_0_2_output in {"8", "2", "0", "1", "12"}:
            return token == "11"
        elif attn_0_2_output in {"10", "<s>"}:
            return token == "8"
        elif attn_0_2_output in {"11"}:
            return token == "9"
        elif attn_0_2_output in {"4", "7", "3"}:
            return token == "5"
        elif attn_0_2_output in {"5"}:
            return token == "12"
        elif attn_0_2_output in {"6"}:
            return token == "3"
        elif attn_0_2_output in {"9"}:
            return token == "</s>"
        elif attn_0_2_output in {"</s>"}:
            return token == "10"

    attn_1_4_pattern = select_closest(tokens, attn_0_2_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_3_output, token):
        if attn_0_3_output in {"0"}:
            return token == "11"
        elif attn_0_3_output in {
            "8",
            "10",
            "2",
            "</s>",
            "3",
            "4",
            "5",
            "11",
            "6",
            "1",
            "7",
            "<s>",
            "9",
        }:
            return token == "12"
        elif attn_0_3_output in {"12"}:
            return token == "</s>"

    attn_1_5_pattern = select_closest(tokens, attn_0_3_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(attn_0_0_output, token):
        if attn_0_0_output in {
            "8",
            "10",
            "0",
            "2",
            "4",
            "5",
            "11",
            "9",
            "6",
            "1",
            "7",
            "<s>",
            "12",
        }:
            return token == "3"
        elif attn_0_0_output in {"3"}:
            return token == "</s>"
        elif attn_0_0_output in {"</s>"}:
            return token == "<pad>"

    attn_1_6_pattern = select_closest(tokens, attn_0_0_outputs, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, tokens)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_2_output, token):
        if attn_0_2_output in {
            "8",
            "10",
            "0",
            "</s>",
            "3",
            "5",
            "11",
            "9",
            "6",
            "1",
            "7",
            "<s>",
            "12",
        }:
            return token == "2"
        elif attn_0_2_output in {"2"}:
            return token == "</s>"
        elif attn_0_2_output in {"4"}:
            return token == "1"

    attn_1_7_pattern = select_closest(tokens, attn_0_2_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, tokens)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_3_output, attn_0_2_output):
        if attn_0_3_output in {"8", "10", "0", "5", "11", "1", "7"}:
            return attn_0_2_output == "</s>"
        elif attn_0_3_output in {"<s>", "12"}:
            return attn_0_2_output == "12"
        elif attn_0_3_output in {"2", "</s>"}:
            return attn_0_2_output == "2"
        elif attn_0_3_output in {"3"}:
            return attn_0_2_output == "3"
        elif attn_0_3_output in {"4"}:
            return attn_0_2_output == "<pad>"
        elif attn_0_3_output in {"6"}:
            return attn_0_2_output == "6"
        elif attn_0_3_output in {"9"}:
            return attn_0_2_output == "9"

    num_attn_1_0_pattern = select(attn_0_2_outputs, attn_0_3_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_0_output, attn_0_6_output):
        if attn_0_0_output in {"4", "<s>", "0", "6"}:
            return attn_0_6_output == "<pad>"
        elif attn_0_0_output in {"3", "5", "11", "1", "7", "9"}:
            return attn_0_6_output == "</s>"
        elif attn_0_0_output in {"10"}:
            return attn_0_6_output == "10"
        elif attn_0_0_output in {"8", "12"}:
            return attn_0_6_output == "12"
        elif attn_0_0_output in {"2", "</s>"}:
            return attn_0_6_output == "2"

    num_attn_1_1_pattern = select(attn_0_6_outputs, attn_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_0_output, attn_0_5_output):
        if attn_0_0_output in {"8", "10", "2", "0", "3", "4", "11", "1"}:
            return attn_0_5_output == "</s>"
        elif attn_0_0_output in {"12"}:
            return attn_0_5_output == "12"
        elif attn_0_0_output in {"5"}:
            return attn_0_5_output == "5"
        elif attn_0_0_output in {"<s>", "9", "6"}:
            return attn_0_5_output == "6"
        elif attn_0_0_output in {"7", "</s>"}:
            return attn_0_5_output == "7"

    num_attn_1_2_pattern = select(attn_0_5_outputs, attn_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_0_output, attn_0_6_output):
        if attn_0_0_output in {"0", "4", "5", "9", "8", "12"}:
            return attn_0_6_output == "</s>"
        elif attn_0_0_output in {"1", "10"}:
            return attn_0_6_output == "<pad>"
        elif attn_0_0_output in {"11"}:
            return attn_0_6_output == "11"
        elif attn_0_0_output in {"2", "</s>", "6"}:
            return attn_0_6_output == "2"
        elif attn_0_0_output in {"3"}:
            return attn_0_6_output == "3"
        elif attn_0_0_output in {"7"}:
            return attn_0_6_output == "10"
        elif attn_0_0_output in {"<s>"}:
            return attn_0_6_output == "6"

    num_attn_1_3_pattern = select(attn_0_6_outputs, attn_0_0_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(attn_0_0_output, attn_0_6_output):
        if attn_0_0_output in {"0"}:
            return attn_0_6_output == "0"
        elif attn_0_0_output in {"1"}:
            return attn_0_6_output == "1"
        elif attn_0_0_output in {"10", "3", "4", "5", "11", "6", "8", "7", "12"}:
            return attn_0_6_output == "</s>"
        elif attn_0_0_output in {"2", "</s>"}:
            return attn_0_6_output == "2"
        elif attn_0_0_output in {"9"}:
            return attn_0_6_output == "9"
        elif attn_0_0_output in {"<s>"}:
            return attn_0_6_output == "<pad>"

    num_attn_1_4_pattern = select(attn_0_6_outputs, attn_0_0_outputs, num_predicate_1_4)
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, ones)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_3_output, attn_0_2_output):
        if attn_0_3_output in {"2", "0", "3", "4", "5", "7", "9"}:
            return attn_0_2_output == "</s>"
        elif attn_0_3_output in {"1"}:
            return attn_0_2_output == "1"
        elif attn_0_3_output in {"10", "</s>"}:
            return attn_0_2_output == "10"
        elif attn_0_3_output in {"11"}:
            return attn_0_2_output == "11"
        elif attn_0_3_output in {"12"}:
            return attn_0_2_output == "12"
        elif attn_0_3_output in {"6"}:
            return attn_0_2_output == "6"
        elif attn_0_3_output in {"8"}:
            return attn_0_2_output == "8"
        elif attn_0_3_output in {"<s>"}:
            return attn_0_2_output == "<pad>"

    num_attn_1_5_pattern = select(attn_0_2_outputs, attn_0_3_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, ones)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(attn_0_3_output, attn_0_2_output):
        if attn_0_3_output in {"0"}:
            return attn_0_2_output == "0"
        elif attn_0_3_output in {"8", "10", "3", "5", "1", "9"}:
            return attn_0_2_output == "</s>"
        elif attn_0_3_output in {"<s>", "11", "12"}:
            return attn_0_2_output == "12"
        elif attn_0_3_output in {"2", "</s>"}:
            return attn_0_2_output == "2"
        elif attn_0_3_output in {"4"}:
            return attn_0_2_output == "4"
        elif attn_0_3_output in {"6"}:
            return attn_0_2_output == "<s>"
        elif attn_0_3_output in {"7"}:
            return attn_0_2_output == "7"

    num_attn_1_6_pattern = select(attn_0_2_outputs, attn_0_3_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, ones)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(attn_0_0_output, attn_0_6_output):
        if attn_0_0_output in {"8", "2", "0", "5", "11", "6", "1", "12"}:
            return attn_0_6_output == "</s>"
        elif attn_0_0_output in {"10"}:
            return attn_0_6_output == "10"
        elif attn_0_0_output in {"9", "3"}:
            return attn_0_6_output == "3"
        elif attn_0_0_output in {"4", "</s>"}:
            return attn_0_6_output == "4"
        elif attn_0_0_output in {"7"}:
            return attn_0_6_output == "7"
        elif attn_0_0_output in {"<s>"}:
            return attn_0_6_output == "6"

    num_attn_1_7_pattern = select(attn_0_6_outputs, attn_0_0_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, ones)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_7_output, attn_1_3_output):
        key = (attn_0_7_output, attn_1_3_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "10"),
            ("0", "11"),
            ("0", "12"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "6"),
            ("0", "8"),
            ("0", "9"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("1", "0"),
            ("1", "1"),
            ("1", "10"),
            ("1", "12"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "6"),
            ("1", "8"),
            ("1", "9"),
            ("1", "</s>"),
            ("10", "0"),
            ("10", "1"),
            ("10", "10"),
            ("10", "12"),
            ("10", "2"),
            ("10", "3"),
            ("10", "4"),
            ("10", "6"),
            ("10", "8"),
            ("10", "9"),
            ("10", "</s>"),
            ("11", "0"),
            ("11", "6"),
            ("11", "9"),
            ("12", "0"),
            ("12", "6"),
            ("12", "9"),
            ("2", "0"),
            ("2", "1"),
            ("2", "12"),
            ("2", "2"),
            ("2", "4"),
            ("2", "6"),
            ("2", "9"),
            ("3", "0"),
            ("3", "6"),
            ("3", "9"),
            ("4", "0"),
            ("4", "6"),
            ("4", "9"),
            ("6", "0"),
            ("6", "1"),
            ("6", "10"),
            ("6", "12"),
            ("6", "2"),
            ("6", "4"),
            ("6", "6"),
            ("6", "7"),
            ("6", "9"),
            ("6", "</s>"),
            ("7", "0"),
            ("7", "6"),
            ("7", "9"),
            ("8", "0"),
            ("8", "1"),
            ("8", "2"),
            ("8", "6"),
            ("8", "9"),
            ("9", "0"),
            ("9", "6"),
            ("9", "9"),
            ("</s>", "0"),
            ("</s>", "6"),
            ("</s>", "9"),
            ("<s>", "0"),
        }:
            return 4
        elif key in {
            ("0", "5"),
            ("1", "5"),
            ("10", "5"),
            ("2", "5"),
            ("3", "5"),
            ("5", "0"),
            ("5", "1"),
            ("5", "10"),
            ("5", "12"),
            ("5", "2"),
            ("5", "3"),
            ("5", "4"),
            ("5", "5"),
            ("5", "6"),
            ("5", "8"),
            ("5", "9"),
            ("5", "</s>"),
            ("8", "5"),
            ("9", "5"),
        }:
            return 0
        elif key in {
            ("0", "7"),
            ("1", "7"),
            ("10", "7"),
            ("11", "7"),
            ("12", "7"),
            ("2", "7"),
            ("3", "7"),
            ("4", "7"),
            ("5", "7"),
            ("7", "7"),
            ("8", "7"),
            ("9", "7"),
            ("</s>", "7"),
            ("<s>", "7"),
        }:
            return 12
        elif key in {("11", "5"), ("12", "5"), ("4", "5"), ("7", "5"), ("</s>", "5")}:
            return 8
        return 11

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_7_outputs, attn_1_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(num_mlp_0_0_output, attn_1_1_output):
        key = (num_mlp_0_0_output, attn_1_1_output)
        return 12

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(num_mlp_0_0_outputs, attn_1_1_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_4_output, num_attn_1_2_output):
        key = (num_attn_1_4_output, num_attn_1_2_output)
        if key in {
            (0, 12),
            (0, 13),
            (0, 14),
            (0, 15),
            (0, 16),
            (0, 17),
            (0, 18),
            (0, 19),
            (0, 20),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (1, 21),
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (2, 22),
            (2, 23),
            (2, 24),
            (2, 25),
            (2, 26),
            (2, 27),
            (2, 28),
            (2, 29),
            (2, 30),
            (3, 27),
            (3, 28),
            (3, 29),
            (3, 30),
            (3, 31),
        }:
            return 7
        elif key in {
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
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
            (1, 30),
            (1, 31),
            (2, 31),
        }:
            return 12
        return 6

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_4_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_5_output, num_attn_1_7_output):
        key = (num_attn_1_5_output, num_attn_1_7_output)
        return 1

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_5_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_1_0_output, token):
        if attn_1_0_output in {"0"}:
            return token == "7"
        elif attn_1_0_output in {"10", "2", "5", "1", "</s>"}:
            return token == "3"
        elif attn_1_0_output in {"11"}:
            return token == "5"
        elif attn_1_0_output in {"12"}:
            return token == "2"
        elif attn_1_0_output in {"4", "9", "3"}:
            return token == "<s>"
        elif attn_1_0_output in {"6"}:
            return token == "8"
        elif attn_1_0_output in {"7"}:
            return token == "9"
        elif attn_1_0_output in {"8"}:
            return token == "12"
        elif attn_1_0_output in {"<s>"}:
            return token == "<pad>"

    attn_2_0_pattern = select_closest(tokens, attn_1_0_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_6_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_5_output, position):
        if attn_0_5_output in {"0", "12"}:
            return position == 12
        elif attn_0_5_output in {"1"}:
            return position == 13
        elif attn_0_5_output in {"7", "10", "3"}:
            return position == 2
        elif attn_0_5_output in {"4", "11"}:
            return position == 9
        elif attn_0_5_output in {"8", "2", "6"}:
            return position == 3
        elif attn_0_5_output in {"5", "<s>"}:
            return position == 6
        elif attn_0_5_output in {"9"}:
            return position == 10
        elif attn_0_5_output in {"</s>"}:
            return position == 0

    attn_2_1_pattern = select_closest(positions, attn_0_5_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_5_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_3_output, token):
        if attn_0_3_output in {"5", "0"}:
            return token == "8"
        elif attn_0_3_output in {"1", "8", "12"}:
            return token == "7"
        elif attn_0_3_output in {"10"}:
            return token == "3"
        elif attn_0_3_output in {"11"}:
            return token == "0"
        elif attn_0_3_output in {"7", "2", "3"}:
            return token == "<s>"
        elif attn_0_3_output in {"4"}:
            return token == "2"
        elif attn_0_3_output in {"6"}:
            return token == "12"
        elif attn_0_3_output in {"9"}:
            return token == "11"
        elif attn_0_3_output in {"</s>"}:
            return token == "</s>"
        elif attn_0_3_output in {"<s>"}:
            return token == "<pad>"

    attn_2_2_pattern = select_closest(tokens, attn_0_3_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_5_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0, 13}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 13
        elif q_position in {2}:
            return k_position == 7
        elif q_position in {3}:
            return k_position == 9
        elif q_position in {4, 12, 7}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 6
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {8, 10, 11, 14}:
            return k_position == 1
        elif q_position in {9, 15}:
            return k_position == 3

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, tokens)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(attn_0_3_output, token):
        if attn_0_3_output in {
            "8",
            "10",
            "0",
            "2",
            "3",
            "4",
            "5",
            "11",
            "9",
            "6",
            "1",
            "7",
            "12",
        }:
            return token == "<s>"
        elif attn_0_3_output in {"</s>"}:
            return token == "12"
        elif attn_0_3_output in {"<s>"}:
            return token == "3"

    attn_2_4_pattern = select_closest(tokens, attn_0_3_outputs, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, mlp_1_0_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 15
        elif q_position in {1}:
            return k_position == 12
        elif q_position in {2, 3}:
            return k_position == 10
        elif q_position in {4, 5}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 5
        elif q_position in {8, 9}:
            return k_position == 4
        elif q_position in {10, 11}:
            return k_position == 2
        elif q_position in {12, 13, 14}:
            return k_position == 1
        elif q_position in {15}:
            return k_position == 3

    attn_2_5_pattern = select_closest(positions, positions, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, tokens)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(attn_1_0_output, token):
        if attn_1_0_output in {"0", "3"}:
            return token == "12"
        elif attn_1_0_output in {"1"}:
            return token == "6"
        elif attn_1_0_output in {"10"}:
            return token == "4"
        elif attn_1_0_output in {"11"}:
            return token == "10"
        elif attn_1_0_output in {"5", "8", "12", "9"}:
            return token == "<s>"
        elif attn_1_0_output in {"2"}:
            return token == "11"
        elif attn_1_0_output in {"4", "<s>"}:
            return token == "<pad>"
        elif attn_1_0_output in {"7", "6"}:
            return token == "3"
        elif attn_1_0_output in {"</s>"}:
            return token == "9"

    attn_2_6_pattern = select_closest(tokens, attn_1_0_outputs, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_4_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(attn_1_0_output, attn_1_7_output):
        if attn_1_0_output in {"0"}:
            return attn_1_7_output == "12"
        elif attn_1_0_output in {"1"}:
            return attn_1_7_output == "<s>"
        elif attn_1_0_output in {"10", "</s>", "3", "11", "9", "6", "7", "<s>", "12"}:
            return attn_1_7_output == "2"
        elif attn_1_0_output in {"2"}:
            return attn_1_7_output == "0"
        elif attn_1_0_output in {"4"}:
            return attn_1_7_output == "<pad>"
        elif attn_1_0_output in {"5"}:
            return attn_1_7_output == "8"
        elif attn_1_0_output in {"8"}:
            return attn_1_7_output == "9"

    attn_2_7_pattern = select_closest(attn_1_7_outputs, attn_1_0_outputs, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_1_7_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_0_output, attn_0_6_output):
        if attn_0_0_output in {"<s>", "0"}:
            return attn_0_6_output == "0"
        elif attn_0_0_output in {"10", "2", "4", "9", "6", "1", "7", "12"}:
            return attn_0_6_output == "</s>"
        elif attn_0_0_output in {"11"}:
            return attn_0_6_output == "11"
        elif attn_0_0_output in {"3"}:
            return attn_0_6_output == "3"
        elif attn_0_0_output in {"5", "</s>"}:
            return attn_0_6_output == "5"
        elif attn_0_0_output in {"8"}:
            return attn_0_6_output == "8"

    num_attn_2_0_pattern = select(attn_0_6_outputs, attn_0_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_7_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_3_output, attn_0_2_output):
        if attn_0_3_output in {"8", "0", "3", "4", "6", "1", "7", "12"}:
            return attn_0_2_output == "</s>"
        elif attn_0_3_output in {"10"}:
            return attn_0_2_output == "10"
        elif attn_0_3_output in {"11"}:
            return attn_0_2_output == "11"
        elif attn_0_3_output in {"<s>", "2", "</s>"}:
            return attn_0_2_output == "2"
        elif attn_0_3_output in {"5"}:
            return attn_0_2_output == "5"
        elif attn_0_3_output in {"9"}:
            return attn_0_2_output == "9"

    num_attn_2_1_pattern = select(attn_0_2_outputs, attn_0_3_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_2_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_5_output, attn_1_0_output):
        if attn_1_5_output in {"5", "0"}:
            return attn_1_0_output == "5"
        elif attn_1_5_output in {"3", "12", "6", "1", "<s>", "9"}:
            return attn_1_0_output == "1"
        elif attn_1_5_output in {"8", "10", "</s>"}:
            return attn_1_0_output == "8"
        elif attn_1_5_output in {"4", "7", "11"}:
            return attn_1_0_output == "4"
        elif attn_1_5_output in {"2"}:
            return attn_1_0_output == "2"

    num_attn_2_2_pattern = select(attn_1_0_outputs, attn_1_5_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_0_output, attn_0_6_output):
        if attn_0_0_output in {"1", "11", "0", "6"}:
            return attn_0_6_output == "<pad>"
        elif attn_0_0_output in {"10", "<s>", "9"}:
            return attn_0_6_output == "9"
        elif attn_0_0_output in {"2", "3", "4", "5", "12"}:
            return attn_0_6_output == "</s>"
        elif attn_0_0_output in {"7", "</s>"}:
            return attn_0_6_output == "7"
        elif attn_0_0_output in {"8"}:
            return attn_0_6_output == "8"

    num_attn_2_3_pattern = select(attn_0_6_outputs, attn_0_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, ones)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(attn_0_3_output, attn_0_2_output):
        if attn_0_3_output in {"2", "0", "4", "11", "6", "8", "9"}:
            return attn_0_2_output == "</s>"
        elif attn_0_3_output in {"1"}:
            return attn_0_2_output == "1"
        elif attn_0_3_output in {"10", "</s>"}:
            return attn_0_2_output == "10"
        elif attn_0_3_output in {"12"}:
            return attn_0_2_output == "12"
        elif attn_0_3_output in {"<s>", "3"}:
            return attn_0_2_output == "3"
        elif attn_0_3_output in {"5"}:
            return attn_0_2_output == "5"
        elif attn_0_3_output in {"7"}:
            return attn_0_2_output == "<pad>"

    num_attn_2_4_pattern = select(attn_0_2_outputs, attn_0_3_outputs, num_predicate_2_4)
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_2_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_0_0_output, attn_0_6_output):
        if attn_0_0_output in {"7", "<s>", "0"}:
            return attn_0_6_output == "0"
        elif attn_0_0_output in {"1", "</s>"}:
            return attn_0_6_output == "<pad>"
        elif attn_0_0_output in {"10", "2", "3", "9", "8", "12"}:
            return attn_0_6_output == "</s>"
        elif attn_0_0_output in {"11"}:
            return attn_0_6_output == "11"
        elif attn_0_0_output in {"4"}:
            return attn_0_6_output == "1"
        elif attn_0_0_output in {"5"}:
            return attn_0_6_output == "5"
        elif attn_0_0_output in {"6"}:
            return attn_0_6_output == "6"

    num_attn_2_5_pattern = select(attn_0_6_outputs, attn_0_0_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, ones)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_0_3_output, attn_0_2_output):
        if attn_0_3_output in {"0"}:
            return attn_0_2_output == "0"
        elif attn_0_3_output in {"8", "2", "4", "11", "1", "12"}:
            return attn_0_2_output == "</s>"
        elif attn_0_3_output in {"10", "<s>", "</s>", "9"}:
            return attn_0_2_output == "10"
        elif attn_0_3_output in {"3"}:
            return attn_0_2_output == "6"
        elif attn_0_3_output in {"5"}:
            return attn_0_2_output == "<s>"
        elif attn_0_3_output in {"6"}:
            return attn_0_2_output == "<pad>"
        elif attn_0_3_output in {"7"}:
            return attn_0_2_output == "7"

    num_attn_2_6_pattern = select(attn_0_2_outputs, attn_0_3_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_2_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(attn_1_2_output, attn_1_0_output):
        if attn_1_2_output in {"6", "2", "0", "3"}:
            return attn_1_0_output == "<pad>"
        elif attn_1_2_output in {"10", "9", "1", "<s>", "</s>"}:
            return attn_1_0_output == "9"
        elif attn_1_2_output in {"5", "8", "11"}:
            return attn_1_0_output == "11"
        elif attn_1_2_output in {"7", "12"}:
            return attn_1_0_output == "12"
        elif attn_1_2_output in {"4"}:
            return attn_1_0_output == "4"

    num_attn_2_7_pattern = select(attn_1_0_outputs, attn_1_2_outputs, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_2_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_1_4_output, attn_1_7_output):
        key = (attn_1_4_output, attn_1_7_output)
        return 2

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_1_4_outputs, attn_1_7_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_0_1_output, attn_0_2_output):
        key = (attn_0_1_output, attn_0_2_output)
        if key in {
            ("0", "1"),
            ("0", "4"),
            ("1", "1"),
            ("1", "4"),
            ("10", "1"),
            ("10", "4"),
            ("11", "1"),
            ("11", "4"),
            ("12", "1"),
            ("12", "4"),
            ("2", "1"),
            ("2", "4"),
            ("3", "1"),
            ("3", "4"),
            ("4", "1"),
            ("4", "4"),
            ("5", "1"),
            ("5", "4"),
            ("6", "1"),
            ("6", "4"),
            ("7", "1"),
            ("7", "4"),
            ("8", "1"),
            ("8", "4"),
            ("9", "1"),
            ("9", "4"),
            ("<s>", "1"),
            ("<s>", "4"),
        }:
            return 6
        elif key in {
            ("0", "5"),
            ("0", "6"),
            ("1", "5"),
            ("1", "6"),
            ("10", "5"),
            ("10", "6"),
            ("11", "5"),
            ("11", "6"),
            ("12", "5"),
            ("12", "6"),
            ("2", "5"),
            ("2", "6"),
            ("3", "5"),
            ("3", "6"),
            ("4", "5"),
            ("4", "6"),
            ("5", "5"),
            ("5", "6"),
            ("6", "5"),
            ("6", "6"),
            ("7", "5"),
            ("7", "6"),
            ("8", "5"),
            ("8", "6"),
            ("9", "5"),
            ("9", "6"),
            ("<s>", "5"),
            ("<s>", "6"),
        }:
            return 0
        elif key in {
            ("0", "0"),
            ("0", "10"),
            ("1", "0"),
            ("1", "10"),
            ("10", "0"),
            ("10", "10"),
            ("11", "0"),
            ("11", "10"),
            ("12", "0"),
            ("12", "10"),
            ("2", "0"),
            ("2", "10"),
            ("3", "0"),
            ("3", "10"),
            ("4", "0"),
            ("4", "10"),
            ("5", "0"),
            ("5", "10"),
            ("6", "0"),
            ("6", "10"),
            ("7", "0"),
            ("7", "10"),
            ("8", "0"),
            ("8", "10"),
            ("9", "0"),
            ("9", "10"),
            ("<s>", "0"),
            ("<s>", "10"),
        }:
            return 10
        elif key in {
            ("0", "12"),
            ("1", "12"),
            ("10", "12"),
            ("11", "12"),
            ("12", "12"),
            ("2", "12"),
            ("3", "12"),
            ("4", "12"),
            ("5", "12"),
            ("6", "12"),
            ("7", "12"),
            ("8", "12"),
            ("9", "12"),
            ("<s>", "12"),
        }:
            return 8
        elif key in {
            ("</s>", "0"),
            ("</s>", "1"),
            ("</s>", "11"),
            ("</s>", "4"),
            ("</s>", "6"),
            ("</s>", "8"),
        }:
            return 7
        elif key in {("</s>", "10")}:
            return 3
        return 15

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_2_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_2_output, num_attn_1_7_output):
        key = (num_attn_2_2_output, num_attn_1_7_output)
        return 9

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_2_outputs, num_attn_1_7_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_2_output, num_attn_1_6_output):
        key = (num_attn_1_2_output, num_attn_1_6_output)
        return 4

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_6_outputs)
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
                attn_0_4_output_scores,
                attn_0_5_output_scores,
                attn_0_6_output_scores,
                attn_0_7_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
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
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
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
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
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


print(run(["<s>", "3", "3", "2", "4", "</s>"]))
