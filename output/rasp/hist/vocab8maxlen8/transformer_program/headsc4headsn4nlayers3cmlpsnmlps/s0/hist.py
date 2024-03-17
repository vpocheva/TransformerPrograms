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
        "output/rasp/hist/vocab8maxlen8/transformer_program/headsc4headsn4nlayers3cmlpsnmlps/s0/hist_weights.csv",
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
    def predicate_0_0(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "<s>"
        elif q_token in {"1", "2", "3"}:
            return k_token == "1"
        elif q_token in {"4", "<s>"}:
            return k_token == "0"
        elif q_token in {"5"}:
            return k_token == "5"

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"4", "0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "<pad>"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "1"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == "4"

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, positions)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 6}:
            return token == "5"
        elif position in {1, 2}:
            return token == "<s>"
        elif position in {3, 7}:
            return token == "4"
        elif position in {4, 5}:
            return token == "0"

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "1"
        elif q_token in {"4", "1"}:
            return k_token == "0"
        elif q_token in {"5", "2", "<s>"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "<pad>"

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_token, k_token):
        if q_token in {"0"}:
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
        elif q_token in {"<s>"}:
            return k_token == "<pad>"

    num_attn_0_0_pattern = select(tokens, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "2"
        elif q_token in {"1"}:
            return k_token == "3"
        elif q_token in {"4", "5", "2"}:
            return k_token == "1"
        elif q_token in {"3"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == "<s>"

    num_attn_0_1_pattern = select(tokens, tokens, num_predicate_0_1)
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
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"<s>", "3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"

    num_attn_0_2_pattern = select(tokens, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"4", "0"}:
            return position == 1
        elif token in {"1", "3"}:
            return position == 3
        elif token in {"2"}:
            return position == 7
        elif token in {"5"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 0

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(token, position):
        key = (token, position)
        return 2

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(tokens, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_2_output, attn_0_3_output):
        key = (attn_0_2_output, attn_0_3_output)
        return 2

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_3_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output):
        key = num_attn_0_2_output
        return 1

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(one, num_attn_0_2_output):
        key = (one, num_attn_0_2_output)
        if key in {
            (0, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
            (3, 5),
            (4, 4),
            (4, 5),
            (4, 6),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 7),
            (6, 5),
            (6, 6),
            (6, 7),
            (7, 5),
            (7, 6),
            (7, 7),
        }:
            return 0
        elif key in {
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (2, 5),
            (2, 6),
            (2, 7),
            (3, 6),
            (3, 7),
            (4, 7),
        }:
            return 7
        elif key in {(1, 2), (3, 3), (6, 4)}:
            return 3
        return 1

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1) for k0, k1 in zip(ones, num_attn_0_2_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_num_mlp_0_1_output, k_num_mlp_0_1_output):
        if q_num_mlp_0_1_output in {0}:
            return k_num_mlp_0_1_output == 3
        elif q_num_mlp_0_1_output in {1}:
            return k_num_mlp_0_1_output == 5
        elif q_num_mlp_0_1_output in {2, 4, 5, 7}:
            return k_num_mlp_0_1_output == 7
        elif q_num_mlp_0_1_output in {3, 6}:
            return k_num_mlp_0_1_output == 0

    attn_1_0_pattern = select_closest(
        num_mlp_0_1_outputs, num_mlp_0_1_outputs, predicate_1_0
    )
    attn_1_0_outputs = aggregate(attn_1_0_pattern, num_mlp_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(mlp_0_0_output, num_mlp_0_0_output):
        if mlp_0_0_output in {0, 2, 3, 6}:
            return num_mlp_0_0_output == 0
        elif mlp_0_0_output in {1}:
            return num_mlp_0_0_output == 5
        elif mlp_0_0_output in {4}:
            return num_mlp_0_0_output == 6
        elif mlp_0_0_output in {5, 7}:
            return num_mlp_0_0_output == 3

    attn_1_1_pattern = select_closest(
        num_mlp_0_0_outputs, mlp_0_0_outputs, predicate_1_1
    )
    attn_1_1_outputs = aggregate(attn_1_1_pattern, num_mlp_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(token, num_mlp_0_0_output):
        if token in {"0", "2", "3"}:
            return num_mlp_0_0_output == 0
        elif token in {"1"}:
            return num_mlp_0_0_output == 2
        elif token in {"4"}:
            return num_mlp_0_0_output == 7
        elif token in {"5"}:
            return num_mlp_0_0_output == 5
        elif token in {"<s>"}:
            return num_mlp_0_0_output == 3

    attn_1_2_pattern = select_closest(num_mlp_0_0_outputs, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, num_mlp_0_0_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_attn_0_2_output, k_attn_0_2_output):
        if q_attn_0_2_output in {"1", "2", "4", "0", "3"}:
            return k_attn_0_2_output == "<pad>"
        elif q_attn_0_2_output in {"5"}:
            return k_attn_0_2_output == "<s>"
        elif q_attn_0_2_output in {"<s>"}:
            return k_attn_0_2_output == "2"

    attn_1_3_pattern = select_closest(attn_0_2_outputs, attn_0_2_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, positions)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(q_token, k_token):
        if q_token in {"5", "0"}:
            return k_token == "4"
        elif q_token in {"4", "1"}:
            return k_token == "2"
        elif q_token in {"2"}:
            return k_token == "3"
        elif q_token in {"3"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == "<pad>"

    num_attn_1_0_pattern = select(tokens, tokens, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_2_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_token, k_token):
        if q_token in {"0"}:
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
        elif q_token in {"<s>"}:
            return k_token == "<pad>"

    num_attn_1_1_pattern = select(tokens, tokens, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_1_output, mlp_0_0_output):
        if attn_0_1_output in {0}:
            return mlp_0_0_output == 7
        elif attn_0_1_output in {1}:
            return mlp_0_0_output == 6
        elif attn_0_1_output in {2, 3, 4, 5, 6}:
            return mlp_0_0_output == 0
        elif attn_0_1_output in {7}:
            return mlp_0_0_output == 4

    num_attn_1_2_pattern = select(mlp_0_0_outputs, attn_0_1_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_2_output, mlp_0_1_output):
        if attn_0_2_output in {"5", "0"}:
            return mlp_0_1_output == 3
        elif attn_0_2_output in {"1", "<s>"}:
            return mlp_0_1_output == 4
        elif attn_0_2_output in {"4", "2", "3"}:
            return mlp_0_1_output == 0

    num_attn_1_3_pattern = select(mlp_0_1_outputs, attn_0_2_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_3_output, attn_0_1_output):
        key = (attn_0_3_output, attn_0_1_output)
        return 6

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_1_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(num_mlp_0_1_output, num_mlp_0_0_output):
        key = (num_mlp_0_1_output, num_mlp_0_0_output)
        return 6

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, num_mlp_0_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_2_output, num_attn_0_0_output):
        key = (num_attn_0_2_output, num_attn_0_0_output)
        if key in {(0, 0)}:
            return 2
        return 7

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_2_output, num_attn_1_2_output):
        key = (num_attn_0_2_output, num_attn_1_2_output)
        if key in {(0, 2), (0, 3), (0, 4)}:
            return 5
        elif key in {(0, 0), (0, 1)}:
            return 7
        return 2

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, mlp_0_1_output):
        if token in {"5", "0", "1"}:
            return mlp_0_1_output == 3
        elif token in {"2", "<s>"}:
            return mlp_0_1_output == 5
        elif token in {"3"}:
            return mlp_0_1_output == 2
        elif token in {"4"}:
            return mlp_0_1_output == 4

    attn_2_0_pattern = select_closest(mlp_0_1_outputs, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, num_mlp_1_1_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_1_2_output, num_mlp_0_1_output):
        if attn_1_2_output in {0}:
            return num_mlp_0_1_output == 4
        elif attn_1_2_output in {1, 4, 5}:
            return num_mlp_0_1_output == 7
        elif attn_1_2_output in {2}:
            return num_mlp_0_1_output == 1
        elif attn_1_2_output in {3, 6}:
            return num_mlp_0_1_output == 6
        elif attn_1_2_output in {7}:
            return num_mlp_0_1_output == 0

    attn_2_1_pattern = select_closest(
        num_mlp_0_1_outputs, attn_1_2_outputs, predicate_2_1
    )
    attn_2_1_outputs = aggregate(attn_2_1_pattern, num_mlp_0_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(num_mlp_1_0_output, mlp_0_1_output):
        if num_mlp_1_0_output in {0, 1, 2, 3, 4, 5, 7}:
            return mlp_0_1_output == 2
        elif num_mlp_1_0_output in {6}:
            return mlp_0_1_output == 6

    attn_2_2_pattern = select_closest(
        mlp_0_1_outputs, num_mlp_1_0_outputs, predicate_2_2
    )
    attn_2_2_outputs = aggregate(attn_2_2_pattern, mlp_0_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_2_output, attn_1_1_output):
        if attn_0_2_output in {"4", "0"}:
            return attn_1_1_output == 1
        elif attn_0_2_output in {"1"}:
            return attn_1_1_output == 2
        elif attn_0_2_output in {"2"}:
            return attn_1_1_output == 4
        elif attn_0_2_output in {"5", "3"}:
            return attn_1_1_output == 6
        elif attn_0_2_output in {"<s>"}:
            return attn_1_1_output == 0

    attn_2_3_pattern = select_closest(attn_1_1_outputs, attn_0_2_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, mlp_1_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(token, attn_0_2_output):
        if token in {"0"}:
            return attn_0_2_output == "0"
        elif token in {"1"}:
            return attn_0_2_output == "1"
        elif token in {"2"}:
            return attn_0_2_output == "2"
        elif token in {"3"}:
            return attn_0_2_output == "3"
        elif token in {"4"}:
            return attn_0_2_output == "4"
        elif token in {"5"}:
            return attn_0_2_output == "<pad>"
        elif token in {"<s>"}:
            return attn_0_2_output == "<s>"

    num_attn_2_0_pattern = select(attn_0_2_outputs, tokens, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(num_mlp_1_0_output, num_mlp_1_1_output):
        if num_mlp_1_0_output in {0, 3}:
            return num_mlp_1_1_output == 3
        elif num_mlp_1_0_output in {1, 2, 6, 7}:
            return num_mlp_1_1_output == 7
        elif num_mlp_1_0_output in {4}:
            return num_mlp_1_1_output == 4
        elif num_mlp_1_0_output in {5}:
            return num_mlp_1_1_output == 5

    num_attn_2_1_pattern = select(
        num_mlp_1_1_outputs, num_mlp_1_0_outputs, num_predicate_2_1
    )
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_2_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(mlp_1_1_output, attn_0_2_output):
        if mlp_1_1_output in {0, 4}:
            return attn_0_2_output == "1"
        elif mlp_1_1_output in {1, 5, 6}:
            return attn_0_2_output == "3"
        elif mlp_1_1_output in {2, 7}:
            return attn_0_2_output == "2"
        elif mlp_1_1_output in {3}:
            return attn_0_2_output == "<pad>"

    num_attn_2_2_pattern = select(attn_0_2_outputs, mlp_1_1_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, ones)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_0_output, attn_0_2_output):
        if attn_0_0_output in {"0", "1"}:
            return attn_0_2_output == "<pad>"
        elif attn_0_0_output in {"2"}:
            return attn_0_2_output == "3"
        elif attn_0_0_output in {"4", "3"}:
            return attn_0_2_output == "2"
        elif attn_0_0_output in {"5"}:
            return attn_0_2_output == "1"
        elif attn_0_0_output in {"<s>"}:
            return attn_0_2_output == "0"

    num_attn_2_3_pattern = select(attn_0_2_outputs, attn_0_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, ones)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(num_mlp_0_0_output, num_mlp_1_1_output):
        key = (num_mlp_0_0_output, num_mlp_1_1_output)
        return 0

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(num_mlp_0_0_outputs, num_mlp_1_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(num_mlp_0_1_output, num_mlp_0_0_output):
        key = (num_mlp_0_1_output, num_mlp_0_0_output)
        return 3

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, num_mlp_0_0_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_2_output):
        key = num_attn_0_2_output
        return 7

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_0_2_outputs]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_0_2_output):
        key = num_attn_0_2_output
        if key in {0, 1}:
            return 2
        return 7

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_0_2_outputs]
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


print(run(["<s>", "1", "5", "1", "2", "0", "3"]))
