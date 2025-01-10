
def ltr_parts(parts_dict):
    # When we flip image left parts became right parts and vice versa.
    # This is the list of parts to exchange each other.
    # left_parts = [parts_dict[p] for p in []]
    # left_parts = [parts_dict[p] for p in ["side"]]
    left_parts = [parts_dict[p] for p in ["side", "side_0", "side_1"]]
    right_parts = [parts_dict[p] for p in []]
    return left_parts, right_parts


class RmpeGlobalConfig:

    width = 368
    height = 368

    stride = 8

    # parts = ["top"]
    # parts = ["top", "side"]
    parts = ["top", "side", "side_0", "side_1"]
    num_parts = len(parts)
    parts_dict = dict(zip(parts, range(num_parts)))
    parts += ["background"]
    num_parts_with_background = len(parts)

    left_parts, right_parts = ltr_parts(parts_dict)

    heat_layers = num_parts
    num_layers = heat_layers + 1

    heat_start = 0
    bkg_start = heat_layers

    data_shape = (3, height, width)  # 3, 368, 368
    mask_shape = (height//stride, width//stride)  # 46, 46
    parts_shape = (num_layers, height//stride, width//stride)  # 57, 46, 46

    box_types = [["blue", "BoxBlue"], ["yellow", "BoxYellow"],
                 ["green", "BoxGreen"]]
    box_type_idx = 2


class TransformationParams:

    target_dist = 0.6
    # TODO: this is actually scale unprobability, i.e. 1 = off, 0 = always.
    # Not sure if it is a bug or not
    scale_prob = 0  # 1
    scale_min = 0.5
    scale_max = 1.1
    max_rotate_degree = 40.
    center_perterb_max = 40.
    flip_prob = 0.5
    sigma = 7.
    paf_thre = 8.  # It is original 1.0 * stride in this program


# more information on keypoints mapping is here
# https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/7


def check_layer_dictionary():

    dct = RmpeGlobalConfig.parts[:]
    dct = [None]*(RmpeGlobalConfig.num_layers-len(dct)) + dct

    for (i, (fr, to)) in enumerate(RmpeGlobalConfig.limbs_conn):
        name = "%s->%s" % (RmpeGlobalConfig.parts[fr],
                           RmpeGlobalConfig.parts[to])
        print(i, name)
        x = i*2
        y = i*2+1

        assert dct[x] is None
        dct[x] = name + ":x"
        assert dct[y] is None
        dct[y] = name + ":y"

    print(dct)


if __name__ == "__main__":
    check_layer_dictionary()
