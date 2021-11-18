def define_label(label_list):
    #---------------------DEFINE LABEL-------------------------
    label_dict = {}
    label_dict_rev = {}
    for i in range(len(label_list)):
        label_dict[label_list[i]] = i
        label_dict_rev[i] = label_list[i]
    return label_dict, label_dict_rev