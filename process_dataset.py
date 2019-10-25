import csv


resize_width = 196
resize_height = 256


def process(dir):
    pass


def process_dataset(root_dir):
    headers = ['img', 'cloth']
    train_rows = []
    test_rows = []

    with open(root_dir + '/data/all_poseA_poseB_clothes_0607.csv') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            img_1 = row[0]
            img_2 = row[1]
            cloth = row[2]
            is_train = row[3] == 'train'

            # remove images tagged as back images
            if 'back' in img_1 or 'back' in img_2:
                continue

            if is_train:
                train_rows.append({'img': img_1, 'cloth': cloth})
                train_rows.append({'img': img_2, 'cloth': cloth})
            else:
                test_rows.append({'img': img_1, 'cloth': cloth})
                test_rows.append({'img': img_2, 'cloth': cloth})

    with open(root_dir + '/data/train_pairs.csv', 'w') as f:
        f_csv = csv.DictWriter(f, headers)
        f_csv.writeheader()
        f_csv.writerows(train_rows)

    with open(root_dir + '/data/test_pairs.csv', 'w') as f:
        f_csv = csv.DictWriter(f, headers)
        f_csv.writeheader()
        f_csv.writerows(test_rows)

    # for
