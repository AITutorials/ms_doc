import pandas as pd

path = "./dev_sent_emo.csv"
dev_list = pd.read_csv(path).values.tolist()
print(dev_list)


with open("dev_sent_emo.jsonl", "w") as f:
    for dl in dev_list:
        if dl[4] != "neutral":
            if dl[4] == "negative":
                label = 0
            else:
                label = 1
            f.write(
                str(
                    {
                        "id": dl[0],
                        "text": dl[1],
                        "img": "./dev_sent_emo/dia"
                        + str(dl[5])
                        + "_utt"
                        + str(dl[6])
                        + ".jpg",
                        "label": label,
                    }
                )
                + "\n"
            )
