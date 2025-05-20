"""
Fake-server đọc CIFAR-10 binary ⇒ gửi JSON qua socket TCP.
"""
import os, json, time, socket, pickle, argparse
import numpy as np
from tqdm import tqdm

TCP_IP   = "localhost"
TCP_PORT = 6100

def get_args():
    p = argparse.ArgumentParser("CIFAR-10 streamer")
    p.add_argument("-f","--folder",      required=True,  type=str)
    p.add_argument("-b","--batch-size",  required=True,  type=int)
    p.add_argument("-s","--split",       default="train",choices=["train","test"])
    p.add_argument("-t","--sleep",       default=3,      type=int)
    p.add_argument("-e","--endless",     action="store_true")
    return p.parse_args()

# --------- TCP helper ---------- #
def open_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
    sock.bind((TCP_IP, TCP_PORT))
    sock.listen(1)
    print(f"[Streamer]  listening on {TCP_PORT} …")
    conn, addr = sock.accept()
    print(f"[Streamer]  connected to {addr}")
    return conn

# --------- dataset helper ---------- #
def cifar_batches(folder:str, split:str):
    batches = [f"data_batch_{i}" for i in range(1,6)] if split=="train" else ["test_batch"]
    return [os.path.join(folder, b) for b in batches]

def load_one(file_path):
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    return batch[b"data"], batch[b"labels"]

def gen_payload(images, labels):
    # images: (N, 3072)
    payload = {i: {**{f"feature-{k}": int(v) for k,v in enumerate(img)},
                   "label": int(labels[i])} for i, img in enumerate(images)}
    return (json.dumps(payload)+"\n").encode()

# --------- main streaming loop --------- #
def stream_dataset(conn, folder, bs, split, sleep):
    for batch_file in cifar_batches(folder, split):
        imgs, labels = load_one(batch_file)
        imgs   = imgs.reshape(-1, 3072)
        total  = (len(imgs)//bs)*bs
        imgs   = imgs[:total]
        labels = labels[:total]

        pbar = tqdm(range(0, total, bs), desc=f"Streaming {batch_file}")
        for i in pbar:
            payload = gen_payload(imgs[i:i+bs], labels[i:i+bs])
            try:
                conn.send(payload)
            except BrokenPipeError:
                print("Socket closed by receiver")
                return
            time.sleep(sleep)

# --------------------------------------- #
if __name__ == "__main__":
    args  = get_args()
    conn  = open_socket()

    try:
        while True:
            stream_dataset(conn, args.folder, args.batch_size,
                           args.split, args.sleep)
            if not args.endless:
                break
    finally:
        conn.close()
