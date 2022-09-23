import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import multiprocessing

import dataset
import feature_extraction
import myconfig


class BaseSpeakerEncoder(nn.Module):
    # 从硬盘中去加载一个已经训练好的模型
    def _load_from(self, saved_model):
        var_dict = torch.load(saved_model, map_location=myconfig.DEVICE)
        self.load_state_dict(var_dict["encoder_state_dict"])


class LstmSpeakerEncoder(BaseSpeakerEncoder):

    # 初始化函数
    def __init__(self, saved_model=""):
        super(LstmSpeakerEncoder, self).__init__()
        # Define the LSTM network.
        self.lstm = nn.LSTM(
            input_size=myconfig.N_MFCC,
            hidden_size=myconfig.LSTM_HIDDEN_SIZE,
            num_layers=myconfig.LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=myconfig.BI_LSTM)

        # Load from a saved model if provided.
        if saved_model:
            self._load_from(saved_model)

    def _aggregate_frames(self, batch_output):
        """Aggregate output frames."""
        # 平均值池化
        if myconfig.FRAME_AGGREGATION_MEAN:
            return torch.mean(
                batch_output, dim=1, keepdim=False)
        # 最后帧
        else:
            return batch_output[:, -1, :]

    # 正向传播函数
    def forward(self, x):
        # LSTM是不是双向LSTM
        # print("x.shape:", x.shape)

        D = 2 if myconfig.BI_LSTM else 1

        # 初始状态h0,c0
        h0 = torch.zeros(
            D * myconfig.LSTM_NUM_LAYERS, x.shape[0],  myconfig.LSTM_HIDDEN_SIZE
        ).to(myconfig.DEVICE)

        c0 = torch.zeros(
            D * myconfig.LSTM_NUM_LAYERS, x.shape[0], myconfig.LSTM_HIDDEN_SIZE
        ).to(myconfig.DEVICE)

        # 根据初始状态(h0,c0)和输入x，返回输出y，和新的状态hn,cn
        y, (hn, cn) = self.lstm(x, (h0, c0))

        # print("y:", y.shape)
        # print("y_a:", self._aggregate_frames(y).shape)

        # 将输出y聚合为声纹嵌入码
        return self._aggregate_frames(y)


class TransformerSpeakerEncoder(BaseSpeakerEncoder):

    def __init__(self, saved_model=""):
        super(TransformerSpeakerEncoder, self).__init__()
        # Define the Transformer network.
        self.linear_layer = nn.Linear(myconfig.N_MFCC, myconfig.TRANSFORMER_DIM)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=myconfig.TRANSFORMER_DIM, nhead=myconfig.TRANSFORMER_HEADS,
            batch_first=True),
            num_layers=myconfig.TRANSFORMER_ENCODER_LAYERS)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=myconfig.TRANSFORMER_DIM, nhead=myconfig.TRANSFORMER_HEADS,
            batch_first=True),
            num_layers=1)

        # Load from a saved model if provided.
        if saved_model:
            self._load_from(saved_model)

    def forward(self, x):
        encoder_input = torch.sigmoid(self.linear_layer(x))
        encoder_output = self.encoder(encoder_input)
        tgt = torch.zeros(x.shape[0], 1, myconfig.TRANSFORMER_DIM).to(
            myconfig.DEVICE)
        output = self.decoder(tgt, encoder_output)
        return output[:, 0, :]


def get_speaker_encoder(load_from=""):
    """Create speaker encoder model or load it from a saved model."""
    if myconfig.USE_TRANSFORMER:
        return TransformerSpeakerEncoder(load_from).to(myconfig.DEVICE)
    else:
        return LstmSpeakerEncoder(load_from).to(myconfig.DEVICE)


def get_triplet_loss(anchor, pos, neg):
    """Triplet loss defined in https://arxiv.org/pdf/1705.02304.pdf."""
    # 实现了上篇论文中定义的三元损失函数
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    # 使用余弦相似度
    return torch.maximum(
        cos(anchor, neg) - cos(anchor, pos) + myconfig.TRIPLET_ALPHA,
        torch.tensor(0.0))


def get_triplet_loss_from_batch_output(batch_output, batch_size):
    """Triplet loss from N*(a|p|n) batch output."""
    # 计算一个数据批次的输出对应的损失函数
    batch_output_reshaped = torch.reshape(
        batch_output, (batch_size, 3, batch_output.shape[1]))
    # print(batch_output_reshaped.shape)

    batch_loss = get_triplet_loss(
        batch_output_reshaped[:, 0, :],
        batch_output_reshaped[:, 1, :],
        batch_output_reshaped[:, 2, :])
    loss = torch.mean(batch_loss)
    return loss


def save_model(saved_model_path, encoder, losses, start_time):
    """Save model to disk."""
    training_time = time.time() - start_time
    os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
    if not saved_model_path.endswith(".pt"):
        saved_model_path += ".pt"
    torch.save({"encoder_state_dict": encoder.state_dict(),
                "losses": losses,
                "training_time": training_time},
               saved_model_path)


def train_network(spk_to_utts, num_steps, saved_model=None, pool=None):
    start_time = time.time()
    # 用来记录损失函数的列表
    losses = []
    # 1、创建神经网络的函数（transformer or lstm）
    encoder = get_speaker_encoder()

    # 2、创建Adam优化器
    optimizer = optim.Adam(encoder.parameters(), lr=myconfig.LEARNING_RATE)
    print("Start training")

    # 3、根据训练步数进行主循环
    for step in range(num_steps):
        # 梯度清零
        optimizer.zero_grad()

        # 4、从spk_to_utts的字典中构建一个训练数据的批次
        '''
        myconfig.BATCH_SIZE：批次的大小
        pool：多进程的poll
        batch_input：接收返回的该批次训练数据的张量
        '''

        # batch_input.shape: (24, 100, 40)
        batch_input = feature_extraction.get_batched_triplet_input(
            spk_to_utts, myconfig.BATCH_SIZE, pool).to(myconfig.DEVICE)
        # print(batch_input.shape)

        # 5、使用训练数据批次来调用声纹编码器神经网络的推理，返回数据中所有utt的嵌入码
        # batch_output.shape:（24，128）
        batch_output = encoder(batch_input)
        # print(batch_output.shape)

        # 6、计算三元损失函数
        loss = get_triplet_loss_from_batch_output(
            batch_output, myconfig.BATCH_SIZE)

        # 7、通过反向传播来计算损失函数的梯度
        loss.backward()

        # 8、根据计算的梯度来优化神经网络的参数
        optimizer.step()
        losses.append(loss.item())
        print("step:", step, "/", num_steps, "loss:", loss.item())

        if (saved_model is not None and
                (step + 1) % myconfig.SAVE_MODEL_FREQUENCY == 0):
            checkpoint = saved_model
            if checkpoint.endswith(".pt"):
                checkpoint = checkpoint[:-3]
            checkpoint += ".ckpt-" + str(step + 1) + ".pt"
            save_model(checkpoint,
                       encoder, losses, start_time)

    training_time = time.time() - start_time
    print("Finished training in", training_time, "seconds")
    if saved_model is not None:
        save_model(saved_model, encoder, losses, start_time)
    return losses


def run_training():

    # 构建数据集，得到spk_to_utts的字典结构
    if myconfig.TRAIN_DATA_CSV:
        spk_to_utts = dataset.get_csv_spk_to_utts(
            myconfig.TRAIN_DATA_CSV)
        print("Training data:", myconfig.TRAIN_DATA_CSV)
    else:
        spk_to_utts = dataset.get_librispeech_spk_to_utts(
            myconfig.TRAIN_DATA_DIR)
        print("Training data:", myconfig.TRAIN_DATA_DIR)

    # 调用multiprocess.Pool 通过多进程来更好的利用电脑上的多个CPU核，使模型训练高效
    with multiprocessing.Pool(myconfig.NUM_PROCESSES) as pool:
        losses = train_network(spk_to_utts,
                               myconfig.TRAINING_STEPS,
                               myconfig.SAVED_MODEL_PATH,
                               pool)  # 多进程的pool
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()


if __name__ == "__main__":
    run_training()
