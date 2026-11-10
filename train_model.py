from trainer.pretrain import train, train_parser

if __name__ == "__main__":
    train(train_parser().parse_args())
