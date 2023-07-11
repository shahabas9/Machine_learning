import argparse



if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument('--name','-n',default='shahabas',type=str)
    args.add_argument('--age','-a',default=29,type=int)
    parse_args=args.parse_args()

    print(parse_args.name,parse_args.age)