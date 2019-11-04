import argparse
from dataset.synthetic.XOR_problem import xor_data
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synthetic Data generator')
    parser.add_argument('--type', type=str, default='xor',help='what type you want to generate')

    args = parser.parse_args()


    if(args.type == 'xor'):
        data = xor_data()
    elif(args.type == ''):
        file_name = ''
    else:
        print('There is no {0}. Plz select between "xor" and ""'.format(args.type))
    data.generate()


