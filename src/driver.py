# Driver Script

import argparse

import process_data

def main():
    parser = argparse.ArgumentParser(
        description=('Neural network architecture to predict '
            'secondary structure in proteins'))
    parser.add_argument('filename', help='Path to file of protein data')
    args = parser.parse_args()
    train_data, dev_data, test_data = process_data.convert_data_labels(args.filename)
    


if __name__ == '__main__':
    main()
