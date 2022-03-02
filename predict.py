import io
import sys
import threading
import time
from os import path
import pickle
from sklearn.calibration import CalibratedClassifierCV
import argparse


def info_print():
    global j
    global k
    while True:
        time.sleep(5)
        print("URLs (+): {}, URLs (-): {}".format(j, k))


def predict():
    global args
    global j
    global k

    while True:

        line = r.readline()

        if not line:
            print("Reached end of file....")
            break

        split_line = line.split(',')

        url, text = split_line[0], split_line[1]

        if text:
            results = [url]

            predictions = model.predict_proba(vector.transform([text]))[0]
            for prediction in predictions:
                results.append(str(prediction))

            j += 1

            w.write(','.join(results) + '\n')

            if not args.total_texts == 0 and j >= args.total_texts:
                print("Reached Total...")
                break
        else:
            k += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="binary serialized prediction model.")
    parser.add_argument('-v', '--vector', type=str, required=True,
                        help="binary serialized prediction model.")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="input file with urls and scraped-prepped text.")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="output file to write results into.")
    parser.add_argument('-ap', '--append_mode', type=bool, required=False, default=False,
                        help='sets programm to append or overwrite results.')
    parser.add_argument('-sk', '--skip', type=int, required=False, default=0,
                        help='skip a number of urls in the file.')
    parser.add_argument('-tt', '--total_texts', type=int, required=False, default=0,
                        help='number of texts to predict')

    args = parser.parse_args()

    # check integrity of (some) arguments
    if not path.exists(args.input):
        print("Could not find the input file.")
        sys.exit(1)

    output_filename = path.basename(path.normpath(args.output))

    if path.exists(args.output):
        print("The output file already exists and program is set to: {}... Continue? (y/n)".format(
            'OVERWRITE' if not args.append_mode else 'APPEND'))
        answer = input(">> ")
        if answer != 'y':
            sys.exit(0)
    else:
        print("Creating a new output file: {}".format(output_filename))

    print("Opening files...")
    # open files to read and write to
    r = io.open(args.input)
    w = io.open(args.output, 'a' if args.append_mode else 'w')
    # load ml classes for prediction
    model: CalibratedClassifierCV = pickle.loads(open(args.model, "rb").read())
    vector: object = pickle.loads(open(args.vector, "rb").read())

    # skip lines if requested
    if args.skip != 0:
        print("Skipping {} lines...".format(args.skip))
        for i in range(0, args.skip):
            r.readline()
    j = 0
    k = 0

    thread = threading.Thread(target=info_print)
    thread.setDaemon(True)
    thread.start()

    # run prediction model
    predict()


