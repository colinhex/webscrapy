# script for large scale web-scraping for text content curation with spaCy natural language processing.
# sorts out duplicates and connection errors

# imports
from concurrent.futures import \
    wait, ThreadPoolExecutor, ALL_COMPLETED, as_completed
from os import path
from pathlib import Path
import sys
import threading
import time
import warnings
import requests
import io
import queue
import argparse
from bs4 import BeautifulSoup
from bs4.element import Comment
import spacy as sp


def info_print(estimated_time):
    """Displays time passed."""
    global url_good
    global url_bad
    global unique_url_set
    global duplicates
    global lines_read
    last_time = time.time()

    print('Info...')
    print('\tURLs (+) - successfully scraped and processed urls.')
    print('\tURLs (-) - urls failed to process for any reason.')
    print('\tT - time that has passed.')
    print('\tds - size of the set of unique urls.')
    print('\tdh - duplicate hits')
    print('\tlr - total lines read in input file.')
    print('\n')

    while not finished:
        new_time = time.time()
        print('\tURLs (+): {}, URLs (-): {},  T:  {}/{}, ds: {}, dh {}, lr {}'.format(
            url_good, url_bad, str(int(new_time - last_time)) + " sec", estimated_time,
            len(unique_url_set), duplicates, lines_read))
        time.sleep(5)


def write_results():
    while not finished or not good_queue.empty():
        result = good_queue.get()
        w.write(result)
        global url_good
        url_good += 1
        good_queue.task_done()


def write_exceptions():
    while not finished or not bad_queue.empty():
        result = bad_queue.get()
        bw.write(result)
        global url_bad
        url_bad += 1
        bad_queue.task_done()


def load_url(url, timeout, headers):
    """Send get request to website."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # verify false is okay here for this project
            return url, requests.get(url, headers=headers, timeout=timeout, verify=False), None
    except Exception as exception:
        return url, None, exception


def handle_response(futures):
    """Write results to file."""
    for future in as_completed(futures):
        url, response, err = future.result()
        if response:
            good_queue.put(url + "," + textify(response.content) + "\n")
        else:
            bad_queue.put(url + "," + str(err) + "\n")


def read_urls(number: int):
    """Get urls from dataset."""
    global duplicates
    global lines_read
    url_list = []
    for n in range(0, number):
        # Assuming plaintext, one url per line.
        _line = r.readline()
        lines_read += 1

        if not _line:
            print('Reader returned empty line.')
            return None
        elif unique_url_set.__contains__(_line):
            url_list.append(_line)
            unique_url_set.add(_line)
        else:
            duplicates += 1

    # project specific
    # ----------------------------
    # if len(url_list) == 0:
    #     return ['#']
    # ----------------------------

    return url_list


def textify(content):
    """
        Scrapes text from html content and processes language tokens.
        :param content:     html content of http response
        """
    soup = BeautifulSoup(content, "html.parser")
    tags = soup.find_all(text=True)
    result = []
    for tag in tags:
        stripped_tag = tag.strip()
        if tag.parent.name not in tags_to_ignore \
                and not isinstance(tag, Comment) \
                and not stripped_tag.isnumeric() \
                and len(stripped_tag) > 0:
            result.append(stripped_tag)

    text = nlp(' '.join(result))
    tokens = []
    exclusion_list = ["nan"]
    for token in text:
        if token.is_stop \
                or token.is_punct \
                or token.text.isnumeric() \
                or not token.text.isalnum() \
                or token.text in exclusion_list:
            continue
        token = str(token.lemma_.lower().strip())
        tokens.append(token)
    return " ".join(tokens)


def scrape(
        conn: int,
        nlps: int,
        urlr: int,
        tout: int,
        insl: int,
        urlt: int
) -> None:
    """
    Runs a scraping session. Having the NLP threads run concurrently is not ideal but
    it does the job # Todo separate
    :param conn: concurrent connections
    :param nlps: concurrent nlp threads
    :param urlr: urls read per cycle
    :param tout: timeout on website
    :param insl: intermediate sleep
    :param urlt: total number of urls
    :return:
    """
    print("Initializing Threads...")

    # run thread that writes down results
    result_writer = threading.Thread(target=write_results)
    result_writer.start()

    # run thread that writes down bad results (timeouts ect.)
    exception_writer = threading.Thread(target=write_exceptions)
    exception_writer.start()

    # create connection thread pool
    connection_executor = ThreadPoolExecutor(max_workers=conn)
    # create nlp thread pool
    nlp_executor = ThreadPoolExecutor(max_workers=nlps)

    # run a thread to display timer (for now)
    print("""
    Running scraping session...
    \tconcurrent connections: {}
    \tconcurrent nlp threads: {}
    \turls read per cycle: {}
    \ttimeout on website: {}
    \tintermediate sleep: {}
    \ttotal number of urls: {}\n
    """.format(
        conn,
        nlps,
        urlr,
        tout,
        insl,
        urlt
    ))
    print("Estimated max duration of scraping: {}\n".format('Unknown' if urlt == 0 else str(int(urlt / 4.0)) + " sec"))
    timer_thread = threading.Thread(target=info_print, args=['Unknown' if urlt == 0 else str(int(urlt / 4.0)) + " sec"])
    timer_thread.setDaemon(True)
    timer_thread.start()

    start = time.time()

    url_total = int(float('inf') if urlt == 0 else urlt)
    url_count = 0
    while True:
        urls = read_urls(urlr)
        url_count += urlr  # just for cl output

        if not urls:
            print("Reached the end of the file.")
            break

        # project specific
        # ---------------------------------------------------
        # elif urls[0] == '#':
        #     print('Skipping high number of duplicates...')
        # ---------------------------------------------------

        else:
            time.sleep(insl)  # sleep a few seconds

            # deal with urls
            futures = [connection_executor.submit(load_url, url, tout, REQ_HEADERS) for url in urls]

            nlp_executor.submit(
                handle_response, futures
            )

            if url_count >= url_total:
                print("Reached total number of URLs.")
                break

    print("Waiting for nlp & connection threads to complete...")
    wait(futures, return_when=ALL_COMPLETED)
    nlp_executor.shutdown(wait=True)  # waits until all texts done
    global finished
    finished = True

    print("Waiting for writer threads to complete...")

    end = time.time()

    time.sleep(5)

    # write report
    print("""
    Finished Scraping Session:\n
    ---------------------------\n
    Total Time: {},\n
    Successfully scraped: {}/{}\n
    Lines Read: {}
    """.format(
        (end - start),
        url_good,
        urlt,
        lines_read
    ))


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser('Text cleaning web-scrape.')
    parser.add_argument('-i', "--input", type=str, required=True,
                        help='file with one url per line.')
    parser.add_argument('-o', "--output", type=str, required=True,
                        help='output file.')
    parser.add_argument('-conn', '--connections', type=int, required=False, default=100,
                        help='number of concurrent connections.')
    parser.add_argument('-nlps', '--nlp_threads', type=int, required=False, default=2,
                        help='number of concurrent nlp threads.')
    parser.add_argument('-urlr', '--urls_read', type=int, required=False, default=100,
                        help='number of urls read from file per loop.')
    parser.add_argument('-insl', '--in_sleep', type=float, required=False, default=0.5,
                        help='seconds slept between reading of urls.')
    parser.add_argument('-urlt', '--total_urls', type=int, required=False, default=0,
                        help='max cap on urls read from file.')
    parser.add_argument('-sk', '--skip', type=int, required=False, default=0,
                        help='skip a number of urls in the file.')
    parser.add_argument('-ap', '--append_mode', type=bool, required=False, default=False,
                        help='sets programm to append or overwrite results.')
    parser.add_argument('-to', '--timeout', type=int, required=False, default=5,
                        help='sets timeout for website responses.')

    args = parser.parse_args()

    # check integrity of (some) arguments
    if not path.exists(args.input):
        print("Could not find the input file.")
        sys.exit(1)

    output_filename = path.basename(path.normpath(args.output))

    # initialize global unique url set
    unique_url_set = set()  # prevents duplicates
    duplicates = 0  # counter for individual duplicate hits.

    if path.exists(args.output):
        print("The output file already exists and program is set to: {}... Continue? (y/n)".format(
            'OVERWRITE' if not args.append_mode else 'APPEND'))
        answer = input(">> ")
        if answer != 'y':
            sys.exit(0)
        if args.append_mode:
            sr = io.open(args.output)
            line = '#'
            while line:
                line = sr.readline().split(',')[0]
                unique_url_set.add(line)
            sr.flush()
            sr.close()
    else:
        print("Creating a new output file: {}".format(output_filename))

    print("Opening files...")
    # open files to read and write to
    r = io.open(args.input)
    w = io.open(args.output, 'a' if args.append_mode else 'w')
    bw = io.open(Path(path.normpath(args.output).__str__()[0: -len(output_filename)],
                      'bad_urls_session_1.txt'), 'a' if args.append_mode else 'w')

    # skip lines if requested
    lines_read = 0  # lines read in input file
    if args.skip != 0:
        print("Skipping {} lines...".format(args.skip))
        for i in range(0, args.skip):
            lines_read += 1
            try:
                r.readline()
            except Exception as exc:
                print('Exception occured during reading...')
                print(exc)
                print('Line: ' + str(lines_read))
                exit(1)

    # Request headers for get request
    REQ_HEADERS = {
        "Accept-Language": "en-US, en;q=0.9",
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:32.0) Gecko/20100101 Firefox/32.0'
    }

    url_good = 0  # good urls count
    url_bad = 0  # bad urls count

    # Tags to ignore when scraping text from html.
    tags_to_ignore = ['style', 'script', 'head', '[document]', "h1", "h2", "h3", "h4", "h5", "h6",
                      "noscript"]

    print("Attempt to switch to gpu for nlp...")
    # load natural language processing core for english
    nlp = sp.load('en_core_web_sm')

    # automatically switch to gpu if configured
    if sp.prefer_gpu():
        print("\tusing gpu...")
    else:
        print("\tusing cpu...")

    # initialize global queues
    good_queue = queue.Queue()  # out - queue
    bad_queue = queue.Queue()  # out - queue

    # coordinate end of threads
    finished = False

    # Run scraping session
    scrape(
        args.connections,
        args.nlp_threads,
        args.urls_read,
        args.timeout,
        args.in_sleep,
        args.total_urls
    )





