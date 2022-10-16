import wget
import os
import sys
import shutil
import tarfile
import argparse


urls = {
    "dataset": "https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz",
    "model": "https://github.com/mlcommons/tiny/blob/master/v0.5/training/visual_wake_words/trained_models/vww_96.h5"
}


def download_file(type_, url, path):
    def bar_progress(current, total, width=80):
        progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
        # Don't use print() as it will print in new line every time.
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()

    def members(tf):
        l = len("vw_coco2014_96/")
        for member in tf.getmembers():
            if member.path.startswith("vw_coco2014_96/"):
                member.path = member.path[l:]
                yield member

    try:
        os.remove('/tmp/onnc_dataset')
    except:
        pass

    if type_ == "dataset":
        try:
            wget.download(url, '/tmp/onnc_dataset', bar=bar_progress)
        except Exception as e:
            print(e)
            raise Exception(f"{url} Download failure")

        if url.endswith('.gz'):
            print('Extracting...')

            with tarfile.open('/tmp/onnc_dataset') as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, path)
                tar.close()

            os.remove('/tmp/onnc_dataset')
        else:
            shutil.move('/tmp/onnc_dataset', path)
        print('Done')
    else:
        try:
            wget.download(url, path, bar=bar_progress)
        except Exception as e:
            print(e)
            raise Exception(f"{url} Download failure")


def parse_args():
    # type: () -> Args
    parser = argparse.ArgumentParser(
        description='ArgumentParser')
    parser.add_argument(
        '-t',
        '--type',
        required=True,
        choices=('dataset', 'model', 'samples'),
        help='Type of file to download.')

    parser.add_argument(
        '-o',
        '--output',
        default=None,
        help='Output path.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.type == 'dataset':
        download_file(args.type, urls[args.type], f"./{args.type}")
    elif args.type == 'model':
        download_file(args.type, urls[args.type], f"./{args.type}.h5")
    elif args.type == 'samples':
        download_file(args.type, urls[args.type], f"./{args.type}.npy")

