"""
Contains some simple tests.
The purpose of this tests is to detect crashes and hangs
but NOT to guarantee the corectness of the operations.
For this we want another set of testcases using pytest.

Due to my lazy coding, DON'T USE PATHES WITH BLANKS !
"""

import sys
from subprocess import check_call, CalledProcessError
import urllib
from urllib.request import urlretrieve
import os
from os.path import join as pathjoin, expanduser

FAIL_COUNT = 0
TEST_COUNT = 0
_COLORS = {
    "FAIL": "\033[1;31m",
    "OK": "\033[1;32m",
    "STATUS": "\033[1;37m",
    "BOLD": "\033[1m",
    "ENDC": "\033[0m"
}


def print_colored(text, color="OK", bold=False):
    """ Print colored text
    This might not work on windows,
    although travis runs windows stuff in git bash, so it might ?
    """
    color = _COLORS.get(color, color)
    print("%s%s%s%s" % (
        color, "" if not bold else _COLORS["BOLD"], text, _COLORS["ENDC"]
    ))


def print_ok(text):
    """ Print ok in colored text """
    print_colored(text, "OK", True)


def print_fail(text):
    """ Print fail in colored text """
    print_colored(text, "FAIL", True)


def print_status(text):
    """ Print status in colored text """
    print_colored(text, "STATUS", True)


def run_test(name, cmd):
    """ run a test """
    global FAIL_COUNT, TEST_COUNT  # pylint:disable=global-statement
    print_status("[?] running %s" % name)
    print("Cmd: %s" % " ".join(cmd))
    TEST_COUNT += 1
    try:
        check_call(cmd)
        print_ok("[+] Test success")
        return True
    except CalledProcessError as err:
        print_fail("[-] Test failed with %s" % err)
        FAIL_COUNT += 1
        return False


def download_file(url, filename):  # TODO: retry
    """ Download a file from given url """
    if os.path.isfile(filename):
        print_status("[?] '%s' already cached as '%s'" % (url, filename))
        return filename
    try:
        print_status("[?] Downloading '%s' to '%s'" % (url, filename))
        video, _ = urlretrieve(url, filename)
        return video
    except urllib.error.URLError as err:
        print_fail("[-] Failed downloading: %s" % err)
        return None


def extract_args(detector, aligner, in_path, out_path, args=None):
    """ Extraction command """
    py_exe = sys.executable
    _extract_args = "%s faceswap.py extract -i %s -o %s -D %s -A %s" % (
        py_exe, in_path, out_path, detector, aligner
    )
    if args:
        _extract_args += " %s" % args
    return _extract_args.split()


def train_args(model, model_path, faces, alignments, iterations=5, batchsize=8, extra_args=""):
    """ Train command """
    py_exe = sys.executable
    args = "%s faceswap.py train -A %s -ala %s -B %s -alb %s -m %s -t %s -bs %i -it %s %s" % (
        py_exe, faces, alignments, faces,
        alignments, model_path, model, batchsize, iterations, extra_args
    )
    return args.split()


def convert_args(in_path, out_path, model_path, writer, args=None):
    """ Convert command """
    py_exe = sys.executable
    conv_args = "%s faceswap.py convert -i %s -o %s -m %s -w %s" % (
        py_exe, in_path, out_path, model_path, writer
    )
    if args:
        conv_args += " %s" % args
    return conv_args.split()  # Don't use pathes with spaces ;)


def sort_args(in_path, out_path, sortby="face", groupby="hist", method="rename"):
    """ Sort command """
    py_exe = sys.executable
    _sort_args = "%s tools.py sort -i %s -o %s -s %s -fp %s -g %s -k" % (
        py_exe, in_path, out_path, sortby, method, groupby
    )
    return _sort_args.split()


def main():
    """ Main testing script """
    vid_src = "https://faceswap.dev/data/test.mp4"
    img_src = "https://archive.org/download/GPN-2003-00070/GPN-2003-00070.jpg"
    base_dir = pathjoin(expanduser("~"), "cache", "tests")

    vid_base = pathjoin(base_dir, "vid")
    img_base = pathjoin(base_dir, "imgs")
    os.makedirs(vid_base, exist_ok=True)
    os.makedirs(img_base, exist_ok=True)
    py_exe = sys.executable
    was_trained = False

    vid_path = download_file(vid_src, pathjoin(vid_base, "test.mp4"))
    if not vid_path:
        print_fail("[-] Aborting")
        sys.exit(1)
    vid_extract = run_test(
        "Extraction video with cv2-dnn detector and cv2-dnn aligner.",
        extract_args("Cv2-Dnn", "Cv2-Dnn", vid_path, pathjoin(vid_base, "faces"))
    )

    img_path = download_file(img_src, pathjoin(img_base, "test_img.jpg"))
    if not img_path:
        print_fail("[-] Aborting")
        sys.exit(1)
    run_test(
        "Extraction images with cv2-dnn detector and cv2-dnn aligner.",
        extract_args("Cv2-Dnn", "Cv2-Dnn", img_base, pathjoin(img_base, "faces"))
    )

    if vid_extract:
        run_test(
            "Sort faces.",
            sort_args(
                pathjoin(vid_base, "faces"), pathjoin(vid_base, "faces_sorted"),
                sortby="face", method="rename"
            )
        )

        run_test(
            "Rename sorted faces.",
            (
                py_exe, "tools.py", "alignments", "-j", "rename",
                "-a", pathjoin(vid_base, "test_alignments.fsa"),
                "-fc", pathjoin(vid_base, "faces_sorted"),
            )
        )

        run_test(
            "Train lightweight model for 1 iteration with WTL.",
            train_args(
                "lightweight", pathjoin(vid_base, "model"),
                pathjoin(vid_base, "faces"), pathjoin(vid_base, "test_alignments.fsa"),
                iterations=1, extra_args="-wl"
            )
        )

        was_trained = run_test(
            "Train lightweight model for 5 iterations WITHOUT WTL.",
            train_args(
                "lightweight", pathjoin(vid_base, "model"),
                pathjoin(vid_base, "faces"), pathjoin(vid_base, "test_alignments.fsa")
            )
        )

    if was_trained:
        run_test(
            "Convert video.",
            convert_args(
                vid_path, pathjoin(vid_base, "conv"),
                pathjoin(vid_base, "model"), "ffmpeg"
            )
        )

        run_test(
            "Convert images.",
            convert_args(
                img_base, pathjoin(img_base, "conv"),
                pathjoin(vid_base, "model"), "opencv"
            )
        )

    if FAIL_COUNT == 0:
        print_ok("[+] Failed %i/%i tests." % (FAIL_COUNT, TEST_COUNT))
        sys.exit(0)
    else:
        print_fail("[-] Failed %i/%i tests." % (FAIL_COUNT, TEST_COUNT))
        sys.exit(1)


if __name__ == '__main__':
    main()
