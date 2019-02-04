"""Prepares data for training.

Adapted from https://pytorch.org/tutorials/beginner/chatbot_tutorial.html

Copyright 2017, PyTorch.
"""
from __future__ import print_function

import argparse
import codecs
import csv
import os

import yaml

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=".")
    p.add_argument("--output-dir", default=".")
    return p.parse_args()

def main(args):
    corpus_name = "cornell movie-dialogs corpus"
    corpus = os.path.join(args.data_dir, corpus_name)

    # Define path to new file
    datafile = os.path.join(args.output_dir, "formatted_movie_lines.txt")

    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Initialize lines dict, conversations list, and field ids
    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = [
        "lineID",
        "characterID",
        "movieID",
        "character",
        "text"]
    MOVIE_CONVERSATIONS_FIELDS = [
        "character1ID",
        "character2ID",
        "movieID",
        "utteranceIDs"]

    # Load lines and process conversations
    print("Processing corpus...")
    lines = loadLines(
        os.path.join(corpus, "movie_lines.txt"),
        MOVIE_LINES_FIELDS)

    print("Loading conversations...")
    conversations = loadConversations(
        os.path.join(corpus, "movie_conversations.txt"),
        lines, MOVIE_CONVERSATIONS_FIELDS)

    # Write new csv file
    print("Writing newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(
            outputfile, delimiter=delimiter,
            lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

    # Print a sample of lines
    print("Sample lines from file:")
    printLines(datafile)

# Splits each line of the file into a dictionary of fields
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines

# Groups fields of lines from `loadLines` into conversations based
# on *movie_conversations.txt*

def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] ==
            # "['L598485', 'L598486', ...]")
            lineIds = yaml.safe_load(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations

# Extracts pairs of sentences from conversations

def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation - we ignore
        # the last line (no answer for it)
        for i in range(len(conversation["lines"]) - 1):
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs

def printLines(file, n=10):
    with open(file, 'r') as datafile:
        lines = [
            l.strip().replace("\t", "  ")
            for l in datafile.readlines()[:n]]
    print(yaml.dump(lines, default_flow_style=False))

if __name__ == "__main__":
    main(parse_args())
