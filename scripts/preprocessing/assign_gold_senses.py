from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from pathlib import Path
from pprint import pp
from typing import List

from tqdm import tqdm
from jsonlines import open as open_jsonl

from resolve.training.data import (
    read_training_sentences,
    TrainingSentence,
    compute_summary_stats,
)
from resolve.training.mwe_preproc.common import (
    compute_output_path,
    fix_mwe_discontinuity,
)


@dataclass
class SenseAdditionStats:
    added_senses: int = 0
    multi_sense_additions: int = 0
    deletions: int = 0
    already_had_sense: int = 0
    pos_mismatch_overwrite: int = 0


def add_senses(
    sentences: List[TrainingSentence], lang: str
) -> tuple[List[TrainingSentence], SenseAdditionStats]:
    stats = SenseAdditionStats()
    for sent in tqdm(sentences, 'Adding gold senses'):
        # sent.get_mwe_groups() returns a list of elements, each of which corresponds to an MWE
        for sense_data, words in sent.get_mwe_groups():
            # sense_data is a SenseData object
            # words is a list of TrainingWord objects, e.g., [<pointing>, <out>]
            if sense_data.gold_sense is not None:
                stats.already_had_sense += 1
                continue

            try:
                sense_data = next(
                    word.mwe_sense_data
                    for word in words
                    if word.mwe_sense_data.get_definitions(lang) is not None
                )
                # Assign a value like 'point_out%2:32:01::' to sense_data.gold_sense
                # 0 is "label", which is the index of the synset in the synset list. Using 0 means that we use the most frequent sense
                sense_data.gold_sense = sense_data.label_to_key(0, fallback=True)
                for word in words:
                    word.mwe_sense_data = sense_data

                if len(set(word.mwe_sense_data.pos for word in words)) != 1:
                    stats.pos_mismatch_overwrite += 1

                definitions = sense_data.get_definitions(lang)
                if len(definitions) > 1:
                    stats.multi_sense_additions += 1

                stats.added_senses += 1
            except StopIteration:
                for word in words:
                    word.mwe_sense_data = None

                stats.deletions += 1

        fix_mwe_discontinuity(sent)

    return sentences, stats


def main():
    parser = ArgumentParser()
    parser.add_argument('input_path', type=Path)
    parser.add_argument('--lang', default='eng')

    args = parser.parse_args()

    sentences = list(read_training_sentences(args.input_path, None))
    print_summary_stats('Initial summary stats', sentences)

    sentences, stats = add_senses(sentences, args.lang)
    pp(asdict(stats))

    output = compute_output_path(args.input_path, 'sense')
    print(f'Writing updated data to {output}')
    with open_jsonl(output, 'w') as outfile:
        for training_sentence in sentences:
            outfile.write(training_sentence.to_json())

    print_summary_stats('Final summary stats', sentences)
    print('Done')


def print_summary_stats(title: str, sentences: List[TrainingSentence]):
    summary_stats = compute_summary_stats(sentences)
    print(title)
    pp(summary_stats.to_dict())


if __name__ == '__main__':
    main()
