from resolve.training.data import SenseData


# Test SenseData
def test_label_to_key():
    sense_data = SenseData.from_json(
        {
            'lemma': 'red_ink',
            'pos': 'n',
            'idx': 0,  # doesn't matter
            'gold_sense': None,
        },
        manager=None,  # seems that this is the default value
        allow_single_def=False,  # seems that this is the default value
    )
    key = sense_data.label_to_key(0, fallback=True)

    assert key.split('%')[0] == 'red_ink'
