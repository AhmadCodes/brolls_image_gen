INPUT_SCHEMA = {
    'word_level_transcript': {
        'type': list,
        'required': True,
    },
    'context_start_s' :
    {
        'type': int,
        'required': True
    },
    # context_end_s = 30
    # context_buffer_s = 5
    'context_end_s' : {
        'type': int,
        'required': True
    },
    'context_buffer_s' : {
        'type': int,
        'required': False,
    },
    
    'generation_steps': {
        'type': int,
        'required': False,
        "min": 10,
        "max": 50
    },
}
