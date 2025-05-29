DEFAULT_SAMPLE_RATE = 32000 #try 22050 as in tabcnn
DEFAULT_HOP_WIDTH = 128 #hopsize 512 was used in tabcnn
DEFAULT_NUM_MEL_BINS = 192#512 #192 in cqt, if you do this change sie of input embedding layer
FFT_SIZE = 2048
MEL_LO_HZ = 20.0
MEL_FMIN = 20.0
MEL_FMAX = 8000#15900#7600.0
#tweak these, sample rate should be higher and fmin and fmax adjusted to the piano keyboard range
#we want a higher max freq to capture unique frequencies for guitar strings

#resample in advance -> in place is slow


#mess aroudn with cqt as well -> 192 bins, 25 bins per octave

RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_VEL = 128
RANGE_TIME_SHIFT = 205

RANGE_FRET = 24
RANGE_STRING = 6

START_IDX = {
    'note_on': 0,
    'note_off': RANGE_NOTE_ON,
    'time_shift': RANGE_NOTE_ON + RANGE_NOTE_OFF,
    'velocity': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT
}#add tokens for tablatures here

START_IDX_TAB = {
    'string_on': 0,
    'fret_on': RANGE_STRING,
    'string_off': RANGE_STRING + RANGE_FRET,
    'fret_off': RANGE_STRING  + RANGE_FRET + RANGE_STRING, #not sure if needed for on and off
    'time_shift': RANGE_STRING + RANGE_FRET + RANGE_STRING + RANGE_FRET, #try with absolute, relative, compare
    'velocity': RANGE_STRING + RANGE_FRET + RANGE_STRING + RANGE_FRET + RANGE_TIME_SHIFT
}

TOKEN_END               = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT
TOKEN_START             = TOKEN_END + 1
TOKEN_PAD               = TOKEN_START + 1
VOCAB_SIZE              = TOKEN_PAD + 1
PREPEND_ZEROS_WIDTH     = 4

TOKEN_END_TAB = RANGE_STRING + RANGE_FRET + RANGE_STRING + RANGE_FRET + RANGE_TIME_SHIFT + RANGE_VEL
TOKEN_START_TAB = TOKEN_END_TAB + 1
TOKEN_PAD_TAB = TOKEN_START_TAB + 1
VOCAB_SIZE_TAB = TOKEN_PAD_TAB + 1