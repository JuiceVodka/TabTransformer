"""
To Process Midi Files by preprocessing algorithm suggested by Magenta, 
referenced https://github.com/jason9693/midi-neural-processor/blob/bea0dc612b7f687f964d0f6d54d1dbf117ae1307/processor.py
"""

import pretty_midi
from data.constants import *
#from constants import *

import jams

class SustainAdapter:
    def __init__(self, time, type):
        self.start =  time
        self.type = type


class SustainDownManager:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.managed_notes = []
        self._note_dict = {} # key: pitch, value: note.start

    def add_managed_note(self, note: pretty_midi.Note):
        self.managed_notes.append(note)

    def transposition_notes(self):
        for note in reversed(self.managed_notes):
            try:
                note.end = self._note_dict[note.pitch]
            except KeyError:
                note.end = max(self.end, note.end)
            self._note_dict[note.pitch] = note.start


# Divided note by note_on, note_off
class SplitNote:
    def __init__(self, type, time, value, velocity):
        ## type: note_on, note_off
        self.type = type
        self.time = time
        self.velocity = velocity
        self.value = value

    def __repr__(self):
        return '<[SNote] time: {} type: {}, value: {}, velocity: {}>'\
            .format(self.time, self.type, self.value, self.velocity)

#my class
class SplitNoteTab:
    def __init__(self, type, time, string, fret, velocity):
        self.type = type
        self.time = time
        self.velocity = velocity
        self.string = string
        self.fret = fret
    
    def __repr__(self):
        return '<[SNoteTab] time: {} type: {}, string: {}, fret: {}, velocity: {}>'\
            .format(self.time, self.type, self.string, self.fret, self.velocity)

#DO splitNote for tabs
#check _tokenize -> calls whichever function you want to use for encoding
#also you have to use the right function to transform to events

class Event:
    def __init__(self, event_type, value):
        self.type = event_type
        self.value = value

    def __repr__(self):
        return '<Event type: {}, value: {}>'.format(self.type, self.value)

    def to_int(self):
        #print("MONKEY")
        return START_IDX[self.type] + self.value

    @staticmethod
    def from_int(int_value):
        info = Event._type_check(int_value)
        return Event(info['type'], info['value'])

    @staticmethod
    def _type_check(int_value):
        range_note_on = range(0, RANGE_NOTE_ON)
        range_note_off = range(RANGE_NOTE_ON, RANGE_NOTE_ON+RANGE_NOTE_OFF)
        range_time_shift = range(RANGE_NOTE_ON+RANGE_NOTE_OFF,RANGE_NOTE_ON+RANGE_NOTE_OFF+RANGE_TIME_SHIFT)

        valid_value = int_value

        if int_value in range_note_on:
            return {'type': 'note_on', 'value': valid_value}
        elif int_value in range_note_off:
            valid_value -= RANGE_NOTE_ON
            return {'type': 'note_off', 'value': valid_value}
        elif int_value in range_time_shift:
            valid_value -= (RANGE_NOTE_ON + RANGE_NOTE_OFF)
            return {'type': 'time_shift', 'value': valid_value}
        else:
            valid_value -= (RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT)
            return {'type': 'velocity', 'value': valid_value}

#my class
class EventTab:
    def __init__(self, event_type, value):
        self.type = event_type
        self.value = value

    def __repr__(self):
        return '<EventTab type: {}, value: {}>'.format(self.type, self.value)

    def to_int(self):
        """if (START_IDX_TAB[self.type] + self.value) > 396:
            print(f"---------------HERE--------------, {self.type}, {self.value}")
        elif (START_IDX_TAB[self.type] + self.value) > 300:
            print(f"---------------CLOSE--------------, {self.type}, {self.value}")
        elif (START_IDX_TAB[self.type] + self.value) > 200:
            print(START_IDX_TAB[self.type] + self.value)"""
        return START_IDX_TAB[self.type] + self.value

    @staticmethod
    def from_int(int_value):
        info = Event._type_check(int_value)
        return Event(info['type'], info['value'])

    @staticmethod
    def _type_check(int_value):
        range_string_on = range(START_IDX_TAB['string_on'], START_IDX_TAB['fret_on'])
        range_fret_on = range(START_IDX_TAB['fret_on'], START_IDX_TAB['string_off'])
        range_string_off = range(START_IDX_TAB['string_off'], START_IDX_TAB['fret_off'])
        range_fret_off = range(START_IDX_TAB['fret_off'], START_IDX_TAB['time_shift'])
        range_time_shift = range(START_IDX_TAB['time_shift'], START_IDX_TAB['velocity'])
        range_velocity = range(START_IDX_TAB['velocity'], TOKEN_END_TAB)

        valid_value = int_value

        if int_value in range_string_on:
            valid_value -= START_IDX_TAB['string_on']
            return {'type': 'string_on', 'value': valid_value}
        elif int_value in range_fret_on:
            valid_value -= START_IDX_TAB['fret_on']
            return {'type': 'fret_on', 'value': valid_value}
        elif int_value in range_string_off:
            valid_value -= START_IDX_TAB['string_off']
            return {'type': 'string_off', 'value': valid_value}
        elif int_value in range_fret_off:
            valid_value -= START_IDX_TAB['fret_off']
            return {'type': 'fret_off', 'value': valid_value}
        elif int_value in range_time_shift:
            valid_value -= START_IDX_TAB['time_shift']
            return {'type': 'time_shift', 'value': valid_value}
        elif int_value in range_velocity:
            valid_value -= START_IDX_TAB['velocity']
            return {'type': 'velocity', 'value': valid_value}
        else:
            raise ValueError(f"Invalid token value: {int_value}")


def _divide_note(notes):
    result_array = []
    notes.sort(key=lambda x: x.start)

    for note in notes:
        on = SplitNote('note_on', note.start, note.pitch, note.velocity)
        off = SplitNote('note_off', note.end, note.pitch, None)
        result_array += [on, off]
    return result_array


def _merge_note(snote_sequence):
    note_on_dict = {}
    result_array = []

    for snote in snote_sequence:
        # print(note_on_dict)
        if snote.type == 'note_on':
            note_on_dict[snote.value] = snote
        elif snote.type == 'note_off':
            try:
                on = note_on_dict[snote.value]
                off = snote
                if off.time - on.time == 0:
                    continue
                result = pretty_midi.Note(on.velocity, snote.value, on.time, off.time)
                result_array.append(result)
            except:
                print('info removed pitch: {}'.format(snote.value))
    return result_array


def _snote2events(snote: SplitNote, prev_vel: int):
    result = []
    if snote.velocity is not None:
        modified_velocity = snote.velocity // 4
        if prev_vel != modified_velocity:
            result.append(Event(event_type='velocity', value=modified_velocity))
    result.append(Event(event_type=snote.type, value=snote.value))
    return result

#my function
def _snotetab2events(snote_tab: SplitNoteTab, prev_vel: int):
    events = []

    if snote_tab.velocity is not None:
        modified_velocity = snote_tab.velocity // 4
        if prev_vel != modified_velocity:
            events.append(EventTab(event_type='velocity', value=modified_velocity))
    
    if snote_tab.type == 'note_on':
        events.append(EventTab(event_type='string_on', value=snote_tab.string))
        events.append(EventTab(event_type='fret_on', value=snote_tab.fret))
    elif snote_tab.type == 'note_off':
        events.append(EventTab(event_type='string_off', value=snote_tab.string))
        events.append(EventTab(event_type='fret_off', value=snote_tab.fret))
    return events


def _event_seq2snote_seq(event_sequence):
    timeline = 0
    velocity = 0
    snote_seq = []

    for event in event_sequence:
        if event.type == 'time_shift':
            timeline += ((event.value+1) / 100)
        if event.type == 'velocity':
            velocity = event.value * 4
        else:
            snote = SplitNote(event.type, timeline, event.value, velocity)
            snote_seq.append(snote)
    return snote_seq


def _make_time_sift_events(prev_time, post_time):
    time_interval = int(round((post_time - prev_time) * 100))
    results = []
    while time_interval >= RANGE_TIME_SHIFT:
        results.append(Event(event_type='time_shift', value=RANGE_TIME_SHIFT-1))
        time_interval -= RANGE_TIME_SHIFT
    if time_interval == 0:
        return results
    else:
        return results + [Event(event_type='time_shift', value=time_interval-1)]

def _make_time_sift_events_tab(prev_time, post_time): #check if time shift events are applied correctly
    time_interval = int(round((post_time - prev_time) * 100))
    results = []
    while time_interval >= RANGE_TIME_SHIFT:
        results.append(EventTab(event_type='time_shift', value=RANGE_TIME_SHIFT-1))
        time_interval -= RANGE_TIME_SHIFT
    if time_interval == 0:
        return results
    else:
        return results + [EventTab(event_type='time_shift', value=time_interval-1)]

"""Time [6,000 values] Indicates the absolute time location
within the segment, quantized into 10 ms bins. This
time will apply to all subsequent Note events until the next Time event. Time events must occur in
chronological order. We define the vocabulary with
times up to 60 seconds for flexibility, but because
time resets for each segment, in practice we use only
the first few hundred events of this type."""

#in the article, absolutie time is used, try that -> article report 0.2 increase in F score

def _control_preprocess(ctrl_changes):
    sustains = []

    manager = None
    for ctrl in ctrl_changes:
        if ctrl.value >= 64 and manager is None:
            # sustain down
            manager = SustainDownManager(start=ctrl.time, end=None)
        elif ctrl.value < 64 and manager is not None:
            # sustain up
            manager.end = ctrl.time
            sustains.append(manager)
            manager = None
        elif ctrl.value < 64 and len(sustains) > 0:
            sustains[-1].end = ctrl.time
    return sustains


def _note_preprocess(susteins, notes):
    note_stream = []

    for sustain in susteins:
        for note_idx, note in enumerate(notes):
            if note.start < sustain.start:
                note_stream.append(note)
            elif note.start > sustain.end:
                notes = notes[note_idx:]
                sustain.transposition_notes()
                break
            else:
                sustain.add_managed_note(note)

    for sustain in susteins:
        note_stream += sustain.managed_notes

    if len(susteins) == 0:
        for note_idx, note in enumerate(notes):
            note_stream.append(note)

    note_stream.sort(key= lambda x: x.start)
    return note_stream

#my function
def midi_to_events(file_path):
    dnotes = encode_midi(file_path)

     # Convert SplitNote objects to Event objects and then to integer tokens
    events = []
    prev_time = 0
    prev_vel = 0
    for snote in dnotes:
        events += _make_time_sift_events(prev_time, snote.time)
        events += _snote2events(snote, prev_vel)
        prev_time = snote.time
        if snote.velocity is not None:
            prev_vel = snote.velocity // 4
    #print(events)
    token_sequence = [event.to_int() for event in events]
    return token_sequence

#my function
def tabEvents_to_intEvents(dnotes):
     # Convert SplitNote objects to Event objects and then to integer tokens
    events = []
    prev_time = 0
    prev_vel = 0
    for snote in dnotes:
        events += _make_time_sift_events(prev_time, snote.time)
        events += _snotetab2events(snote, prev_vel)
        prev_time = snote.time
        if snote.velocity is not None:
            prev_vel = snote.velocity // 4
    """types = []
    for event in events:
        if event.type not in types:
            types.append(event.type)
    print(types)"""
    token_sequence = [event.to_int() for event in events]
    return token_sequence

#my function
#then make a new event that has instead of midi value of tone two values; string and fret
def midi_to_string_and_fret(midi_note):
    # Example mapping for standard tuning
    #string 0 is the bottom string (bass)
    string_tuning = [40, 45, 50, 55, 59, 64]  # MIDI notes for EADGBE strings
    min_fret = 25
    min_string = -1
    for string, tuning in enumerate(string_tuning):
        fret = midi_note - tuning
        if 0 <= fret <= 24:  # Ensure the fret is within range
            if fret < min_fret:
                min_fret = fret
                min_string = string
            #return string, fret
    
    #if i want toake it fancy i could deoverlap tones placed on the same string
    if min_fret < 25:
        return min_string, min_fret

    return None, None  # If no valid string/fret is found

def encode_midi(file_path):
    notes = []
    mid = pretty_midi.PrettyMIDI(midi_file=file_path)
    #print(mid)

    for inst in mid.instruments:
        inst_notes = inst.notes
        #print(inst_notes)
        # ctrl.number is the number of sustain control. If you want to know abour the number type of control,
        # see https://www.midi.org/specifications-old/item/table-3-control-change-messages-data-bytes-2
        ctrls = _control_preprocess([ctrl for ctrl in inst.control_changes if ctrl.number == 64])
        #print(ctrls)
        notes += _note_preprocess(ctrls, inst_notes)

    #print(notes)
    dnotes = _divide_note(notes)

    #print(dnotes)
    dnotes.sort(key=lambda x: x.time)
    # print('sorted:')
    # print(dnotes)
    return dnotes

#my function 
def encode_midi_tab(file_path):
    notes = []
    mid = pretty_midi.PrettyMIDI(midi_file=file_path)

    for inst in mid.instruments:
        inst_notes = inst.notes
        ctrls = _control_preprocess([ctrl for ctrl in inst.control_changes if ctrl.number == 64])
        notes += _note_preprocess(ctrls, inst_notes)

    dnotes = _divide_note(notes)
    dnotes.sort(key=lambda x: x.time)

    # Transform SplitNote objects into SplitNoteTab objects
    split_note_tabs = []
    for snote in dnotes:
        string, fret = midi_to_string_and_fret(snote.value)
        if string is not None and fret is not None:
            split_note_tab = SplitNoteTab(
                type=snote.type,
                time=snote.time,
                string=string,
                fret=fret,
                velocity=snote.velocity
            )
            split_note_tabs.append(split_note_tab)

    return split_note_tabs


def parse_jam(jm):
    events = []
    stringsMidiDict = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64} #standard tuning

    for i, stringNotes in enumerate(jm):
        #0 is bottom string
        for note in stringNotes:
            fret = round(note[2] - stringsMidiDict[i])
            time_start = note[0] #try rounding it a bit maybe (or do it in post)
            time_end = note[0] + note[1]
            event_start = SplitNoteTab("note_on", time_start, i, fret, None)
            event_end = SplitNoteTab("note_off", time_end, i, fret, None)
            events.append(event_start)
            events.append(event_end)

    events.sort(key = lambda x: x.time)
    return events

def encode_jam(file_path):
    jm = jams.load(file_path)
    jmNotes = jm.search(namespace="note_midi")
    tabEvents = parse_jam(jmNotes)
    return tabEvents

#zaenkrt samo na guitarset kr je un tadrug 1.3 TB

def decode_midi(idx_array, file_path=None):
    event_sequence = [Event.from_int(idx) for idx in idx_array]
    #print(event_sequence)
    snote_seq = _event_seq2snote_seq(event_sequence)
    note_seq = _merge_note(snote_seq)
    note_seq.sort(key=lambda x:x.start)

    mid = pretty_midi.PrettyMIDI()
    # if want to change instument, see https://www.midi.org/specifications/item/gm-level-1-sound-set
    instument = pretty_midi.Instrument(1, False, "Developed By Yang-Kichang")
    instument.notes = note_seq

    mid.instruments.append(instument)
    if file_path is not None:
        mid.write(file_path)
    return mid


if __name__ == '__main__':
    mdfile = "datasets/tests/test-in.mid"

    encoded = encode_midi(mdfile)
    print(encoded)

    events_encoded = midi_to_events(mdfile)
    #print(events_encoded)

    decoded = decode_midi(events_encoded,file_path='datasets/tests/test-out.mid')

    ins = pretty_midi.PrettyMIDI(mdfile)
    #print(ins)
    #print(ins.instruments[0])
    #for i in ins.instruments:
    #    print(i.control_changes)
    #    print(i.notes)

    # Encode the MIDI file as tokens
    encoded = encode_midi_tab(mdfile)
    #print("Encoded Tokens:", encoded) #TOTO TEST THIS
    tkns = tabEvents_to_intEvents(encoded)
    #print(tkns)

    tabEvents = encode_jam("datasets/guitarset/annotations/00_BN1-129-Eb_comp.jams")
    print(tabEvents)
    