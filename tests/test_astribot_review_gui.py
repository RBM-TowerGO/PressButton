import copy
import json
import os
import time
import tkinter as tk

import pytest

from astribot_data_review import write_json, load_review, export, get_entry, sha256
from astribot_review_gui import ReviewStore, ReviewWindow
from test_astribot_data_review import review_fixture


def make_review(tmp_path):
    review, source = review_fixture(tmp_path)
    second = copy.deepcopy(review['episodes'][0]); second['original_id'] = 12
    review['episodes'].append(second)
    path = tmp_path/'review.json'; write_json(path, review)
    return path, source


def test_delete_restore_undo_and_export_order(tmp_path):
    path, source = make_review(tmp_path); digest = sha256(source)
    store = ReviewStore(path)
    store.set_status(9, 'keep', 0, 5, 'good')
    store.set_status(9, 'reject', 0, 5, 'deleted')
    assert get_entry(load_review(path), 9)['status'] == 'reject'
    store.set_status(9, 'restore', 0, 5, '')
    assert get_entry(store.data, 9)['status'] == 'keep'
    store.set_status(12, 'keep', 1, 4, 'crop')
    store.move(12, -1)
    assert [e['original_id'] for e in load_review(path)['episodes']] == [12,9]
    export(store.data, tmp_path/'export')
    mapping = json.loads((tmp_path/'export/episode_mapping.json').read_text())
    assert [e['original_id'] for e in mapping] == [12,9]
    assert [e['new_id'] for e in mapping] == [0,1]
    store.undo()
    assert [e['original_id'] for e in load_review(path)['episodes']] == [9,12]
    assert sha256(source) == digest


def test_external_edit_and_bad_range_do_not_overwrite(tmp_path):
    path, _ = make_review(tmp_path)
    store = ReviewStore(path); digest = sha256(path)
    with pytest.raises(ValueError): store.set_status(9, 'keep', 4, 5, '')
    assert sha256(path) == digest
    data = load_review(path); data['external_change'] = True; write_json(path, data)
    with pytest.raises(ValueError, match='其他程序'): store.move(12, -1)
    assert load_review(path)['external_change']


def test_bulk_keep_only_pending_preserves_fields_and_is_undoable(tmp_path):
    path, source = make_review(tmp_path)
    data = load_review(path)
    for i, status in enumerate(['reject', 'hold', 'keep', 'pending', 'pending'], start=20):
        e = copy.deepcopy(data['episodes'][0])
        e.update(original_id=i, status=status)
        if i == 23:
            e['audit']['errors'] = ['bad image bytes']
        if i == 24:
            e['end_frame'] = 1
        data['episodes'].append(e)
    data['episodes'][0].update(start_frame=1, end_frame=4, note='keep this note')
    write_json(path, data)
    store = ReviewStore(path)
    kept, skipped = store.keep_all_pending()
    assert kept == [9, 12] and skipped == [23, 24]
    saved = load_review(path)
    expected = copy.deepcopy(data)
    for e in expected['episodes'][:2]: e['status'] = 'keep'
    assert saved == expected
    assert store.keep_all_pending() == ([], [23, 24])
    store.undo()
    assert load_review(path) == data


@pytest.mark.skipif(not os.environ.get('DISPLAY'), reason='requires desktop display')
def test_gui_navigation_crop_delete_restore_and_playback(tmp_path):
    path, source = make_review(tmp_path)
    root = tk.Tk(); app = ReviewWindow(root, path)
    errors = []; app.report_error = errors.append
    def pump(predicate):
        end = time.monotonic()+5
        while time.monotonic()<end:
            root.update()
            if predicate(): return
            time.sleep(.01)
        raise AssertionError('GUI did not settle')
    try:
        pump(lambda: app.data is not None)
        assert app.current_id == 9
        assert app.autoplay_next.get() and not app.playing
        app.start.set(1); app.end.set(3); app.note.set('inclusive crop')
        app.adjacent(1)
        pump(lambda: app.current_id == 12 and app.data is not None)
        assert app.playing
        assert get_entry(load_review(path),9)['end_frame'] == 4
        app.adjacent(-1)
        pump(lambda: app.current_id == 9 and app.data is not None)
        assert not app.playing
        assert app.start.get() == 1 and app.end.get() == 3
        app.status_clicked('keep'); app.status_clicked('reject'); app.status_clicked('restore')
        assert get_entry(load_review(path),9)['status'] == 'keep'
        app.toggle_play(); pump(lambda: not app.playing)
        assert app.frame == 4
        app.autoplay_next.set(False)
        app.adjacent(1)
        pump(lambda: app.current_id == 12 and app.data is not None)
        assert not app.playing and app.frame == 0
        app.filter.set('已删除'); app.refresh_list()
        app.keep_all_pending_clicked()
        assert get_entry(load_review(path),12)['status'] == 'keep'
        assert not errors
    finally:
        app.pause(); root.destroy()
