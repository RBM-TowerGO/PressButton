"""Conservative checks of explicit scene claims in a short action explanation.

These checks compare words with the current model assessment. They neither prove
the assessment correct nor map image directions to robot coordinates. Unmatched
language is deliberately left alone rather than guessed at.
"""
import re


_CAMERAS = {
    'head': r'(?:\bhead\b|头(?:部)?(?:相机|图|视角|画面)?)',
    'wrist': r'(?:\bwrist\b|腕(?:部)?(?:相机|图|视角|画面)?)',
}
_CLAUSE = re.compile(r'[.,;，。；\n]|\b(?:but|however|yet)\b|但是|然而|不过|但', re.I)
_MODIFIER_ZH = r'(?:(?:清楚|清晰|明确|完整|直接|可靠|完全|明显|真正)(?:地)?|仍然|仍|还)*'
_NEGATIVE_VISIBILITY = re.compile(
    r'(?:并不|并未|尚未|未|不|无法|没(?:有)?)(?:能|能够)?' + _MODIFIER_ZH +
    r'(?:看到|看见|看清|识别到|辨认出|显示|可见)|'
    r'看不到|看不见|不可见|未见|'
    r'\b(?:cannot|can\s*not|can[’\']t|could\s+not|couldn[’\']t|'
    r'does\s+not|doesn[’\']t|do\s+not|don[’\']t|not|never|no\s+longer)\s+'
    r'(?:(?:clearly|reliably|fully|yet|currently)\s+)*(?:sees?|shows?|visible)\b|'
    r'\bshows?\s+no\s+(?:(?:visible|target|down|call)\s+)*button\b', re.I)
_UNCERTAIN_VISIBILITY = re.compile(
    r'\b(?:may|might|could)\s+(?:be\s+)?(?:clearly\s+)?(?:show|see|visible)\b|'
    r'(?:可能|也许|似乎|不确定是否|无法确认是否)' + _MODIFIER_ZH + r'(?:看到|看见|可见|显示)', re.I)


def _visibility_claims(reason, assessment):
    # Mask the negative predicate, not the whole clause: a later affirmative
    # assertion (even in the same sentence) must still be checked.
    cleaned = _NEGATIVE_VISIBILITY.sub(' UNKNOWN_VIEW ', reason)
    cleaned = _UNCERTAIN_VISIBILITY.sub(' UNKNOWN_VIEW ', cleaned)
    for clause in _CLAUSE.split(cleaned):
        for camera, label in _CAMERAS.items():
            if assessment.get(camera + '_visible') == 'yes':
                continue
            # Do not attribute a following HEAD assertion to WRIST or vice versa.
            other = _CAMERAS['head' if camera == 'wrist' else 'wrist']
            for match in re.finditer(label, clause, re.I):
                tail = clause[match.end():]
                boundary = re.search(other, tail, re.I)
                if boundary:
                    tail = tail[:boundary.start()]
                patterns = (
                    r'\b(?:shows?|sees?|visible|see)\b.{0,50}\bbutton\b',
                    r'\bbutton\b\s+(?:is\s+)?(?:(?:clearly|fully)\s+)?visible\b',
                    r'(?:可见|看到|看见|显示).{0,30}按钮',
                    r'按钮.{0,15}(?:位于|在)两指',
                    r'按钮(?:清晰|清楚)?可见',
                )
                if any(re.search(p, tail, re.I) for p in patterns):
                    raise ValueError('reason_conflicts_with_current_' + camera + '_visibility')


_FINGER_EN = (r'(?:(?:the|open|selected|chosen|right|left|contacting|visible|preferred)\s+){0,6}'
              r'(?:fingertip|finger\s+tip|finger|contact\s+end)')
_FORWARD_EN = re.compile(r'\bbutton\b(?P<bridge>.{1,90}?)(?P<finger>' + _FINGER_EN + r')\b', re.I)
_INVERSE_EN = re.compile(r'\b(?P<finger>' + _FINGER_EN + r')\b(?P<bridge>.{1,90}?)'
                         r'\b(?P<object>(?:(?:the|target|down|call|down-call)\s+)*button|it)\b', re.I)
_FINGER_SUBJECT_EN = re.compile(r'\b' + _FINGER_EN + r'\b', re.I)
_BRIDGE_WORDS = set('is lies sits appears remains still visibly clearly slightly well at to on '
                    'the of and above below left right level with relative'.split())
_FINGER_ZH = r'(?:(?:所选|选定|默认|可见|张开的?|右侧?|左侧?)的?){0,4}(?:手指|指尖)'
_LOCATION_ZH = r'(?:右上方|右下方|左上方|左下方|上方|下方|左侧|右侧|左边|右边|上面|下面)'
_FORWARD_ZH = re.compile(r'按钮(?:仍然|仍)?(?:位于|处于|在)(?P<finger>' + _FINGER_ZH +
                         r')(?:的)?(?P<location>' + _LOCATION_ZH + r')')
_INVERSE_ZH = re.compile(r'(?P<finger>' + _FINGER_ZH + r')(?:仍然|仍)?(?:位于|处于|在)'
                         r'(?:目标)?按钮(?:的)?(?P<location>' + _LOCATION_ZH + r')')


def _selected_finger(phrase, selected):
    side = 'left' if selected == 'head_image_left' else 'right'
    other = 'right' if side == 'left' else 'left'
    if re.search(r'\b' + other + r'\b', phrase, re.I) or ('右' if other == 'right' else '左') in phrase:
        return False
    return True


def _bridge_location(bridge):
    # Restrict to a small declarative relation grammar. Action verbs, future or
    # conditional phrases, negation and uncertainty are not affirmative facts.
    words = re.findall(r'[a-z]+', bridge.lower())
    if not words or any(w not in _BRIDGE_WORDS for w in words):
        return {}
    relation = {}
    for token, field, value in (('left', 'head_horizontal', 'left'), ('right', 'head_horizontal', 'right'),
                               ('above', 'head_vertical', 'above'), ('below', 'head_vertical', 'below')):
        if token in words:
            if field in relation:
                return {}  # Contradictory/ambiguous language is not parsed here.
            relation[field] = value
    # "On the right" alone is an image location, not relative to the finger.
    if 'head_horizontal' in relation and 'of' not in words and 'relative' not in words and 'head_vertical' not in relation:
        return {}
    if words[-1:] == ['and']:
        return {}
    return relation


def _zh_location(location):
    return {field: value for token, field, value in (
        ('左', 'head_horizontal', 'left'), ('右', 'head_horizontal', 'right'),
        ('上', 'head_vertical', 'above'), ('下', 'head_vertical', 'below')) if token in location}


def _check_relation(relation, inverse, assessment):
    opposite = {'left': 'right', 'right': 'left', 'above': 'below', 'below': 'above'}
    for field, value in relation.items():
        claimed = opposite[value] if inverse else value
        current = assessment.get(field, 'unknown')
        if current != 'unknown' and current != claimed:
            raise ValueError('reason_conflicts_with_current_' + field + ':' + current)


def _position_claims(reason, assessment):
    if assessment.get('head_visible') != 'yes' or assessment.get('contact_finger', 'unknown') == 'unknown':
        return
    camera, button_seen = None, False
    for clause in _CLAUSE.split(reason):
        # Camera context can carry across a semicolon, but an explicit different
        # camera wins. Split further at camera names for mixed-view sentences.
        pieces = re.split('(' + '|'.join(_CAMERAS.values()) + ')', clause, flags=re.I)
        for piece in pieces:
            if re.fullmatch(_CAMERAS['head'], piece, re.I):
                camera = 'head'
                continue
            if re.fullmatch(_CAMERAS['wrist'], piece, re.I):
                camera = 'wrist'
                continue
            if camera != 'head':
                continue
            button_seen = button_seen or bool(re.search(r'\bbutton\b|按钮', piece, re.I))
            # In "button right of the finger and below it", the first finger
            # is an object, not the subject of an inverse relation. Do not guess
            # what the trailing pronoun denotes. Keep explicit new subjects,
            # e.g. "button right of the finger and the finger is below button".
            forward_objects = [match.span('finger') for match in _FORWARD_EN.finditer(piece)
                               if _bridge_location(match['bridge'])]
            for pattern, inverse, zh in ((_FORWARD_EN, False, False), (_INVERSE_EN, True, False),
                                         (_FORWARD_ZH, False, True), (_INVERSE_ZH, True, True)):
                if inverse and not zh:
                    # Try each explicit subject independently: a broad invalid
                    # match must not swallow a later, separate finger subject.
                    matches = (pattern.match(piece, subject.start())
                               for subject in _FINGER_SUBJECT_EN.finditer(piece))
                else:
                    matches = pattern.finditer(piece)
                for match in matches:
                    if match is None:
                        continue
                    if inverse and not zh and any(start <= match.start('finger') < end
                                                  for start, end in forward_objects):
                        continue
                    if not _selected_finger(match['finger'], assessment['contact_finger']):
                        continue
                    if not zh and inverse and match['object'].lower() == 'it' and not button_seen:
                        continue
                    # Negated/uncertain Chinese positions and English subject
                    # clauses ("cannot tell whether the button is above...")
                    # should not be interpreted as an asserted current relation.
                    prefix = piece[:match.start()]
                    if re.search(r'\b(?:not|no|cannot|can[’\']t|unclear|unknown|whether|if|might|may)\b|'
                                 r'不|未|无法|不能|不确定|可能|如果|是否', prefix, re.I):
                        continue
                    relation = _zh_location(match['location']) if zh else _bridge_location(match['bridge'])
                    _check_relation(relation, inverse, assessment)


def reason_conflicts(reason, assessment):
    """Reject explicit visibility/relative-position contradictions, not actions.

    A successful check is not proof of visual accuracy or a calibrated motion
    direction. In particular, this function does not impose depth/forward gates.
    """
    if not isinstance(reason, str):
        return  # Canonical proposal parsing validates the reason's type.
    _visibility_claims(reason, assessment)
    _position_claims(reason, assessment)


_MOVING_PART = r'(?:(?:the|selected|right|left|open)\s+)*(?:hand|finger|fingertip|gripper|end[- ]effector)'
_MOVE_VERB = r'\b(?:move|shift|translate)\s+(?:' + _MOVING_PART + r'\s+)?'
_MOTION_INTENTS = (
    (2, 1, r'\b(?:raise|lift)\s+' + _MOVING_PART + r'\b|' + _MOVE_VERB +
     r'(?:(?:right|left)\s+and\s+)?(?:up|upward|upwards)\b|(?:抬高|抬起|提升)(?:右侧?|左侧?)?(?:手部|手指|指尖|夹爪)|向上(?:移动|平移|调整)'),
    (2, -1, r'\blower\s+' + _MOVING_PART + r'\b|' + _MOVE_VERB +
     r'(?:(?:right|left)\s+and\s+)?(?:down|downward|downwards)\b|(?:降低|下移)(?:右侧?|左侧?)?(?:手部|手指|指尖|夹爪)|向下(?:移动|平移|调整)'),
    (1, -1, _MOVE_VERB + r'(?:to\s+)?robot[- ]right\b|向机器人右(?:侧|方)?(?:移动|平移|调整)'),
    (1, 1, _MOVE_VERB + r'(?:to\s+)?robot[- ]left\b|向机器人左(?:侧|方)?(?:移动|平移|调整)'),
    (0, 1, r'\b(?:move|shift|translate|push)\s+(?:' + _MOVING_PART + r'\s+)?forward(?:s)?\b|'
     r'\badvance\s+(?:' + _MOVING_PART + r'\b|forward\b)|(?:向前|往前)(?:移动|平移|推)|(?:前推|前进)(?:手部|夹爪|手指)?'),
    (0, -1, r'\b(?:retract|withdraw|retreat)\b|' + _MOVE_VERB +
     r'backward(?:s)?\b|(?:向后|往后)(?:移动|平移|退)|(?:后撤(?!回)|撤回)(?:手部|夹爪|手指)?'),
)
# Preserve commas within an explicitly coordinated motion list. Only a shared
# physical motion verb supplies the omitted verbs; bare image left/right never
# acquire robot-coordinate semantics.
_MOTION_CLAUSE = re.compile(r'[.;。；\n]|\b(?:but|however|yet)\b|但是|然而|不过|但', re.I)
_LIST_DIRECTION = r'(?:forward(?:s)?|backward(?:s)?|robot[- ](?:right|left)|up(?:wards?)?|down(?:wards?)?|lower)\b'
_COORDINATED_MOVE = re.compile(_MOVE_VERB + r'(?P<directions>' + _LIST_DIRECTION +
                               r'(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+)' +
                               _LIST_DIRECTION + r'){1,})', re.I)


def _motion_claims(clause):
    lists = list(_COORDINATED_MOVE.finditer(clause))
    for axis, sign, pattern in _MOTION_INTENTS:
        for match in re.finditer(pattern, clause, re.I):
            if not any(item.start() <= match.start() < item.end() for item in lists):
                yield match.start(), match.end(), axis, sign
    for item in lists:
        for direction in re.finditer(_LIST_DIRECTION, item['directions'], re.I):
            word = direction.group().lower().replace(' ', '-')
            if word.startswith('robot-'):
                axis, sign = 1, -1 if word == 'robot-right' else 1
            elif word.startswith(('forward', 'backward')):
                axis, sign = 0, 1 if word.startswith('forward') else -1
            else:
                axis, sign = 2, 1 if word.startswith('up') else -1
            # Use the shared verb's scope for *every* list member, so negated,
            # historical and "then" clauses cannot become current directions.
            yield item.start(), item.end(), axis, sign


def motion_reason_conflicts(reason, xyz_mm):
    """Check an explicit physical motion verb against its declared XYZ sign.

    This does not derive a required command from an image relation: e.g. a button
    above a finger is not itself an instruction to raise the hand. Bare image
    left/right likewise is not assigned to a robot Y direction.
    """
    if not isinstance(reason, str) or not isinstance(xyz_mm, (list, tuple)) or len(xyz_mm) != 3:
        return
    prior_motion = False
    for clause in _MOTION_CLAUSE.split(reason):
        for start, end, axis, sign in sorted(_motion_claims(clause)):
            prefix, suffix = clause[:start], clause[end:]
            # Negated or conditional motion and requested image effects are
            # not statements that this command physically takes that direction.
            if re.search(r'\b(?:not|never|cannot|can[’\']t|if|might|may|no\s+need\s+to)\s*(?:\w+\s+){0,2}$|'
                         r'(?:不要|不用|无需|不应|不能|不再|不|未|如果|是否)\s*$', prefix, re.I):
                continue
            if re.search(r'\b(?:before|until|once|previously|earlier|later)\s*(?:\w+\s+){0,2}$|'
                         r'(?:上一轮|上轮|刚才|之前|曾经|已经|已|尚无|缺少|为了|为|之后再|稍后)\s*$', prefix, re.I):
                continue
            if prior_motion and re.search(r'\bthen\s*$|(?:然后|随后|再)\s*$', prefix, re.I):
                continue
            if re.match(r'\s*(?:evidence|record|confirmation)\b|(?:的)?(?:证据|记录|确认)', suffix, re.I):
                continue
            if re.search(r'^\s*(?:in|within)\s+(?:the\s+)?(?:head|wrist|image|pixel|frame|projection)\b|'
                         r'^\s*(?:在)?(?:头图|腕图|图像|画面)(?:中|里)', suffix, re.I):
                continue
            prior_motion = True
            if xyz_mm[axis] * sign <= 0:
                raise ValueError('reason_motion_direction_conflicts_with_' + 'XYZ'[axis] +
                                 (':positive_required' if sign > 0 else ':negative_required'))
