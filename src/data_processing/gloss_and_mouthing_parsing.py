import re
from typing import Optional
import logging

import pandas as pd

from src.data_processing.text_processors import remove_speaker_prefix
from src.data_processing.types import GlossAndMouthingClassificationResult


logger = logging.getLogger(__name__)

def missing_chars_assertion(_text: str, _original_text: str) -> None:
    assert len(_text) == 0, \
        f'Characters remaining after extraction from text: {_original_text}, remaining: "{_text}"'


def get_mouthing_metadata(cue_text: str) -> Optional[GlossAndMouthingClassificationResult]:
    """
    Detects and classifies annotations of mouth movements in cue text.

    Returns a dictionary with boolean flags for each category and the extracted value,
    or None if no mouth movement is detected.
    """

    def single_detection_assertion(
            detected_yet: bool,
            _text: str,
            _result: GlossAndMouthingClassificationResult
    ) -> None:
        assert not detected_yet, f'Multiple mouthing categories detected in {_text}. Results: {_result}'

    text: str = remove_speaker_prefix(cue_text)

    # Return early if gloss is certain
    certain_gloss_starting_chars: list[str] = ['$', '|']
    if text[0] in certain_gloss_starting_chars:
        return None

    # Initialize result dictionary
    result: GlossAndMouthingClassificationResult = {
        'is_regular_mouthing': False,
        'is_incomplete_mouthing': False,
        'is_unclear_mouthing': False,
        'is_mouth_gesture': False,
        'is_oral_articulation': False,
        'is_sound_imitation': False,
        'is_undocumented_mouthing': False,
        'mouthing_token': ''
    }

    detected: bool = False

    hyphen_chars: list[str] = ['-', ' ']
    allowed_chars: list[str] = hyphen_chars + ["'"]
    empty_string: str = ''

    anomaly_sentence_list: list[str] = [
        'normal [T05]',
        '[L07] voll'
    ]

    if text in anomaly_sentence_list:
        # Log anomaly
        logger.debug(f'Anomaly mouthing classification detected: "{text}"')
        # Data
        result['mouthing_token'] = text
        result['is_undocumented_mouthing'] = True
        detected = True

    # 0.1) Check for unknown pattern (e.g. "[L02]" or "[L27]" or "[T05] or "[C04]")
    anomaly_sound_imitation_pattern = re.compile(r'^\[[LTCD]\d{2}\]$')
    has_anomaly_sound_imitation: bool = bool(anomaly_sound_imitation_pattern.match(text))
    if has_anomaly_sound_imitation:
        # Sanity check
        single_detection_assertion(detected, text, result)
        # Log anomaly
        logger.debug(f'Undocumented mouthing classification detected: "{text}"')
        # Data
        result['mouthing_token'] = text
        result['is_undocumented_mouthing'] = True
        detected = True
        
    # 1) Check for unclear mouthing (denoted with "??")
    unclear_mouthing_pattern: str = '??'
    if unclear_mouthing_pattern in text:
        # Sanity check
        single_detection_assertion(detected, text, result)
        # Data
        result['mouthing_token'] = text
        result['is_unclear_mouthing'] = True
        detected = True
        # Sanity check
        remaining_text: str = text.replace(unclear_mouthing_pattern, empty_string)
        if '[MG]' in remaining_text:
            logger.debug(f'Unknown condition in mouth gesture ("[MG]") in unclear mouthing ("??") classification detected: "{text}"')
        else:
            missing_chars_assertion(remaining_text, text)

    # 2) Check for mouth gesture (denoted with "[MG]")
    mouth_gesture_pattern: str = '[MG]'
    if mouth_gesture_pattern in text:
        if not text.startswith(mouth_gesture_pattern):
            # Sanity check
            logger.debug(f'Unknown condition in mouth gesture ("[MG]") classification detected: "{text}"')

        # Sanity check
        if not result['is_unclear_mouthing']:
            single_detection_assertion(detected, text, result)
        else:
            logger.warning(f'Double detection of unclear mouthing ("??") and mouth gesture ("[MG]") detected: "{text}"')
        # Data
        result['is_mouth_gesture'] = True
        result['mouthing_token'] = text
        detected = True

    # 3) Check for orally articulated (starts with "#", can also contain curly brackets "{ / }")
    oral_articulation_pattern: str = '#'
    if text.startswith(oral_articulation_pattern):
        # Sanity check
        single_detection_assertion(detected, text, result)
        # Data
        result['is_oral_articulation'] = True
        result['mouthing_token'] = text
        detected = True

    # 4) Check for sound imitation (starts with '[LM' and ends with ']')
    sound_imitation_pattern_start: str = '[LM'
    sound_imitation_pattern_end: str = ']'
    if sound_imitation_pattern_start in text:
        if not (
                text.startswith(sound_imitation_pattern_start)
                and text.endswith(sound_imitation_pattern_end)
        ):
            # Sanity check
            logger.debug(f'Unknown condition in sound imitation mouthing ("[LM:...]") classification detected: "{text}"')

        # Sanity check
        # allow double detection in case of mouth gesture
        if not result['is_mouth_gesture']:
            single_detection_assertion(detected, text, result)
        else:
            logger.warning(f'Double detection of mouth gesture ("[MG]") and sound imitation mouthing ("[LM:...]") detected: "{text}"')

        # Data
        result['is_sound_imitation'] = True
        result['mouthing_token'] = text
        detected = True

    # 5) Check for incomplete mouthing (lowercase with curly braces)
    incomplete_mouthing_pattern_open_bracket: str = '{'
    incomplete_mouthing_pattern_close_bracket: str = '}'
    incomplete_mouthing_pattern_hashtag: str = '#'
    if (
            incomplete_mouthing_pattern_open_bracket in text
            and incomplete_mouthing_pattern_close_bracket in text
            and not text.startswith(incomplete_mouthing_pattern_hashtag)
            and text
            .replace(incomplete_mouthing_pattern_open_bracket, empty_string)
            .replace(incomplete_mouthing_pattern_close_bracket, empty_string)
            .replace(hyphen_chars[0], empty_string)
            .replace(hyphen_chars[1], empty_string)
            .islower()
    ):
        # Sanity check
        single_detection_assertion(detected, text, result)
        # Data
        result['is_incomplete_mouthing'] = True
        result['mouthing_token'] = text
        detected = True

    # 7) Check for regular mouthing (all lowercase letters, possibly with hyphens and spaces)
    mouthed_name_pattern: str = r' #name\d*'
    processed_text: str = re.sub(mouthed_name_pattern, empty_string, text).strip()
    if all([c.islower() or c in allowed_chars for c in processed_text]):
        # Sanity check
        single_detection_assertion(detected, text, result)
        # Data
        result['is_regular_mouthing'] = True
        result['mouthing_token'] = text
        detected = True

    # Sanity check
    if (not detected
            and (
                    unclear_mouthing_pattern in text
                    or mouth_gesture_pattern in text
                    or (oral_articulation_pattern in text and not text.startswith('$ALPHA'))
                    or sound_imitation_pattern_start in text
                    or incomplete_mouthing_pattern_open_bracket in text
                    or incomplete_mouthing_pattern_close_bracket in text
            )
    ):
        raise ValueError(f'Unknown condition detected in mouthing categorization: "{text}"')

    # If nothing detected, return None
    if not detected:
        # in case it is a gloss
        return None

    return result


def get_gloss_metadata(
        cue_text: str,
        _gloss_types: pd.DataFrame
) -> Optional[GlossAndMouthingClassificationResult]:
    """
    Detects and classifies annotations of glosses/signs in cue text.

    Returns a dictionary with boolean flags for each type and the extracted value,
    or None if no gloss/sign is detected.
    """

    def single_detection_assertion(
            detected_yet: bool,
            _text: str,
            result_dict: GlossAndMouthingClassificationResult
    ) -> None:
        assert not detected_yet, f'Multiple gloss categories detected in {_text}. Result: {result_dict}'

    def is_double_gloss(_text: str, _double_sign_pattern: str) -> bool:
        return (_double_sign_pattern in _text
                and not text.startswith(_double_sign_pattern)
                and not text.endswith(_double_sign_pattern)
                )

    def is_only_second_hand_signing(_text: str, _double_sign_pattern: str) -> bool:
        return text.startswith(_double_sign_pattern)

    def merge_result_dicts(
            d1: GlossAndMouthingClassificationResult,
            d2: GlossAndMouthingClassificationResult,
            _double_gloss_pattern: str,
            join_key_for_str_vals: str
    ):
        assert d1 is not None and d2 is not None, 'Cannot merge None result dictionaries in gloss categorization.'

        merged_dict: GlossAndMouthingClassificationResult = {
            k: (
                f'{d1[k]}||{d2[k]}' if k == join_key_for_str_vals  # process string-based properties
                else d1[k] or d2[k]  # process boolean properties
            )
            for k in d1.keys() | d2.keys()
        }

        return merged_dict

    def analyse_single_gloss(
            _text: str,
            _result: GlossAndMouthingClassificationResult,
            _double_gloss_pattern: str,
            testing_double_gloss_part: bool = False
    ) -> Optional[GlossAndMouthingClassificationResult]:

        detected: bool = False

        empty_string: str = ''

        # 1) Check for name sign (starts with "$NAME")
        name_sign_pattern: str = '$NAME'
        if _text.startswith(name_sign_pattern):
            # Data
            _result['is_name_sign'] = True
            _result['lexical_sign'] = _text
            detected = True

        # 2) Check for city names (starts with "$STÄDTENAMEN")
        city_name_sign_pattern: str = '$STÄDTENAMEN'
        if _text.startswith(city_name_sign_pattern):
            # Sanity check
            single_detection_assertion(detected, _text, _result)
            # Data
            _result['is_city_name_sign'] = True
            _result['lexical_sign'] = _text
            detected = True

        # 3) Check for organisation name sign (starts with "$ORG")
        org_name_sign_pattern: str = '$ORG'
        if _text.startswith(org_name_sign_pattern):
            # Sanity check
            single_detection_assertion(detected, _text, _result)
            # Data
            _result['is_organisation_name_sign'] = True
            _result['lexical_sign'] = _text
            detected = True

        # 4) Check for unknown regional sign (contains "$KANDIDAT")
        unknown_regional_pattern: str = '$KANDIDAT'
        full_unknown_regional_pattern: str = f'-{unknown_regional_pattern}'
        if full_unknown_regional_pattern in _text:
            # Sanity check
            single_detection_assertion(detected, _text, _result)
            # Data
            _result['is_unknown_regional_sign'] = True
            _result['lexical_sign'] = _text
            detected = True

        # 5) Check for bound German morpheme (starts with "$WORTTEIL")
        bound_morpheme_pattern: str = '$WORTTEIL'
        if _text.startswith(bound_morpheme_pattern):
            # Sanity check
            single_detection_assertion(detected, _text, _result)
            # Data
            _result['is_bound_german_morpheme'] = True
            _result['lexical_sign'] = _text
            detected = True

        # 6) Check for productive sign (equals "$PROD")
        productive_sign_pattern: str = '$PROD'
        if _text.startswith(productive_sign_pattern):
            # Sanity check
            single_detection_assertion(detected, _text, _result)
            # Data
            _result['is_productive_sign'] = True
            _result['lexical_sign'] = _text
            detected = True

        # 7) Check for pointing sign (starts with "$INDEX")
        pointing_sign_pattern: str = '$INDEX'
        if _text.startswith(pointing_sign_pattern):
            # Sanity check
            single_detection_assertion(detected, _text, _result)
            # Data
            _result['is_pointing_sign'] = True
            _result['lexical_sign'] = _text
            detected = True

        # 8) Check for finger spelling (starts with "$ALPHA")
        finger_spelling_base_pattern: str = '$ALPHA'
        if _text.startswith(finger_spelling_base_pattern):
            # Sanity check
            single_detection_assertion(detected, _text, _result)

            # Data
            _result['lexical_sign'] = _text
            _result['is_finger_spelling'] = True
            detected = True

        # 9) Check for initialisation sign (starts with "$INIT")
        init_sign_base_pattern: str = '$INIT'
        if _text.startswith(init_sign_base_pattern):
            # Sanity check
            single_detection_assertion(detected, _text, _result)

            # Data
            _result['is_initialisation_sign'] = True
            _result['lexical_sign'] = _text
            detected = True

        # 10) Check for number sign (starts with "$NUM")
        number_sign_pattern: str = '$NUM'
        if _text.startswith(number_sign_pattern):
            # Sanity check
            single_detection_assertion(detected, _text, _result)
            # Data
            _result['is_number_sign'] = True
            _result['lexical_sign'] = _text
            detected = True

        # 11) Check for list buoy sign (starts with "$LIST")
        list_buoy_pattern: str = '$LIST'
        if _text.startswith(list_buoy_pattern):
            # Sanity check
            single_detection_assertion(detected, _text, _result)

            # Data
            _result['is_list_buoy_sign'] = True
            _result['lexical_sign'] = _text
            detected = True

        # 12) Check for gesture sign (starts with "$GEST")
        gesture_sign_pattern: str = '$GEST'
        if _text.startswith(gesture_sign_pattern):
            # Sanity check
            single_detection_assertion(detected, _text, _result)
            # Data
            _result['is_gesture_sign'] = True
            _result['lexical_sign'] = _text
            detected = True

        # 13) Check for mouthing without manual activity (starts with "$ORAL")
        mouthing_without_manual_pattern: str = '$ORAL'
        if _text.startswith(mouthing_without_manual_pattern):
            # Sanity check
            single_detection_assertion(detected, _text, _result)
            # Data
            _result['is_mouthing_without_manual_activity'] = True
            _result['lexical_sign'] = _text
            detected = True

        # 14) Check for cued speech (starts with "$PMS")
        cued_speech_pattern: str = '$PMS'
        if _text.startswith(cued_speech_pattern):
            # Sanity check
            single_detection_assertion(detected, _text, _result)
            # Data
            _result['is_cued_speech'] = True
            _result['lexical_sign'] = _text
            detected = True

        # 15) Check for unclear linguistic manual activity (starts with "$UNKLAR")
        unclear_activity_pattern: str = '$UNKLAR'
        if _text.startswith(unclear_activity_pattern):
            # Sanity check
            single_detection_assertion(detected, _text, _result)
            # Data
            _result['is_unclear_linguistic_manual_activity'] = True
            _result['lexical_sign'] = _text
            detected = True

        # 16) Check for extra-linguistic manual activity (is "$$EXTRA-LING-MAN")
        extra_linguistic_manual_activity_pattern: str = '$$EXTRA-LING-MAN'
        if _text.startswith(extra_linguistic_manual_activity_pattern):
            # Sanity check
            single_detection_assertion(detected, _text, _result)
            # Data
            _result['is_extra_linguistic_manual_activity'] = True
            _result['lexical_sign'] = _text
            detected = True

        # 17) Check for date signs/glosses
        date_sign_pattern: str = '$TAG'
        if _text.startswith(date_sign_pattern):
            # Sanity check
            single_detection_assertion(detected, _text, _result)
            # Data
            _result['is_date_sign'] = True
            _result['lexical_sign'] = _text
            detected = True

        # 18) Check for articulation signs (starts with "$ARTIKULATION")
        articulation_sign_pattern: str = '$ARTIKULATION'
        if _text.startswith(articulation_sign_pattern):
            # Sanity check
            single_detection_assertion(detected, _text, _result)
            # Data
            _result['is_articulation_sign'] = True
            _result['lexical_sign'] = _text
            detected = True

        # 19) Check for foreign sign (can start with it or end with it)
        foreign_sign_pattern: list[str] = ['ASL', 'AUSLAN', 'BSL', 'INTS', 'LIS', 'LSM', 'NZSL', 'PJM']
        for pattern in foreign_sign_pattern:
            # Check if text matches pattern exactly (with optional digit, capital letter, ^, and *)
            pattern_with_optional_suffix = re.match(rf'^{pattern}\d*[A-Z]?\^?\*?$', _text)
            # Check if text contains pattern with dash upfront (with optional digit, optional capital letter, ^, and *)
            pattern_with_dash_match = re.search(rf'-{pattern}\d*[A-Z]?\^?\*?', _text)

            # Sanity check
            assert not (pattern_with_optional_suffix and pattern_with_dash_match), \
                f'Pattern "{pattern}" matches both optional suffix and dash match: {_text}'

            if pattern_with_optional_suffix or pattern_with_dash_match:

                # Sanity checks
                if not _result['is_unknown_regional_sign'] == True:
                    if pattern_with_optional_suffix:
                        remaining_text: str = _text.replace(pattern_with_optional_suffix.group(), empty_string)
                        missing_chars_assertion(remaining_text, _text)
                    elif pattern_with_dash_match:
                        remaining_text: str = _text.replace(_text[:pattern_with_dash_match.end()], empty_string)
                        missing_chars_assertion(remaining_text, _text)

                    if not _result['is_finger_spelling'] == True:
                        single_detection_assertion(detected, _text, _result)

                # Data
                _result['is_foreign_sign'] = True
                _result['lexical_sign'] = _text
                detected = True
                # Done
                break

        # 20) Check for regular gloss (all uppercase, possibly with hyphens and numbers)
        if all(
                c.isupper() or c.isdigit() or c in ['-', '_']
                for c in _text.replace('*', '').replace('^', '')
        ) and not _result['is_foreign_sign']:
            # Sanity check
            single_detection_assertion(detected, _text, _result)
            # Data
            _result['lexical_sign'] = _text
            detected = True

        # ADDITIONAL PROPERTIES

        # Check for modified from citation form (contains "*")
        modified_pattern: str = '*'
        if modified_pattern in _text:
            _result['has_modified_from_citation_form'] = True

        # Set if "type" or "subtype
        type_pattern: str = '^'
        if _text.endswith(type_pattern):
            _result['is_type'] = True
        else:
            _result['is_subtype'] = True

        # Sanity check for every condition, that did not use "if _ in text"
        if (not detected
                and (
                        name_sign_pattern in _text
                        or org_name_sign_pattern in _text
                        or unknown_regional_pattern in _text
                        or bound_morpheme_pattern in _text
                        or productive_sign_pattern in _text
                        or pointing_sign_pattern in _text
                        or finger_spelling_base_pattern in _text
                        or init_sign_base_pattern in _text
                        or number_sign_pattern in _text
                        or list_buoy_pattern in _text
                        or gesture_sign_pattern in _text
                        or mouthing_without_manual_pattern in _text
                        or cued_speech_pattern in _text
                        or unclear_activity_pattern in _text
                        or extra_linguistic_manual_activity_pattern in _text
                        or date_sign_pattern in _text
                )
        ):
            raise ValueError(f'Unknown condition detected in gloss categorization: {_text} - result: {_result}')

        if not detected:
            # in case it is "mouthing"
            return None

        return _result

    text: str = remove_speaker_prefix(cue_text)

    # Initialize result dictionary
    result: GlossAndMouthingClassificationResult = {
        'is_name_sign': False,
        'is_city_name_sign': False,
        'is_organisation_name_sign': False,
        'is_unknown_regional_sign': False,
        'is_bound_german_morpheme': False,
        'is_foreign_sign': False,
        'is_productive_sign': False,
        'is_pointing_sign': False,
        'is_finger_spelling': False,
        'is_initialisation_sign': False,
        'is_number_sign': False,
        'is_list_buoy_sign': False,
        'is_gesture_sign': False,
        'is_mouthing_without_manual_activity': False,
        'is_cued_speech': False,
        'is_unclear_linguistic_manual_activity': False,
        'is_extra_linguistic_manual_activity': False,
        'is_articulation_sign': False,
        'is_date_sign': False,
        'is_double_signing': False,
        'is_second_hand_only': False,
        'has_modified_from_citation_form': False,
        'is_type': False,
        'is_subtype': False,
        'lexical_sign': ''
    }

    # Check for double signing
    double_sign_pattern: str = '||'
    if is_double_gloss(text, double_sign_pattern):
        glosses: list[str] = text.split(double_sign_pattern)
        assert len(glosses) == 2, f'More than two glosses detected: {text}'

        result_one: GlossAndMouthingClassificationResult = analyse_single_gloss(
            glosses[0],
            result.copy(),
            double_sign_pattern,
            testing_double_gloss_part=True
        )

        result_two: GlossAndMouthingClassificationResult = analyse_single_gloss(
            glosses[1],
            result.copy(),
            double_sign_pattern,
            testing_double_gloss_part=True
        )

        assert result_one is not None or result_two is not None, f'Gloss analysis failed for gloss: {text}'

        # Merge results from both glosses
        merged_result: GlossAndMouthingClassificationResult = merge_result_dicts(
            result_one,
            result_two,
            double_sign_pattern,
            join_key_for_str_vals='lexical_sign'
        )

        # Mark double signing
        merged_result['is_double_signing'] = True

        # Set return value
        result = merged_result
    # Check it is only signed with second hand
    elif is_only_second_hand_signing(text, double_sign_pattern):
        stripped_text: str = text.replace(double_sign_pattern, '')
        result = analyse_single_gloss(
            stripped_text,
            result,
            double_sign_pattern
        )
        # Mark second hand only signing
        result['is_second_hand_only'] = True
        # Add full glossing notation instead of stripped version
        result['lexical_sign'] = text
    # Regular gloss
    else:
        result = analyse_single_gloss(
            text,
            result,
            double_sign_pattern
        )

    # Return result
    return result
