"""
Retrieves unique glosses, so called "types", from the DGS-Korpus Release 3 website and saves them as a UTF-8 CSV file.
Provides a ready-to-use list of DGS gloss/type conversations.
"""
from pathlib import Path
from typing import Union
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging

from src.data_processing.types import RetrieveTypesConfig
from src.retrieval.retrieval_utils import setup_session, fetch_html
from src.data_processing.types import RetrievedTypes, TypeCategory


logger = logging.getLogger(__name__)


def parse_types_table(html: str) -> pd.DataFrame:

    def parse_types_text(text: str) -> RetrievedTypes:

        def get_type_category(_lexical_sign: str, _parent_type: str) -> TypeCategory:

            # Token counting (i.e.: "(320 Tokens)") should have been removed by now
            if(
                    '(' in _lexical_sign
                    or
                    ')' in _lexical_sign
            ): raise ValueError(f'Parsing error - sign contains parentheses "(" or ")": {_lexical_sign}')

            if (
                    _parent_type is not None
                    and
                    (
                            '(' in _parent_type
                            or
                            ')' in _parent_type
                    )
            ): raise ValueError(f'Parsing error - sign contains parentheses "(" or ")": {_parent_type}')

            if _parent_type is None:
                if _lexical_sign == '$PROD':
                    # CASE: lexical sign is "productive sign"
                    _type_category: TypeCategory = 'productive sign'
                elif _lexical_sign.endswith('^'):
                    # CASE: lexical sign is a "type"
                    _type_category: TypeCategory = 'type'
                else:
                    raise ValueError(f'Parsing error - lexical sign should be type when parenting type is not available: {_lexical_sign}, {parent_type}')
            else:
                # CASE: lexical sign is "subtype"
                if not _lexical_sign.endswith('^') and parent_type.endswith('^'):
                    _type_category: TypeCategory = 'subtype'
                else:
                    raise ValueError(f'Parsing error - lexical sign should be subtype when parenting type is available: {_lexical_sign}, {_parent_type}')

            return _type_category


        # should return (lexical sign, type, subtype, (parent) type
        types: list[str] = text.split(' → ')
        if len(types) > 2: raise ValueError(f'Parsing error: {text}')

        # get lexical sign
        lexical_sign: str = types[0].split(' ')[0]

        # get parent type
        has_two_data_points: bool = len(types) == 2
        parent_type: Union[str, None] = types[1] if has_two_data_points else None

        # identify type category
        type_category: TypeCategory = get_type_category(lexical_sign, parent_type)

        # return retrieved data
        retrieved_types_data: RetrievedTypes = (lexical_sign, type_category, parent_type)
        return retrieved_types_data

    # Parse the HTML content of the webpage with UTF-8 encoding
    soup = BeautifulSoup(html, 'html.parser')

    # Find the div with class 'textWrapper'
    div = soup.find('div', class_='textWrapper')
    if div is None:
        raise ValueError('Could not find "div.textWrapper" on the page.')

    # Extract all <p> elements within the div. These hold the types/glosses.
    all_p_tags = div.find_all('p')

    # Extract the actual data
    unique_types: set[RetrievedTypes] = {parse_types_text(p.get_text())
                                         for p in all_p_tags}

    # Sort by lexical sign for consistency
    unique_types: list[RetrievedTypes] = sorted(unique_types, key=lambda sample: sample[0])

    # Pack data into Pandas DataFrame
    df = pd.DataFrame(unique_types, columns=['lexical_sign', 'type_category', 'parent_type'])

    return df


def retrieve(cfg: RetrieveTypesConfig):

    logger.info('Configuration:')
    logger.info(cfg)

    # Setup a Requests session
    session = setup_session(cfg.user_agent)

    try:
        # Fetch the HTML content of the index page
        html: str = fetch_html(session, cfg.url, cfg.request_timeout)

        # Parse the HTML table and create a DataFrame
        types_df: pd.DataFrame = parse_types_table(html)

        # Save the DataFrame to a CSV file with UTF-8 encoding
        output_path: Path = cfg.data_dir / cfg.output_file
        types_df.to_csv(output_path, index=False, encoding='utf-8')

        return {
            'ok': True,
            'output_path': str(output_path),
            'num_unique_glosses': len(types_df)
        }

    except requests.RequestException as e:
        logger.exception(f'HTTP/network error while retrieving types: {e}')
        return {'ok': False, 'error': str(e)}

    except ValueError as e:
        logger.exception(f'Parsing error while retrieving types: {e}')
        return {'ok': False, 'error': str(e)}

    except Exception as e:
        logger.exception(f'Unexpected error while retrieving types:{e}')
        return {'ok': False, 'error': str(e)}

    finally:
        session.close()
