"""
Retrieves the public DGS Korpus Release 3 data from its webpage.
Downloads videos, SRT transcripts, OpenPose data and persists metadata,
saving each conversation in a separate folder.
"""

from typing import Optional, Union
from pathlib import Path
import requests
import bs4
from urllib.parse import urljoin
import gzip
import shutil
import time
import logging

from src.data_processing.processing_io_utils import make_paths
from src.data_processing.types import RetrievalConfig, ConversationMeta, ConversationPaths
from src.retrieval.retrieval_utils import setup_session, fetch_html

logger = logging.getLogger(__name__)


def parse_conversation_meta(html: str, base_url: str, max_conversations: int) -> list[ConversationMeta]:
    """
    Collect the conversation meta-data from the given HTML table.
    """
    soup = bs4.BeautifulSoup(html, 'html.parser')

    table: Optional[bs4.Tag] = soup.find('table')
    headers: dict[str, int] = {th.text.strip(): i for i, th in enumerate(table.find_all('th'))}
    # each row is a conversation
    rows: list[bs4.Tag] = table.find_all('tr')[1:]  # leave out first row (header)

    limit_conversations: bool = max_conversations != -1

    if limit_conversations:
        logger.info(f'Limiting conversations to retrieve to {max_conversations}.')
        rows = rows[:max_conversations]

    # iterate over rows (conversations)
    conversations_meta_to_process: list[ConversationMeta] = []
    for row in rows:
        data_cells: bs4.ResultSet[bs4.Tag] = row.find_all('td')

        def get_id_for(_data_cells: bs4.ResultSet[bs4.Tag]) -> str:
            return _data_cells[headers['Transkript']]['title']

        def get_url_for(col: str, _data_cells: bs4.ResultSet[bs4.Tag]) -> Optional[str]:
            a_tag: Optional[bs4.Tag] = data_cells[headers[col]].find('a')
            url: Optional[str] = urljoin(base_url, a_tag['href']) if a_tag and 'href' in a_tag.attrs else None
            return url

        def get_topics(_data_cells: bs4.ResultSet[bs4.Tag]) -> list[str]:
            a_tags: list[bs4.Tag] = data_cells[headers['Themen']].find_all('a')
            _topics: list[str] = [a.text.strip() for a in a_tags]
            return _topics

        def get_format(_data_cells: bs4.ResultSet[bs4.Tag]) -> str:
            a_tag: Optional[bs4.Tag] = data_cells[headers['Format']].find('a')
            _format: str = a_tag.text.strip()
            return _format

        def get_age_group(_data_cells: bs4.ResultSet[bs4.Tag]) -> str:
            age_group: str = data_cells[headers['Alters-gruppe']].text.strip()
            return age_group

        # Get the meta-data for each conversation
        conversation_id: str = get_id_for(data_cells)
        srt_url: Optional[str] = get_url_for('SRT', data_cells)
        video_a_url: Optional[str] = get_url_for('Video A', data_cells)
        video_b_url: Optional[str] = get_url_for('Video B', data_cells)
        openpose_url: Optional[str] = get_url_for('OpenPose', data_cells)
        video_ab_url: Optional[str] = get_url_for('Video AB', data_cells)
        video_long_shot_url: Optional[str] = get_url_for('Video Totale', data_cells)
        ilex_url: Optional[str] = get_url_for('iLex', data_cells)
        conversation_topics: list[str] = get_topics(data_cells)
        conversation_format: str = get_format(data_cells)
        conversation_age_group: str = get_age_group(data_cells)

        # Convert into a dedicated data structure for robustness reasons
        conversations_meta_to_process.append(
            ConversationMeta(
                conversation_id=conversation_id,
                srt_url=srt_url,
                video_a_url=video_a_url,
                video_b_url=video_b_url,
                openpose_url=openpose_url,
                video_ab_url=video_ab_url,
                video_long_shot_url=video_long_shot_url,
                ilex_url=ilex_url,
                topics=conversation_topics,
                format=conversation_format,
                age_group=conversation_age_group
            )
        )

    return conversations_meta_to_process

def process_conversation(
        session: requests.Session,
        cfg: RetrievalConfig,
        conversation_meta: ConversationMeta
) -> dict:

    def download(_session: requests.Session, url: str, dest: Path, timeout: float) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with _session.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(1024 * 64):
                    if chunk:
                        f.write(chunk)

    def decompress_gz(src: Path, dest: Path) -> None:
        with gzip.open(src, 'rb') as f_in, open(dest, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        src.unlink(missing_ok=True)

    def handle_download(
            _session: requests.Session,
            url: Optional[str],
            dest: Path,
            timeout: float,
            test_dest: Path = None
    ) -> bool:

        def skip_download(_dest: Path) -> bool:
            return _dest.exists() and _dest.stat().st_size > 0

        if url:
            # Check if file to download already exists
            destination_to_test: Path = test_dest if test_dest else dest
            do_skip: bool = skip_download(destination_to_test)

            if not do_skip:
                download(_session, url, dest, timeout)
                downloaded: bool = True
            else:
                logger.info(f'Skipping download of {url}. File {str(destination_to_test)} already exists.')
                downloaded: bool = False
        else:
            logger.info(f'No url for {str(dest)}.')
            downloaded: bool = False

        return downloaded

    def persist_metadata(metadata: Union[str, list[str]], file_path: Path) -> None:
        with open(file_path, 'w') as f:
            if isinstance(metadata, list):
                f.write('\n'.join(metadata))
            else:
                f.write(metadata)


    # Create conversation folder
    conversation_dir: Path = cfg.data_dir / f'conversation_{conversation_meta.conversation_id}'
    paths: ConversationPaths = make_paths(conversation_dir)
    paths.conversation_dir.mkdir(parents=True, exist_ok=True)

    # Download transcript
    downloaded_srt: bool = handle_download(session, conversation_meta.srt_url, paths.srt, cfg.request_timeout)

    # Download videos
    downloaded_video_a: bool = handle_download(session, conversation_meta.video_a_url, paths.video_a, cfg.request_timeout)
    downloaded_video_b: bool = handle_download(session, conversation_meta.video_b_url, paths.video_b, cfg.request_timeout)

    # Download OpenPose data
    openpose_archive_downloaded: bool = (
        handle_download(
            session,
            conversation_meta.openpose_url,
            paths.openpose_gz,
            cfg.request_timeout,
            test_dest=paths.openpose_json
        )
    )

    if openpose_archive_downloaded:
        decompress_gz(paths.openpose_gz, paths.openpose_json)

    # Download optional videos
    downloaded_video_ab: bool = (
        handle_download(session, conversation_meta.video_ab_url, paths.video_ab, cfg.request_timeout))\
        if cfg.retrieve_video_ab else False
    downloaded_video_long_shot: bool = (
        handle_download(session, conversation_meta.video_long_shot_url, paths.video_long_shot, cfg.request_timeout))\
        if cfg.retrieve_video_long_shot else False

    # Optionally download iLex data
    downloaded_ilex: bool = (
        handle_download(session, conversation_meta.ilex_url, paths.ilex, cfg.request_timeout))\
    if cfg.retrieve_ilex else False

    # Persist other metadata
    persist_metadata(conversation_meta.topics, paths.conversation_topics)
    persist_metadata(conversation_meta.format, paths.conversation_format)
    persist_metadata(conversation_meta.age_group, paths.conversation_age_group)

    # Do retrieval kindly / Delay retrieval
    has_downloaded: bool = (downloaded_srt
                            or downloaded_video_a
                            or downloaded_video_b
                            or openpose_archive_downloaded
                            or downloaded_video_ab
                            or downloaded_video_long_shot
                            or downloaded_ilex)
    if has_downloaded:
        time.sleep(cfg.delay_seconds)

    return {
        'index': conversation_meta.conversation_id,
        'ok': True,
        #'conversation meta': conversation_meta
    }


def retrieve(cfg: RetrievalConfig) -> dict:

    logger.info('Configuration:')
    logger.info(cfg)

    session = setup_session(cfg.user_agent)

    try:
        html = fetch_html(session, cfg.url, cfg.request_timeout)

        conversation_meta_list: list[ConversationMeta] = (
            parse_conversation_meta(
                html,
                cfg.base_url,
                cfg.max_conversations
            )
        )

        # Count conversations for logging
        fetched_conversations: int = len(conversation_meta_list)

        # Filter out conversations without SRT transcripts, if required.
        if cfg.SRT_required:
            logger.info('Filtering out conversations without SRT transcripts.')
            conversation_meta_list = list(filter(lambda p: p.srt_url, conversation_meta_list))

        # Count final conversations to process for logging
        conversations_to_process: int = len(conversation_meta_list)

        logger.info(f'{conversations_to_process}/{fetched_conversations} conversations to process.')

        # Retrieve conversations
        results = []
        for i, conversation_meta in enumerate(conversation_meta_list, 1):
            try:
                processed_conversation_results: dict = process_conversation(session, cfg, conversation_meta)
                results.append(processed_conversation_results)
                logger.info(f'{i}/{conversations_to_process} conversations processed.')
            except Exception as e:
                results.append({'index': conversation_meta.conversation_id, 'ok': False, 'error': str(e)})
                logger.exception(f'Failed conversation {conversation_meta.conversation_id}: {conversation_meta}', e)

        return {
            'total': len(conversation_meta_list),
            'ok': f'{sum(r['ok'] for r in results)}/{len(results)}',
            'results': results
        }

    except requests.RequestException as e:
        logger.exception(f'HTTP/network error while retrieving data: {e}')
        return {'total': 0, 'ok': 0, 'results': [], 'error': str(e)}

    except Exception as e:
        logger.exception(f'Unexpected error while retrieving data:{e}')
        return {'total': 0, 'ok': 0, 'results': [], 'error': str(e)}

    finally:
        session.close()
