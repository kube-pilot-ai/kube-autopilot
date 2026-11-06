"""HTTP client for the independently running RAG retrieval service."""
import logging
import os
import re

import httpx


def retrieve_context(query):
    token = os.environ.get('RAG_AUTH_TOKEN')
    if not token:
        raise RuntimeError('Set RAG_AUTH_TOKEN in the chat application .env.')
    symptoms = []
    if re.search(r'\b(imagepullbackoff|errimagepull|imagepulloff)\b', query, re.I):
        symptoms.append('ImagePullBackOff')
    if re.search(r'\boomkilled\b|\bout.of.memory\b', query, re.I):
        symptoms.append('OOMKilled')
    url = os.environ.get('RAG_SERVER_URL', 'http://127.0.0.1:8003').rstrip('/')
    try:
        with httpx.Client(timeout=90) as client:
            response = client.post(url + '/retrieve',
                                   headers={'Authorization': 'Bearer ' + token},
                                   json={'query': query, 'symptoms': symptoms,
                                         'top_k': int(os.environ.get('RAG_TOP_K', '4')),
                                         'min_score': float(os.environ.get('RAG_MIN_SCORE', '0.3'))})
            response.raise_for_status()
            result = response.json()
        matches = result['matches']
        logging.getLogger(__name__).info('RAG selected %s chunks: %s', len(matches),
                                        [match['metadata'].get('section_title') for match in matches])
        print('RAG retrieved chunks:', len(matches))
        for match in matches:
            print(f"  {match['metadata'].get('source')} > {match['metadata'].get('section_title')} | score={match['score']:.3f}")
        return matches
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        raise RuntimeError('Unable to retrieve troubleshooting context. Check the RAG server and its settings.') from exc
