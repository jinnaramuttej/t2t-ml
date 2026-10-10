from unittest.mock import patch
import requests
from server import run_vlm_check

img1 = b'dummy1'
img2 = b'dummy2'

def test_failure(name, mock_response_or_exception, expected):
    if isinstance(mock_response_or_exception, Exception):
        with patch('requests.post', side_effect=mock_response_or_exception):
            res = run_vlm_check(None, img1, img2)
    else:
        with patch('requests.post', return_value=mock_response_or_exception):
            res = run_vlm_check(None, img1, img2)
    print(f'[{name}] -> vlm_verdict: {res.get("vlm_verdict")}')

class MockResp:
    def __init__(self, json_data, status=200):
        self.json_data = json_data
        self.status = status
    def json(self): return self.json_data
    def raise_for_status(self):
        if self.status != 200:
            raise requests.exceptions.HTTPError(f'{self.status} Error')

test_failure('Timeout', requests.exceptions.Timeout('Timeout'), 'ERROR')
test_failure('HTTP 429', MockResp(None, 429), 'ERROR')
test_failure('Missing verdict field', MockResp({'choices':[{'message':{'content':'{"reason_code":"XYZ"}'}}]}), 'UNCERTAIN')
test_failure('Unknown verdict string', MockResp({'choices':[{'message':{'content':'{"verdict":"MAYBE"}'}}]}), 'MAYBE')
test_failure('Malformed JSON', MockResp({'choices':[{'message':{'content':'not json'}}]}, 200), 'UNCERTAIN')
