import work.slm_defs as df


def test_redemet_url_has_api_key_placeholder():
    assert df.CS_REDEMET_URL.startswith("https://")
    assert "{0}" in df.CS_REDEMET_URL
