from paperless_ocr import main as main_module


def _set_required_env(monkeypatch):
    monkeypatch.setenv("PAPERLESS_TOKEN", "test_token")
    monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
    monkeypatch.setenv("PRE_TAG_ID", "10")
    monkeypatch.setenv("POST_TAG_ID", "11")
    monkeypatch.setenv("DOCUMENT_WORKERS", "1")


def _stop_after_first_sleep(_):
    raise KeyboardInterrupt


def test_main_exits_on_config_error(mocker, monkeypatch):
    logger = mocker.Mock()
    monkeypatch.setattr(main_module.structlog, "get_logger", mocker.Mock(return_value=logger))
    monkeypatch.setattr(main_module, "Settings", mocker.Mock(side_effect=ValueError("bad")))
    paperless_spy = mocker.Mock()
    monkeypatch.setattr(main_module, "PaperlessClient", paperless_spy)

    main_module.main()

    logger.error.assert_called_once()
    paperless_spy.assert_not_called()


def test_main_filters_post_tag_and_processes_docs(monkeypatch):
    _set_required_env(monkeypatch)
    docs = [
        {"id": 1, "tags": [10]},
        {"id": 2, "tags": [10, 11]},
    ]
    processed = []

    class DummyPaperlessClient:
        instances = []

        def __init__(self, settings):
            self.settings = settings
            self.closed = False
            DummyPaperlessClient.instances.append(self)

        def get_documents_to_process(self):
            return list(docs)

        def close(self):
            self.closed = True

    class DummyOpenAIProvider:
        def __init__(self, settings):
            self.settings = settings

    class DummyProcessor:
        def __init__(self, doc, paperless, ocr_provider, settings):
            self.doc = doc

        def process(self):
            processed.append(self.doc["id"])

    monkeypatch.setattr(main_module, "PaperlessClient", DummyPaperlessClient)
    monkeypatch.setattr(main_module, "OpenAIProvider", DummyOpenAIProvider)
    monkeypatch.setattr(main_module, "DocumentProcessor", DummyProcessor)
    monkeypatch.setattr(main_module, "configure_logging", lambda settings: None)
    monkeypatch.setattr(main_module, "setup_libraries", lambda settings: None)
    monkeypatch.setattr(main_module.time, "sleep", _stop_after_first_sleep)

    main_module.main()

    assert processed == [1]
    assert len(DummyPaperlessClient.instances) == 2
    assert DummyPaperlessClient.instances[1].closed is True


def test_main_continues_after_processing_error(monkeypatch):
    _set_required_env(monkeypatch)
    docs = [
        {"id": 1, "tags": [10]},
        {"id": 2, "tags": [10]},
    ]
    processed = []
    attempted = []

    class DummyPaperlessClient:
        instances = []

        def __init__(self, settings):
            self.settings = settings
            self.closed = False
            DummyPaperlessClient.instances.append(self)

        def get_documents_to_process(self):
            return list(docs)

        def close(self):
            self.closed = True

    class DummyOpenAIProvider:
        def __init__(self, settings):
            self.settings = settings

    class DummyProcessor:
        def __init__(self, doc, paperless, ocr_provider, settings):
            self.doc = doc
            attempted.append(doc["id"])

        def process(self):
            if self.doc["id"] == 1:
                raise RuntimeError("boom")
            processed.append(self.doc["id"])

    monkeypatch.setattr(main_module, "PaperlessClient", DummyPaperlessClient)
    monkeypatch.setattr(main_module, "OpenAIProvider", DummyOpenAIProvider)
    monkeypatch.setattr(main_module, "DocumentProcessor", DummyProcessor)
    monkeypatch.setattr(main_module, "configure_logging", lambda settings: None)
    monkeypatch.setattr(main_module, "setup_libraries", lambda settings: None)
    monkeypatch.setattr(main_module.time, "sleep", _stop_after_first_sleep)

    main_module.main()

    assert attempted == [1, 2]
    assert processed == [2]
    assert len(DummyPaperlessClient.instances) == 3
    assert all(client.closed for client in DummyPaperlessClient.instances[1:])
