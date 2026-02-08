from ocr import daemon as main_module


def _set_required_env(monkeypatch):
    monkeypatch.setenv("PAPERLESS_TOKEN", "test_token")
    monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
    monkeypatch.setenv("PRE_TAG_ID", "10")
    monkeypatch.setenv("POST_TAG_ID", "11")
    monkeypatch.setenv("DOCUMENT_WORKERS", "1")


def test_main_exits_on_config_error(mocker, monkeypatch):
    logger = mocker.Mock()
    monkeypatch.setattr(
        main_module.structlog, "get_logger", mocker.Mock(return_value=logger)
    )
    monkeypatch.setattr(main_module, "Settings", mocker.Mock(side_effect=ValueError("bad")))
    paperless_spy = mocker.Mock()
    monkeypatch.setattr(main_module, "PaperlessClient", paperless_spy)

    main_module.main()

    logger.error.assert_called_once()
    paperless_spy.assert_not_called()


def test_main_skips_docs_with_post_tag_and_cleans_stale_pre_tag(monkeypatch):
    _set_required_env(monkeypatch)

    docs = [
        {"id": 1, "tags": [10]},  # should be processed
        {"id": 2, "tags": [10, 11, 99]},  # should be skipped + pre-tag removed
    ]
    processed = []

    class DummyPaperlessClient:
        instances = []

        def __init__(self, settings):
            self.settings = settings
            self.closed = False
            self.updated_tags = {}
            DummyPaperlessClient.instances.append(self)

        def get_documents_by_tag(self, tag_id):
            assert tag_id == 10
            return list(docs)

        def update_document_metadata(self, doc_id, *, tags=None, **kwargs):
            assert kwargs == {}
            self.updated_tags[doc_id] = set(tags or [])

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

    def run_once(**kwargs):
        items = kwargs["fetch_work"]()
        for item in items:
            kwargs["process_item"](item)

    monkeypatch.setattr(main_module, "PaperlessClient", DummyPaperlessClient)
    monkeypatch.setattr(main_module, "OpenAIProvider", DummyOpenAIProvider)
    monkeypatch.setattr(main_module, "DocumentProcessor", DummyProcessor)
    monkeypatch.setattr(main_module, "configure_logging", lambda settings: None)
    monkeypatch.setattr(main_module, "setup_libraries", lambda settings: None)
    monkeypatch.setattr(main_module, "run_polling_threadpool", run_once)

    main_module.main()

    assert processed == [1]
    # instance[0] is the list client (used for tag hygiene), instance[1] is the per-doc client
    assert len(DummyPaperlessClient.instances) == 2
    assert DummyPaperlessClient.instances[0].updated_tags[2] == {11, 99}
    assert all(client.closed for client in DummyPaperlessClient.instances)


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

        def get_documents_by_tag(self, tag_id):
            assert tag_id == 10
            return list(docs)

        def update_document_metadata(self, doc_id, *, tags=None, **kwargs):
            raise AssertionError("not expected")

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

    def run_once_with_exception_handling(**kwargs):
        items = kwargs["fetch_work"]()
        for item in items:
            try:
                kwargs["process_item"](item)
            except Exception:
                # run_polling_threadpool logs and continues; mimic that.
                pass

    monkeypatch.setattr(main_module, "PaperlessClient", DummyPaperlessClient)
    monkeypatch.setattr(main_module, "OpenAIProvider", DummyOpenAIProvider)
    monkeypatch.setattr(main_module, "DocumentProcessor", DummyProcessor)
    monkeypatch.setattr(main_module, "configure_logging", lambda settings: None)
    monkeypatch.setattr(main_module, "setup_libraries", lambda settings: None)
    monkeypatch.setattr(main_module, "run_polling_threadpool", run_once_with_exception_handling)

    main_module.main()

    assert attempted == [1, 2]
    assert processed == [2]
    assert len(DummyPaperlessClient.instances) == 3  # list client + 2 per-doc clients
    assert all(client.closed for client in DummyPaperlessClient.instances)
