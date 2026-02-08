from unittest.mock import MagicMock

from classifier.worker import TaxonomyCache


def test_taxonomy_cache_refresh_and_lookup_existing_items():
    client = MagicMock()
    client.list_correspondents.return_value = [{"id": 1, "name": "Revolut", "document_count": 10}]
    client.list_document_types.return_value = [{"id": 2, "name": "Invoice", "documents_count": 5}]
    client.list_tags.return_value = [{"id": 3, "name": "Bills", "matching_algorithm": "none"}]

    cache = TaxonomyCache(client, taxonomy_limit=100)
    cache.refresh()

    assert cache.correspondent_names() == ["Revolut"]
    assert cache.document_type_names() == ["Invoice"]
    assert cache.tag_names() == ["Bills"]

    # Correspondents allow substring/normalised matching and strip company suffixes.
    assert cache.get_or_create_correspondent_id("Revolut Ltd") == 1

    # Document types are matched by exact normalised name (no substring).
    assert cache.get_or_create_document_type_id("invoice") == 2

    # Tags are matched case-insensitively.
    assert cache.get_or_create_tag_ids(["bills"]) == [3]

    client.create_correspondent.assert_not_called()
    client.create_document_type.assert_not_called()
    client.create_tag.assert_not_called()


def test_taxonomy_cache_creates_missing_items_and_updates_cache():
    client = MagicMock()
    client.list_correspondents.return_value = []
    client.list_document_types.return_value = []
    client.list_tags.return_value = [{"id": 10, "name": "Existing", "matching_algorithm": "none"}]

    client.create_correspondent.return_value = {"id": 1, "name": "ACME"}
    client.create_document_type.return_value = {"id": 2, "name": "Letter"}
    client.create_tag.side_effect = [
        {"id": 11, "name": "NewTag", "matching_algorithm": "none"},
        {"id": 12, "name": "Another", "matching_algorithm": "none"},
    ]

    cache = TaxonomyCache(client, taxonomy_limit=100)
    cache.refresh()

    assert cache.get_or_create_correspondent_id("ACME") == 1
    assert cache.get_or_create_document_type_id("Letter") == 2
    assert cache.get_or_create_tag_ids(["Existing", "NewTag", "Another"]) == [10, 11, 12]

    # New items should now be available without additional create calls.
    assert cache.get_or_create_correspondent_id("ACME") == 1
    assert cache.get_or_create_document_type_id("Letter") == 2
    assert cache.get_or_create_tag_ids(["NewTag"]) == [11]

    assert client.create_correspondent.call_count == 1
    assert client.create_document_type.call_count == 1
    assert client.create_tag.call_count == 2


def test_taxonomy_cache_tag_creation_uses_matching_algorithm_from_existing_tags():
    client = MagicMock()
    client.list_correspondents.return_value = []
    client.list_document_types.return_value = []
    client.list_tags.return_value = [
        {"id": 3, "name": "Bills", "matching_algorithm": 0},
    ]
    client.create_tag.return_value = {"id": 4, "name": "NewTag", "matching_algorithm": 0}

    cache = TaxonomyCache(client, taxonomy_limit=100)
    cache.refresh()

    ids = cache.get_or_create_tag_ids(["Bills", "NewTag"])
    assert ids == [3, 4]

    # When an existing tag uses an int matching_algorithm, we use 0 for new tags.
    client.create_tag.assert_called_once_with("NewTag", matching_algorithm=0)
