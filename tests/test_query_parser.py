from mocap.query.parser import QueryRequest


def test_valid_query_request() -> None:
    request = QueryRequest(
        query_id="Q01",
        sql="SELECT * FROM customer",
        budget=100.0,
    )

    assert request.query_id == "Q01"
    assert request.budget == 100.0


def test_invalid_budget() -> None:
    try:
        QueryRequest(
            query_id="Q01",
            sql="SELECT * FROM customer",
            budget=0,
        )
    except ValueError:
        return

    raise AssertionError("Expected ValueError for invalid budget.")