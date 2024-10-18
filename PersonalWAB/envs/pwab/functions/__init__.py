from .search_product_by_query import search_product_by_query
from .get_recommendations_by_history import get_recommendations_by_history
from .add_product_review import add_product_review
from .get_product_details_by_asin import get_product_details_by_asin
from .stop import stop

functions = [
    search_product_by_query,
    get_recommendations_by_history,
    add_product_review,
    get_product_details_by_asin,
    stop
]

assert all(tool.__info__ for tool in functions)

for tool in functions:
    assert tool.__name__ == tool.__info__["function"]["name"], tool
    if "properties" in tool.__info__["function"]["parameters"]:
        assert list(tool.__info__["function"]["parameters"]["properties"].keys()) == list(
            tool.__info__["function"]["parameters"]["required"]
        ), tool
