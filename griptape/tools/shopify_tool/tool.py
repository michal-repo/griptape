from __future__ import annotations
import requests
from griptape.tools import BaseTool
from griptape.utils.decorators import activity
from griptape.artifacts import TextArtifact
from schema import Schema, Literal
from attr import field, define


@define
class ShopifyGQLTool(BaseTool):
    """
    A tool for fetching information from the Shopify Admin API using GraphQL.
    Example product ID: 748720914497
    Example customer ID: 624072654913
    Example product: Adictivo Doble Reposado
    Example order name: 135291
    Example order id: 5388378439869
    Example confirmation number: 6MRAR7DEA
    """

    shop_name: str = field(default=None)
    api_key: str = field(default=None)
    api_token: str = field(default=None)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

    def _send_graphql_query_request(self, query: str, variables: dict) -> dict:
        headers = {"Content-Type": "application/json"}

        auth = (self.api_key, self.api_token)

        # print(f"QUERY:{query}")
        # print(f"VARIABLES: {variables}")

        response = requests.post(
            variables["shop_url"], json={"query": query, "variables": variables}, headers=headers, auth=auth
        )

        response_data = response.json()

        # print(f"RAW RESPONSE: {response_data}")

        if "errors" in response_data:
            raise Exception(response_data["errors"])

        return response_data

    @activity(
        config={
            "description": "Can be used to get a product using a field.",
            "schema": Schema(
                {
                    Literal(
                        "shopify_params",
                        description="""
                        A dictionary to pass into the method, the only key needed is product_field and its value (str).
                        For example: {'product_field': 'Compoveda Rosa'} or {'product_field': '748720914000'}
                        """,
                    ): dict
                }
            ),
        }
    )
    def get_product_by_field(self, shopify_params: dict) -> TextArtifact:
        shop_url = (
            f"https://{self.api_key}:{self.api_token}@{self.shop_name}.myshopify.com/admin/api/2024-01/graphql.json"
        )

        query = """
        query products($query: String!) {
          products(query: $query, first: 1) {
            edges {
              node {
                id
                handle
                vendor
                description
                totalInventory
              }
            }
          }
        }
        """
        product_field = shopify_params["values"]["shopify_params"]["product_field"]
        variables = {"query": product_field, "shop_url": shop_url}

        product = self._send_graphql_query_request(query, variables)
        simplified_response = [edge["node"] for edge in product["data"]["products"]["edges"]]
        # print(f"RETURN: {simplified_response}")
        return TextArtifact(str(simplified_response))

    @activity(
        config={
            "description": "Can be used to get Products using a filter",
            "schema": Schema(
                {
                    Literal(
                        "shopify_params",
                        description="""A dictionary to pass into the method, the only key needed is product_filter and its value. 
                        The value should be a string that represents the query the user would be asking For example: 
                        {'product_filter': 'price:<=10'} or {'product_filter': 'inventory_total:0'}""",
                    ): dict
                }
            ),
        }
    )
    def get_product_by_filter(self, shopify_params: dict) -> TextArtifact:
        shop_url = (
            f"https://{self.api_key}:{self.api_token}@{self.shop_name}.myshopify.com/admin/api/2024-01/graphql.json"
        )

        query = """
                query products($query: String!) {
                  products(query: $query, first: 250) {
                    edges {
                      node {
                        id
                      }
                    }
                  }
                }
                """
        product_filter = shopify_params["values"]["shopify_params"]["product_filter"]
        variables = {"query": product_filter, "shop_url": shop_url}

        products = self._send_graphql_query_request(query, variables)
        # print(f"PRE-RESULT: {products}")

        simplified_response = [edge["node"] for edge in products["data"]["products"]["edges"]]
        count = len(simplified_response)
        result = {"ids": simplified_response, "count": count}
        # print(f"RETURN: {result}")
        return TextArtifact(str(result))

    @activity(
        config={
            "description": "Can be used to get a Customer using a field",
            "schema": Schema(
                {
                    Literal(
                        "shopify_params",
                        description="""
                        A dictionary to pass into the method, the only key needed is customer_field and its value.
                        For example: {'customer_field': 'John Doe'} or {'customer_field': 624072654000}
                        """,
                    ): dict
                }
            ),
        }
    )
    def get_customer_by_field(self, shopify_params: dict) -> TextArtifact:
        shop_url = (
            f"https://{self.api_key}:{self.api_token}@{self.shop_name}.myshopify.com/admin/api/2024-01/graphql.json"
        )

        query = """
        query customers($query: String!) {
          customers(query: $query, first: 1) {
            edges {
              node {
                id
                email
                phone
                displayName
                amountSpent {
                    amount
                }
                lastOrder {
                    confirmationNumber
                }
              }
            }
          }
        }
        """
        customer_field = shopify_params["values"]["shopify_params"]["customer_field"]
        variables = {"query": customer_field, "shop_url": shop_url}

        customer = self._send_graphql_query_request(query, variables)
        simplified_response = [edge["node"] for edge in customer["data"]["customers"]["edges"]]
        # print(f"RETURN: {simplified_response}")
        return TextArtifact(str(simplified_response))

    @activity(
        config={
            "description": "Can be used to get Customers using a filter",
            "schema": Schema(
                {
                    Literal(
                        "shopify_params",
                        description="""A dictionary to pass into the method, the only key needed is customer_filter and its value. 
                        The value should be a string that represents the query the user would be asking For example: 
                        {'customer_filter': 'orders_count:<=10'} or {'customer_filter': 'total_spent:0'}""",
                    ): dict
                }
            ),
        }
    )
    def get_customer_by_filter(self, shopify_params: dict) -> TextArtifact:
        shop_url = (
            f"https://{self.api_key}:{self.api_token}@{self.shop_name}.myshopify.com/admin/api/2024-01/graphql.json"
        )

        query = """
                query customers($query: String!) {
                  customers(query: $query, first: 250) {
                    edges {
                      node {
                        id
                      }
                    }
                  }
                }
                """
        customer_filter = shopify_params["values"]["shopify_params"]["customer_filter"]
        variables = {"query": customer_filter, "shop_url": shop_url}

        customers = self._send_graphql_query_request(query, variables)
        # print(f"PRE-RESULT: {customers}")

        simplified_response = [edge["node"] for edge in customers["data"]["customers"]["edges"]]
        count = len(simplified_response)
        result = {"ids": simplified_response, "count": count}
        # print(f"RETURN: {result}")
        return TextArtifact(str(result))

    @activity(
        config={
            "description": "Can be used to get an Order using a field",
            "schema": Schema(
                {
                    Literal(
                        "shopify_params",
                        description="""
                        A dictionary to pass into the method, the only key needed is order_field and its value.
                        For example: {'order_field': 'confirmation_number: 1234'} or {'order_field': 'order_id: 1234'}
                        """,
                    ): dict
                }
            ),
        }
    )
    def get_order_by_field(self, shopify_params: dict) -> TextArtifact:
        shop_url = (
            f"https://{self.api_key}:{self.api_token}@{self.shop_name}.myshopify.com/admin/api/2024-01/graphql.json"
        )

        query = """
        query orders($query: String!) {
          orders(query: $query, first: 1) {
            edges {
              node {
                id
                name
                createdAt
                confirmed
                customer {
                    id
                    displayName
                }
                discountCodes
                totalPriceSet{
                    presentmentMoney{
                        amount
                    }
                }
              }
            }
          }
        }
        """
        order_field = shopify_params["values"]["shopify_params"]["order_field"]
        variables = {"query": order_field, "shop_url": shop_url}

        order = self._send_graphql_query_request(query, variables)
        simplified_response = [edge["node"] for edge in order["data"]["orders"]["edges"]]
        # print(f"RETURN: {simplified_response}")
        return TextArtifact(str(simplified_response))

    @activity(
        config={
            "description": "Can be used to get Orders using a filter",
            "schema": Schema(
                {
                    Literal(
                        "shopify_params",
                        description="""A dictionary to pass into the method, the only key needed is order_filter and its value. 
                        The value should be a string that represents the query the user would be asking For example: 
                        {'order_filter': 'customer_id:624072654913'} or {'order_filter': 'discount_code:ENGRAVING'}""",
                    ): dict
                }
            ),
        }
    )
    def get_order_by_filter(self, shopify_params: dict) -> TextArtifact:
        shop_url = (
            f"https://{self.api_key}:{self.api_token}@{self.shop_name}.myshopify.com/admin/api/2024-01/graphql.json"
        )

        query = """
                query orders($query: String!) {
                  orders(query: $query, first: 250) {
                    edges {
                      node {
                        id
                      }
                    }
                  }
                }
                """
        order_filter = shopify_params["values"]["shopify_params"]["order_filter"]
        variables = {"query": order_filter, "shop_url": shop_url}

        orders = self._send_graphql_query_request(query, variables)
        # print(f"PRE-RESULT: {orders}")

        simplified_response = [edge["node"] for edge in orders["data"]["orders"]["edges"]]
        count = len(simplified_response)
        result = {"ids": simplified_response, "count": count}
        # print(f"RETURN: {result}")
        return TextArtifact(str(result))


# Broken Right now


# @activity(
#     config={
#         "description": "Can be used to get Orders using a filter based on the amount",
#         "schema": Schema(
#             {
#                 Literal(
#                     "shopify_params",
#                     description=
#                     """A dictionary to pass into the method, the only key needed is order_amount_filter and its
#                     value. The value should be a string that represents the query the user would be asking For
#                     example: {'order_amount_filter': 'total_value:>50000'}: """
#                     ,
#                 ): dict,
#             },
#         ),
#     }
# )
# def get_order_by_amount_filter(self, shopify_params: dict) -> TextArtifact:
#     shop_url = f"https://{self.api_key}:{self.api_token}@{self.shop_name}.myshopify.com/admin/api/2024-01/graphql.json"
#
#     query = """
#             query orders($query: String!) {
#               orders(sortKey: $query, first: 250) {
#                 edges {
#                   node {
#                     id
#                   }
#                 }
#               }
#             }
#             """
#     order_filter = shopify_params['values']['shopify_params']['order_filter']
#     variables = {
#         "query": order_filter,
#         "shop_url": shop_url
#     }
#
#     orders = self._send_graphql_query_request(query, variables)
#     print(f"PRE-RESULT: {orders}")
#
#     simplified_response = [edge['node'] for edge in orders['data']['orders']['edges']]
#     count = len(simplified_response)
#     result = {'ids': simplified_response, 'count': count}
#     print(f"RETURN: {result}")
#     return TextArtifact(str(result))
