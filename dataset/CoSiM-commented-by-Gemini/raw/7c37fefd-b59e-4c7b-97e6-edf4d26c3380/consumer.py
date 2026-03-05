"""
This module provides a complex, object-oriented simulation of a marketplace.

It features a producer-consumer model with multiple interacting classes to
represent different components of the system, such as Products, Carts, and
Operations. Unlike simpler models, this simulation employs a fine-grained locking
mechanism where each product in the marketplace has its own lock, aiming for
higher concurrency. The file appears to be a composite of several modules.
"""

import time


from threading import Thread
from typing import List

from tema.marketplace import Marketplace
from tema.product import Product


class Operation:
    """A simple data class to represent a consumer's shopping operation."""
    def __init__(self, op_type: str, product: Product, quantity: int):
        self.op_type: str = op_type
        self.product: Product = product
        self.quantity: int = quantity


class Consumer(Thread):
    """Represents a consumer that performs a series of shopping operations."""

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes the Consumer thread.

        Args:
            carts (list): A list of lists of operations to be performed.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying a failed operation.
            **kwargs: Keyword arguments for the Thread base class.
        """
        super(Consumer, self).__init__(**kwargs)
        self.carts: List[List[Operation]] = []
        for cart_operations in carts:
            ops = []
            for operation in cart_operations:
                ops.append(Operation(
                    operation['type'],
                    operation['product'],
                    operation['quantity'])
                )
            self.carts.append(ops)

        self.marketplace: Marketplace = marketplace
        self.retry_wait_time: float = retry_wait_time
        self.products: List[Product] = []

    def run(self):
        """Main execution loop for the consumer."""
        for operations in self.carts:
            cart_id: int = self.marketplace.new_cart()

            # Process operations sequentially from the list.
            while operations:
                if operations[0].op_type == 'add':
                    # Attempt to add a product. If it fails, sleep and retry.
                    if self.marketplace.add_to_cart(cart_id, operations[0].product):
                        operations[0].quantity -= 1
                    else:
                        time.sleep(self.retry_wait_time)
                        continue

                    # If all quantity of an operation is done, move to the next.
                    if operations[0].quantity == 0:
                        operations = operations[1:]
                elif operations[0].op_type == 'remove':
                    # For removal, loop until the desired quantity is removed.
                    while operations[0].quantity > 0:
                        self.marketplace.remove_from_cart(cart_id, operations[0].product)
                        operations[0].quantity -= 1
                    # Move to the next operation.
                    operations = operations[1:]

            # Finalize the purchase.
            final_products: List[Product] = self.marketplace.place_order(cart_id)
            for product in final_products:
                print(self.name + " bought " + str(product))



from threading import Lock
from typing import List, Dict

from tema.product import Product


class MarketplaceProduct:
    """A wrapper for a Product that includes metadata for the marketplace.

    Attributes:
        producer_id (int): The ID of the producer who created the product.
        product (Product): The product itself.
        lock (Lock): A lock specific to this individual product instance,
                     allowing for fine-grained concurrency control.
    """
    def __init__(self, producer_id: int, product: Product):
        self.producer_id = producer_id
        self.product = product
        self.lock: Lock = Lock()


class Cart:
    """Represents a consumer's shopping cart, containing a list of products."""
    def __init__(self, cart_id: int):
        self.cart_id: int = cart_id
        self.products: List[MarketplaceProduct] = []

    def add_product(self, product: MarketplaceProduct):
        """Adds a product to the cart."""
        self.products.append(product)

    def remove_product(self, product: MarketplaceProduct):
        """Removes a product from the cart."""
        if product in self.products:
            self.products.remove(product)

    def find_product_in_cart(self, product) -> [MarketplaceProduct]:
        """Finds a specific product within the cart."""
        for market_product in self.products:
            if market_product.product == product:
                return market_product

        return None


class Marketplace:
    """A more complex marketplace implementation with per-product locking."""

    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The max number of items a producer
                can have listed at one time.
        """
        self.max_items: int = queue_size_per_producer
        self.consumers: List[int] = []
        self.producers: List[int] = []

        # Data structures for tracking products and carts.
        self.products: Dict[int, List[MarketplaceProduct]] = {}
        self.carts: Dict[int, Cart] = {}

    def register_producer(self) -> int:
        """Registers a new producer, returning a unique ID."""
        producer_id = len(self.producers)
        self.producers.append(producer_id)
        self.products[producer_id] = []
        return producer_id

    def publish(self, producer_id, product):
        """Publishes a product from a given producer.

        Note: This method is not thread-safe and may have race conditions if a
        producer uses multiple threads.
        """
        if len(self.products[producer_id]) > self.max_items:
            return False
        else:
            self.products[producer_id].append(MarketplaceProduct(producer_id, product))
            return True

    def new_cart(self):
        """Creates a new cart for a consumer, returning a unique cart ID."""
        cart_id = len(self.carts) + 1
        self.carts[cart_id] = Cart(cart_id)
        return cart_id

    def add_to_cart(self, cart_id, product):
        """Adds a product to a cart.

        Note: This method is highly inefficient as it performs a linear scan
        through all products from all producers to find an available item.
        This will be a significant bottleneck in a large-scale simulation.
        """
        for producer_products in self.products.values():
            for market_product in producer_products:
                # Find an available product that matches and is not already locked.
                if market_product.product == product and not market_product.lock.locked():
                    market_product.lock.acquire()  # Reserve the product.
                    self.carts[cart_id].add_product(market_product)
                    return True
        return False

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart, releasing its lock.

        Note: This simply releases the lock, making the item available for
        another consumer to potentially add via `add_to_cart`. It does not
        return the item to the producer's list.
        """
        market_product: MarketplaceProduct = self.carts[cart_id].find_product_in_cart(product)
        self.carts[cart_id].remove_product(market_product)
        market_product.lock.release()

    def place_order(self, cart_id) -> List[Product]:
        """Finalizes a purchase, removing items from the marketplace permanently."""
        final_products = [marketProduct.product for marketProduct in self.carts[cart_id].products]

        # Remove the purchased items from the producer's product list.
        for market_product in self.carts[cart_id].products:
            for producer_product in self.products[market_product.producer_id]:
                if producer_product == market_product:
                    self.products[market_product.producer_id].remove(market_product)
                    break
        return final_products

import time


from threading import Thread
from typing import List

from tema.marketplace import Marketplace
from tema.product import Product



class Production:
    """A data class to manage the production of a specific product type."""
    def __init__(self, product: Product, quantity: int, wait_time: float):
        self.product: Product = product
        self.quantity: int = quantity
        self.wait_time: float = wait_time
        self.number_produced = 0


class Producer(Thread):
    """Represents a producer that operates in a round-robin fashion."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the Producer thread.

        Args:
            products (list): A list of product definitions to be produced.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Not used in this implementation's logic.
            **kwargs: Keyword arguments for the Thread base class.
        """
        super(Producer, self).__init__(**kwargs)
        self.productions: List[Production] = []
        for prod in products:
            self.productions.append(Production(prod[0], prod[1], prod[2]))
        self.marketplace: Marketplace = marketplace
        self.republish_wait_time: float = republish_wait_time
        self.producer_id: int = self.marketplace.register_producer()

    def run(self):
        """Main execution loop for the producer.

        This loop continuously produces items in a round-robin cycle. It focuses
        on one product type until its quantity is met, then rotates it to the
        back of the list and starts on the next.
        """
        while True:
            # Check if the current production order is complete.
            if self.productions[0].number_produced < self.productions[0].quantity:
                # Attempt to publish one item.
                if self.marketplace.publish(self.producer_id, self.productions[0].product):
                    self.productions[0].number_produced += 1
                # Wait before producing the next item.
                time.sleep(self.productions[0].wait_time)
            else:
                # When production for this item is done, reset and rotate.
                self.productions[0].number_produced = 0
                self.productions = self.productions[1:] + [self.productions[0]]
