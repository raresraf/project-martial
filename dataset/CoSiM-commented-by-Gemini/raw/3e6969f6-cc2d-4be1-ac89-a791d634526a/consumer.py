
"""
This module simulates a marketplace with a producer-consumer model using
Python's threading library.

It defines four main classes:
- Consumer: A thread that simulates a customer adding items to a cart and buying them.
- Marketplace: The central shared resource that manages inventory, carts, and all
  interactions between producers and consumers. It uses a coarse-grained lock
  for synchronization.
- Producer: A thread that simulates a producer creating products and publishing
  them to the marketplace.
- TestMarketplace: A suite of unit tests to verify the Marketplace's functionality.
"""


from threading import Thread, RLock
import time
import unittest
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler
import sys

# The original file has a sys.path modification, which is kept as-is.
sys.path.insert(1, './tema/')
from product import *


class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace.

    The consumer's behavior is driven by a list of predefined "cart" actions,
    which specify products to add or remove.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        Args:
            carts: A list of shopping lists, where each list contains actions
                   (add/remove) for specific products.
            marketplace: The shared Marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying to
                                   add a product if it's not available.
            **kwargs: Keyword arguments for the Thread base class.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution logic for the consumer thread.

        For each shopping list, it creates a new cart, performs the add/remove
        actions, and finally places the order.
        """
        for cart in self.carts:
            consumer_id = self.marketplace.new_cart()
            for product_action in cart:
                size = product_action["quantity"]
                # Block Logic: Perform the 'add' action for the specified quantity.
                if product_action["type"] == "add":
                    # Invariant: This loop continues until the desired quantity of the
                    # product has been successfully added to the cart.
                    while size > 0:
                        if self.marketplace.add_to_cart(consumer_id, product_action["product"]) is True:
                            size -= 1
                        else:
                            # Inline: If the product is not available, wait before retrying.
                            time.sleep(self.retry_wait_time)
                # Block Logic: Perform the 'remove' action.
                else:
                    while size > 0:
                        self.marketplace.remove_from_cart(
                            consumer_id, product_action["product"])
                        size -= 1

            final_products = self.marketplace.place_order(consumer_id)
            for product in final_products:
                print(self.kwargs['name'], "bought", product, flush=True)


class Marketplace:
    """
    Manages the inventory and all producer-consumer interactions.

    This class acts as the synchronized shared resource. It uses a single
    re-entrant lock (RLock) to protect its shared data, resulting in a
    coarse-grained concurrency model where only one thread can operate on
    the marketplace's state at a time.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can list in the marketplace.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_size = 0
        self.consumer_size = 0
        self.lock = RLock()
        self.carts = []  # A list of lists, representing each consumer's cart.
        self.shop_items = []  # A flat list of all available products.
        self.products_from_producer = []  # Tracks item count per producer.
        
        # Set up logging for marketplace events.
        self.logger = logging.getLogger('marketplace')
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(
            'marketplace.log', maxBytes=2000, backupCount=10)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)8s: %(message)s')
        handler.setFormatter(formatter)
        logging.Formatter.converter = time.gmtime
        self.logger.addHandler(handler)

    def register_producer(self):
        """
        Registers a new producer and returns a unique producer ID.

        This operation is thread-safe.
        """
        self.logger.info("Entering register_producer function")
        with self.lock:
            producer_id = self.producer_size
            self.producer_size += 1
            self.products_from_producer.append(0)
        self.logger.info(
            "Leaving register_producer function with result %d", producer_id)
        return producer_id

    def publish(self, producer_id, product):
        """
        Adds a product from a specific producer to the marketplace.

        Args:
            producer_id (int): The ID of the producer.
            product: The product to be published.

        Returns:
            bool: True if the product was successfully published, False if the
                  producer's queue is full.
        """
        self.logger.info(
            "Entering publish function with producer_id=%d and product=%s", producer_id, product)
        # Pre-condition: Check if the producer has reached their publication limit.
        if self.products_from_producer[producer_id] == self.queue_size_per_producer:
            self.logger.info("Leaving publish function with result %r", False)
            return False
        else:
            prod = {"id": producer_id, "product": product}

            self.products_from_producer[producer_id] += 1
            self.shop_items.append(prod)
            self.logger.info("Leaving publish function with result %r", True)
            return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.

        This operation is thread-safe.

        Returns:
            int: A unique ID for the newly created cart.
        """
        self.logger.info("Entering new_cart function")
        with self.lock:
            cart_id = self.consumer_size
            self.consumer_size += 1
            self.carts.append([])
        self.logger.info("Leaving new_cart function with result %d", cart_id)
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Moves a product from the main shop inventory into a consumer's cart.

        This operation is thread-safe.

        Args:
            cart_id (int): The ID of the target cart.
            product: The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        self.logger.info(
            "Entering add_to_cart function with cart_id =%d and product=%s", cart_id, product)

        found = False
        # Block Logic: Atomically search for and claim a product.
        # A global lock ensures that only one consumer can modify the shop_items list at a time.
        with self.lock:
            for prod in self.shop_items:
                if prod["product"] == product:
                    self.carts[cart_id].append(prod)
                    found = True
                    self.shop_items.remove(prod)
                    break
        self.logger.info(
            "Leaving add_to_cart function with result %r", found)
        return found

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a consumer's cart and returns it to the main shop inventory.
        
        Note: This method as written is not thread-safe and may cause race conditions
        if called concurrently with other marketplace operations.
        """
        self.logger.info(
            "Entering remove_from_cart function with cart_id =%d and product=%s", cart_id, product)
        for prod in self.carts[cart_id]:
            if prod["product"] == product:
                self.shop_items.append(prod)
                self.carts[cart_id].remove(prod)
                break
        self.logger.info("Leaving add_to_cart function")

    def place_order(self, cart_id):
        """
        Finalizes an order by preparing the list of items to be "bought".

        This method decrements the producer's item count for each product in the cart
        and returns the final list of products.

        Note: This method as written is not thread-safe.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: A list of product objects that were in the cart.
        """
        self.logger.info(
            "Entering place_order function with cart_id =%d", cart_id)
        final_list = []
        for prod in self.carts[cart_id]:
            self.products_from_producer[prod["id"]] -= 1
            final_list.append(prod["product"])
        self.logger.info(
            "Leaving place_order function with list: %s", final_list)
        return final_list


class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace class to verify its core logic.
    """

    def setUp(self):
        """Initializes a new Marketplace instance before each test."""
        self.marketplace = Marketplace(3)

    def test_register_producer(self):
        """Tests that producer registration returns sequential IDs."""
        self.assertEqual(self.marketplace.register_producer(), 0)
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)

    def test_new_cart(self):
        """Tests that new cart creation returns sequential IDs."""
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)

    def test_publish(self):
        """Tests that publishing respects the per-producer queue limit."""
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='White Peach', price=5, type='White')))
        self.assertTrue(self.marketplace.publish(
            1, Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')))
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='English Breakfast', price=2, type='Black')))
        # This should fail as the producer's queue (size 3) is full.
        self.assertFalse(self.marketplace.publish(
            1, Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM')))

    def test_add_to_cart(self):
        """Tests adding available and unavailable items to a cart."""
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.assertTrue(self.marketplace.publish(
            1, Tea(name='White Peach', price=5, type='White')))
        # Add an available item.
        self.assertTrue(self.marketplace.add_to_cart(
            0, Tea(name='White Peach', price=5, type='White')))
        # Try to add the same item again (it should not be available).
        self.assertFalse(self.marketplace.add_to_cart(
            0, Tea(name='White Peach', price=5, type='White')))

    def test_remove_from_cart(self):
        """Tests removing an item from a cart."""
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        product = Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')
        self.marketplace.publish(0, product)
        self.marketplace.add_to_cart(0, product)
        self.assertEqual(len(self.marketplace.carts[0]), 1)
        self.marketplace.remove_from_cart(0, product)
        self.assertEqual(len(self.marketplace.carts[0]), 0)

    def test_place_order(self):
        """Tests the full workflow of adding items and placing an order."""
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        p1 = Tea(name='White Peach', price=5, type='White')
        p2 = Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')
        self.marketplace.publish(0, p1)
        self.marketplace.publish(0, p2)
        self.marketplace.add_to_cart(0, p1)
        self.marketplace.add_to_cart(0, p2)
        
        final_order = self.marketplace.place_order(0)
        self.assertEqual(len(final_order), 2)
        self.assertIn(p1, final_order)
        self.assertIn(p2, final_order)


class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer instance.

        Args:
            products: A list of products that the producer will create.
            marketplace: The shared Marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish.
            **kwargs: Keyword arguments for the Thread base class.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution logic for the producer thread.

        Registers with the marketplace and then continuously produces and
        publishes its assigned products as long as its daemon flag is set.
        """
        producer_id = self.marketplace.register_producer()
        # This loop condition is unconventional; a simple `while True` is more common for daemon threads.
        while self.kwargs.get('daemon') is True:
            for product_info in self.products:
                product_to_produce = product_info[0]
                quantity = product_info[1]
                production_time = product_info[2]
                
                # Invariant: This loop ensures the specified quantity of the product is published.
                count_product = quantity
                while count_product > 0:
                    if self.marketplace.publish(producer_id, product_to_produce) is True:
                        count_product -= 1
                        time.sleep(production_time)
                    else:
                        # Inline: If publishing fails (e.g., queue is full), wait before retrying.
                        time.sleep(self.republish_wait_time)


# The original file has these dataclasses at the end, so they are kept here.
@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a product with a name and price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass representing Coffee, inheriting from Product."""
    acidity: str
    roast_level: str
