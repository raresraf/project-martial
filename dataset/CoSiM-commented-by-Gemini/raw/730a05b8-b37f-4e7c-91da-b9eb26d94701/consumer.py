"""
This module simulates a multi-threaded producer-consumer marketplace.

It defines classes for a `Marketplace`, which acts as the central shared
resource, `Consumer` threads that acquire products, and `Producer` threads
that supply products. The simulation uses locks to ensure data consistency in a concurrent environment.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Represents a consumer thread that simulates a user's shopping process.

    Each consumer is initialized with a set of carts, each containing a list of
    actions (add/remove products). The consumer processes these actions against
    the shared marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping carts, where each cart is a list of actions.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying a failed action.
            **kwargs: Keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # A single cart is created for the lifetime of this consumer instance.
        self.cart_id = marketplace.new_cart()

    def run(self):
        """The main execution logic for the consumer thread."""
        # Invariant: The consumer processes each list of shopping actions.
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            # Invariant: Process each action within a cart.
            for cart_item in cart:
                count = 0
                quantity = cart_item["quantity"]
                action = cart_item["type"]
                product = cart_item["product"]

                while count < quantity:
                    # Block Logic: Attempt to add the specified quantity of a product to the cart.
                    if action == "add":
                        # Pre-condition: If adding fails (e.g., out of stock),
                        # block and retry until successful.
                        add = self.marketplace.add_to_cart(cart_id, product)
                        if add is True:
                            count += 1
                        else:
                            time.sleep(self.retry_wait_time)
                    # Block Logic: Remove the specified quantity of a product from the cart.
                    else:
                        self.marketplace.remove_from_cart(cart_id, product)
                        count += 1

            # Finalize the transaction for the current cart.
            for order_product in self.marketplace.place_order(cart_id):
                print(self.name + " bought " + str(order_product))

# --- Start of Marketplace and Testing section ---
# Note: This appears to be a separate file concatenated with the Consumer class.

import unittest
from threading import Lock
import logging
import time
from logging.handlers import RotatingFileHandler
import os
from .product import Tea, Coffee

logging.basicConfig(filename="marketplace.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logging.Formatter.converter = time.gmtime
logger = logging.getLogger()

handler = RotatingFileHandler("marketplace.log", mode='w', backupCount=10)



if os.path.isfile("marketplace.log"):
    handler.doRollover()

logger.setLevel(logging.DEBUG)


class Marketplace:
    """
    A thread-safe marketplace that manages producers, products, and customer carts.

    This class acts as the central shared resource, using locks to coordinate
    concurrent access from multiple producer and consumer threads.
    """
    def __init__(self, queue_size_per_producer):
        """Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products each
                producer can have in their published queue at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producers = []
        self.products = {}
        self.carts = []

        self.producers_lock = Lock()


        self.carts_lock = Lock()

    def register_producer(self):
        """
        Allocates a unique ID for a new producer and sets up their inventory space.

        Returns:
            int: The unique ID for the registered producer.
        """
        logger.info("Entered register_producer")
        with self.producers_lock:
            producer_id = len(self.producers)
            self.producers.append([])
            self.products[str(producer_id)] = []
            logger.info("Exited register_producer with id %d", producer_id)
            return producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        The operation is thread-safe and respects the producer's queue size limit.

        Args:
            producer_id (int): The ID of the producer.
            product: The product to be published.

        Returns:
            bool: True if the product was published successfully, False if the
                  producer's queue was full.
        """
        logger.info("Entered publish with producer_id %s and product %s",
                    producer_id, str(product))
        if len(self.producers[int(producer_id)]) < self.queue_size_per_producer:
            
            self.producers[int(producer_id)].append(product)
            
            self.products[producer_id].append(product)
            logger.info("Exited publish")
            return True

        logger.info("Exited publish")
        return False

    def new_cart(self):
        """
        Creates a new shopping cart and returns its unique ID.

        Returns:
            int: The unique ID for the new cart.
        """

        logger.info("Entered new_cart")
        with self.carts_lock:
            
            cart_id = len(self.carts)
            self.carts.append({})
            logger.info("Exited new_cart with id %d", cart_id)
            return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds an available product to a specified shopping cart.

        This method searches through all producer inventories for an available
        (not booked) item of the requested product type. If found, it marks the
        item as booked and adds it to the cart.

        Args:
            cart_id (int): The ID of the cart to add to.
            product: The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        logger.info("Entered add_to_cart with cart_id %d and product %s", cart_id, product)
        for producer_id, prods in self.products.items():
            if product in prods:
                
                prods.remove(product)
                
                if producer_id in self.carts[cart_id]:
                    self.carts[cart_id][producer_id].append(product)
                else:
                    self.carts[cart_id][producer_id] = []


                    self.carts[cart_id][producer_id].append(product)
                logger.info("Exited add_to_cart")
                return True
        logger.info("Exited add_to_cart")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart, making it available again.

        Args:
            cart_id (int): The ID of the cart.
            product: The product to remove.
        """
        logger.info("Entered remove_from_cart with cart_id %d and product %s", cart_id, product)
        for producer_id in self.carts[cart_id]:
            if product in self.carts[cart_id][producer_id]:
                
                self.carts[cart_id][producer_id].remove(product)
                self.products[producer_id].append(product)
                logger.info("Exited remove_from_cart")
                break

    def place_order(self, cart_id):
        """
        Finalizes an order, permanently removing items from the marketplace inventory.

        Args:
            cart_id (int): The ID of the cart being ordered.

        Returns:
            list: A list of the products that were in the cart.
        """
        logger.info("Entered place_order with cart_id %d", cart_id)
        order = []
        for producer_id in self.carts[cart_id]:
            
            order = order + self.carts[cart_id][producer_id]
            for product in self.carts[cart_id][producer_id]:
                
                self.producers[int(producer_id)].remove(product)
        logger.info("Exited place_order")
        return order


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace logic."""

    def setUp(self):
        """Initializes a marketplace and products for each test."""
        self.marketplace = Marketplace(20)
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.new_cart()

    def test_register_producer(self):
        """Tests if producer registration provides sequential IDs."""
        self.assertEqual(self.marketplace.register_producer(), 2,
                         'Error: wrong producer id')
        self.assertEqual(self.marketplace.producers[0], [],
                         'Error: wrong list of products for the producer')
        self.assertEqual(self.marketplace.producers[1], [],
                         'Error: wrong list of products for the producer')
        self.assertEqual(self.marketplace.producers[2], [],
                         'Error: wrong list of products for the producer')
        self.assertEqual(len(self.marketplace.producers[0]), 0,
                         'Error: wrong number of products for the producer')
        self.assertEqual(len(self.marketplace.producers[1]), 0,
                         'Error: wrong number of products for the producer')
        self.assertEqual(len(self.marketplace.producers[2]), 0,
                         'Error: wrong number of products for the producer')

    def publish(self):
        """Tests that publishing respects the producer's queue size limit."""
        self.assertEqual(self.marketplace.publish("0", Tea("Twinings", 7, "Black Tea")), True,
                         'Error: wrong published item')
        self.assertEqual(self.marketplace.publish("1", Coffee("Colombia", 7, "5.05", "HIGH")),
                         True, 'Error: wrong published item')

    def test_new_cart(self):
        """Tests if new cart creation provides sequential IDs."""
        self.assertEqual(self.marketplace.new_cart(), 2,
                         'Error: wrong cart id')

    def test_add_to_cart(self):
        """Tests that a product can be successfully added to a cart if available."""
        self.marketplace.publish("0", Tea("Twinings", 7, "Black Tea"))
        self.marketplace.publish("0", Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.publish("0", Coffee("Colombia", 7, "5.05", "HIGH"))

        self.assertEqual(self.marketplace.add_to_cart(0, Coffee("Colombia", 7, "5.05", "HIGH")),
                         True, 'Error: wrong product added to cart')
        self.assertEqual(self.marketplace.add_to_cart(0, Coffee("India", 7, "5.05", "LOW")),
                         True, 'Error: wrong product added to cart')
        self.assertEqual(self.marketplace.add_to_cart(0, Coffee("Colombia", 7, "5.05", "HIGH")),
                         False, 'Error: wrong product added to cart')
        self.assertEqual(self.marketplace.add_to_cart(1, Tea("Twinings", 7, "Black Tea")), True,
                         'Error: wrong product added to cart')
        self.assertEqual(self.marketplace.add_to_cart(1, Tea("Twinings", 7, "Black Tea")), False,
                         'Error: wrong product added to cart')

    def test_remove_from_cart(self):
        """Tests that removing a product makes it available again."""
        self.marketplace.publish("0", Tea("Twinings", 7, "Black Tea"))
        self.marketplace.publish("0", Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.publish("0", Coffee("Colombia", 7, "5.05", "HIGH"))

        self.marketplace.add_to_cart(0, Tea("Twinings", 7, "Black Tea"))
        self.marketplace.add_to_cart(0, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(0, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(1, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(1, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(1, Coffee("Colombia", 7, "5.05", "HIGH"))

        self.assertEqual(self.marketplace.remove_from_cart(0, Tea("Twinings", 7, "Black Tea")),
                         None, 'Error: wrong product removed from cart')
        self.assertEqual(self.marketplace.remove_from_cart(1, Coffee("India", 7, "5.05", "LOW")),
                         None, 'Error: wrong product removed from cart')
        self.assertEqual(self.marketplace.remove_from_cart(1, Coffee("Colombia", 7, "5.05",
                                                                     "HIGH")),
                         None, 'Error: wrong product removed from cart')

    def test_place_order(self):
        """Tests the end-to-end process of adding and placing an order."""
        self.marketplace.publish("0", Tea("Twinings", 7, "Black Tea"))
        self.marketplace.publish("0", Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.publish("1", Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.publish("1", Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.publish("0", Coffee("Colombia", 7, "5.05", "HIGH"))

        self.marketplace.add_to_cart(0, Tea("Twinings", 7, "Black Tea"))
        self.marketplace.add_to_cart(0, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(0, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(1, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(1, Coffee("India", 7, "5.05", "LOW"))
        self.marketplace.add_to_cart(1, Coffee("Colombia", 7, "5.05", "HIGH"))

        order0 = [Tea("Twinings", 7, "Black Tea"), Coffee("India", 7, "5.05", "LOW"),
                  Coffee("India", 7, "5.05", "LOW")]

        order1 = [Coffee("India", 7, "5.05", "LOW"), Coffee("Colombia", 7, "5.05", "HIGH")]

        self.assertEqual(self.marketplace.place_order(0), order0,
                         'Error: wrong order')

        self.assertEqual(self.marketplace.place_order(1), order1,
                         'Error: wrong order')

    if __name__ == '__main__':
        unittest.main()


class Producer(Thread):
    """
    Represents a producer thread that continuously adds products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the Producer thread.

        Args:
            products (list): A list of (product, quantity, sleep_time) tuples.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying a publish.
            **kwargs: Additional keyword arguments for the `Thread` constructor.
        """

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        Thread.__init__(self, **kwargs)
        self.producer_id = self.marketplace.register_producer()


    def run(self):
        """The main execution loop for the producer.

        Continuously loops through its product list, publishing each one the
        specified number of times. Retries with a delay if the marketplace is full.
        """
        while True:
            for (prod, quant, w_time) in self.products:
                count = 0
                while count < quant:
                    # Attempt to publish until successful.
                    if self.marketplace.publish(str(self.producer_id), prod):
                        count += 1
                        time.sleep(w_time)
                    else:
                        # If queue is full, wait and retry.
                        time.sleep(self.republish_wait_time)
