"""
This module simulates a simple e-commerce marketplace with concurrent producers
and consumers.

It defines three main classes:
- `Marketplace`: A central, thread-safe hub that manages product inventory,
  producer registration, and consumer carts.
- `Producer`: A thread that continuously publishes a set of products to the
  marketplace.
- `Consumer`: A thread that simulates a user performing actions like adding
  items to a cart, removing them, and finally placing an order.

The simulation uses threading to model concurrent access to the marketplace and
implements a retry mechanism for operations that cannot be completed immediately
(e.g., adding an out-of-stock item).
"""

import time
from threading import Thread


class Consumer(Thread):
    """Represents a consumer that interacts with the marketplace.

    Each consumer runs in its own thread, processing a predefined list of
    shopping actions (adding or removing items from a cart).
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping actions to perform. Each action is a
                dictionary specifying the product, quantity, and type ('add'
                or 'remove').
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying a
                failed action.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        super().__init__(**kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # Each consumer gets a unique cart ID from the marketplace.
        self.cart_id = self.marketplace.new_cart()

    def run(self):
        """The main execution logic for the consumer."""
        # Process all actions in the assigned shopping list.
        for cart in self.carts:
            for data in cart:
                for i in range(data["quantity"]):
                    ret = False
                    # Retry the action until it succeeds.
                    while not ret:
                        if data["type"] == "add":
                            ret = self.marketplace.add_to_cart(self.cart_id, data["product"])
                        else:
                            ret = self.marketplace.remove_from_cart(self.cart_id, data["product"])
                        if not ret:
                            # Wait before retrying if the action failed.
                            time.sleep(self.retry_wait_time)
        # After all actions are done, place the final order.
        self.marketplace.place_order(self.cart_id)

import time
from threading import Lock
import logging
import logging.handlers


class Marketplace:
    """A thread-safe marketplace for producers and consumers.

    This class acts as the central shared resource, managing products, carts,
    and producer/consumer registration. All operations that modify shared state
    are protected by a single lock to ensure thread safety.
    """
    # Class-level setup for logging.
    formatter = logging.Formatter('(%(asctime)s %(threadName)-9s) %(funcName)s %(message)s')
    formatter.converter = time.gmtime
    handler = logging.handlers.RotatingFileHandler('marketplace.log')
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # List of all products currently available in the marketplace.
    MARK = []

    # A dictionary to track which producer published which product.
    GET_PROD = {}

    # A dictionary mapping producer_id to their remaining publishing capacity.
    PROD = {}

    # A dictionary representing consumer carts. Maps cart_id -> {producer_id -> [products]}.
    CONS = {}

    # A single lock to protect all shared data structures.
    lock = Lock()

    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products each
                producer can have "in-flight" in the marketplace at one time.
        """
        self.logger.info(f" <- queue_zie_per_producer = {queue_size_per_producer}")
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        """Registers a new producer and returns a unique producer ID."""
        self.lock.acquire()
        producer_id = len(self.PROD)
        self.PROD[producer_id] = self.queue_size_per_producer
        self.lock.release()
        self.logger.info(f" -> producer_id = {producer_id}")
        return producer_id

    def publish(self, producer_id, product):
        """Allows a producer to publish a product to the marketplace.

        Returns:
            bool: True if the product was published successfully, False if the
                  producer has reached their publishing capacity.
        """
        self.logger.info(f" <- producer_id = {producer_id}, product = {product}")
        self.lock.acquire()
        if self.PROD[producer_id] > 0:
            self.PROD[producer_id] -= 1
            self.MARK.append(product)
            self.GET_PROD[product] = producer_id
            self.lock.release()
            self.logger.info(f" -> True")
            return True
        self.lock.release()
        self.logger.info(f" -> False")
        return False

    def new_cart(self):
        """Creates a new empty cart for a consumer and returns a unique cart ID."""
        self.lock.acquire()
        cart_id = len(self.CONS)
        self.CONS[cart_id] = {}
        self.lock.release()
        self.logger.info(f" -> cart_id = {cart_id}")
        return cart_id

    def add_to_cart(self, cart_id, product):
        """Adds a product from the marketplace to a consumer's cart.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        self.logger.info(f" <- cart_id = {cart_id}, product = {product}")
        self.lock.acquire()
        try:
            self.MARK.remove(product)
        except ValueError:
            # The product is not in the marketplace.
            self.lock.release()
            self.logger.info(f" -> False")
            return False
        # If added to cart, the producer's capacity is restored.
        producer_id = self.GET_PROD[product]
        try:
            self.CONS[cart_id][producer_id].append(product)
        except KeyError:
            self.CONS[cart_id][producer_id] = []
            self.CONS[cart_id][producer_id].append(product)
        self.PROD[producer_id] += 1
        self.lock.release()
        self.logger.info(f" -> True")
        return True

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a consumer's cart and returns it to the marketplace.

        Returns:
            bool: True if the product was found and removed, False otherwise.
        """
        self.logger.info(f" <- cart_id = {cart_id}, product = {product}")
        self.lock.acquire()
        for entry in self.CONS[cart_id]:
            for search_product in self.CONS[cart_id][entry]:
                if product == search_product:
                    self.CONS[cart_id][entry].remove(search_product)
                    self.MARK.append(product)
                    self.PROD[entry] -= 1
                    self.lock.release()
                    self.logger.info(f" -> True")
                    return True
        self.lock.release()
        self.logger.info(f" -> False")
        return False

    def place_order(self, cart_id):
        """Simulates placing an order by printing the contents of the cart."""
        self.logger.info(f" <- cart_id = {cart_id}")
        for prod_list in self.CONS[cart_id].values():
            for prod in prod_list:
                self.lock.acquire()
                print(f'cons{cart_id + 1} bought {prod}')
                self.lock.release()

import time
from threading import Thread


class Producer(Thread):
    """Represents a producer that publishes products to the marketplace.

    Each producer runs in its own thread, continuously attempting to publish
    a list of products according to a specified quantity and frequency.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes a Producer thread.

        Args:
            products (list): A list of products to publish. Each element is a
                tuple of (product_name, quantity, publish_wait_time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time in seconds to wait before
                re-publishing the entire list of products.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """The main execution logic for the producer."""
        # The producer runs in an infinite loop.
        while True:
            for product in self.products:
                for i in range(product[1]):
                    ret = False
                    # Retry publishing until it succeeds.
                    while not ret:
                        ret = self.marketplace.publish(self.producer_id, product[0])
                        time.sleep(product[2])
            # Wait before starting the next cycle of publications.
            time.sleep(self.republish_wait_time)
