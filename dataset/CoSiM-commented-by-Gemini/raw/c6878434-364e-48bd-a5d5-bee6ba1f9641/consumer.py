"""
@file raw/c6878434-364e-48bd-a5d5-bee6ba1f9641/consumer.py
@brief A multi-threaded producer-consumer marketplace simulation with critical concurrency flaws.

This module attempts to simulate a marketplace with multiple producer threads that
publish products and multiple consumer threads that add/remove products from
shopping carts and place orders.

@warning The concurrency implementation in this module is fundamentally broken.
         The `Marketplace` class, which is the central shared resource, is not
         thread-safe. Locks are created locally within methods instead of being
         shared instance attributes, rendering them useless for providing mutual
         exclusion between threads. This leads to numerous race conditions,
         including non-atomic ID increments and non-atomic read/modify/write
         sequences on shared lists. The documentation below describes the
         intended functionality but also highlights these specific flaws.
"""

import time
from threading import Thread, currentThread, Lock
from dataclasses import dataclass
import logging
from logging.handlers import RotatingFileHandler

# --- Product Data Classes ---

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base data class representing a generic product."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A data class for Tea, inheriting from Product."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A data class for Coffee, inheriting from Product."""
    acidity: str
    roast_level: str


# --- Marketplace Simulation ---

logging.basicConfig(handlers=
                    [RotatingFileHandler(filename='./marketplace.log', maxBytes=400000,
                                         backupCount=10)],
                    level=logging.INFO,
                    format="[%(asctime)s]::%(levelname)s::%(message)s")
logging.Formatter.converter = time.gmtime


class Marketplace:
    """
    Manages all shared state for the marketplace, including producer stock and
    consumer carts. This class is intended to be thread-safe, but its
    implementation is flawed.
    """

    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer
        self.list_of_carts = []
        self.list_of_producers = []
        self.id_producer = -1
        self.id_cart = -1
        # A single, shared lock should be used here, not created in each method.
        # e.g., self.lock = Lock()

    def register_producer(self):
        """
        Registers a new producer and returns a unique producer ID.
        @warning Race Condition: The increment of `self.id_producer` is not
                 atomic or protected by a lock, so two concurrent calls could
                 receive the same ID. The lock created is local and useless.
        """
        logging.info("register_producer() called by Thread %s",
                     currentThread().getName())
        register_lock = Lock()
        producers = []
        self.id_producer += 1
        with register_lock:
            self.list_of_producers.append(producers)
        logging.info("Thread %s exited register_producer()",
                     currentThread().getName())
        return str(self.id_producer)

    def publish(self, producer_id, product):
        """
        Adds a product to a specific producer's stock list.
        @warning Race Condition: The check for available space and the subsequent
                 addition of products are not an atomic operation. A local lock
                 is created, which provides no actual protection.
        """
        logging.info("publish() called by Thread %s with producer_id %s to register product %s",
                     currentThread().getName(), str(producer_id), str(product))
        quantity, sleep_time = product[1], product[2]
        id_prod = int(producer_id)
        publish_lock = Lock()
        publish_check = False

        if len(self.list_of_producers[id_prod]) == self.queue_size_per_producer:
            logging.info("Thread %s with producer_id %s exited publish() with %s",
                         currentThread().getName(), str(producer_id), str(publish_check))
            return publish_check

        if len(self.list_of_producers[id_prod]) + quantity < self.queue_size_per_producer:
            with publish_lock:
                while quantity > 0:
                    time.sleep(sleep_time)
                    self.list_of_producers[id_prod].append(product[0])
                    quantity -= 1
        else:
            logging.info("Thread %s with producer_id %s exited publish() with %s",
                         currentThread().getName(), str(producer_id), str(publish_check))
            return publish_check

        publish_check = True
        logging.info("Thread %s with producer_id %s exited publish() with %s",
                     currentThread().getName(), str(producer_id), str(publish_check))
        return publish_check

    def new_cart(self):
        """
        Creates a new empty shopping cart and returns a unique cart ID.
        @warning Race Condition: Similar to `register_producer`, the increment
                 of `self.id_cart` is not protected by the local lock.
        """
        logging.info("new_cart() called by Thread %s",
                     currentThread().getName())
        cart = []
        new_cart_lock = Lock()
        with new_cart_lock:
            self.list_of_carts.append(cart)
            self.id_cart += 1
        return self.id_cart

    def add_to_cart(self, cart_id, product):
        """
        Moves a product from a producer's stock to a consumer's cart.
        @warning Race Condition: The check for product existence and the
                 subsequent removal/addition are not atomic. The local lock is
                 ineffective. This can lead to data corruption.
        """
        logging.info("add_to_cart() called by Thread %s for the cart %s to add product %s",
                     currentThread().getName(), str(cart_id), str(product))
        product_existence = False
        prod_list = []
        add_to_cart_lock = Lock()

        for prod_list_iter in self.list_of_producers:
            if product in prod_list_iter:
                product_existence = True
                prod_list = prod_list_iter
                break

        if product_existence is True:
            with add_to_cart_lock:
                self.list_of_carts[cart_id].append(product)
                prod_list.remove(product)

        logging.info("Thread %s exited add_to_cart() with %s",
                     currentThread().getName(), str(product_existence))
        return product_existence

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to a producer's stock.
        @warning Race Condition: Check-then-act is not atomic. The lock is local.
        @note Strange Logic: The product is always returned to the first
              producer (`list_of_producers[0]`), not its original producer.
        """
        logging.info("remove_from_cart() called by Thread %s for the cart %s to remove product %s",
                     currentThread().getName(), str(cart_id), str(product))
        remove_from_cart_lock = Lock()
        if product in self.list_of_carts[cart_id]:
            with remove_from_cart_lock:
                self.list_of_carts[cart_id].remove(product)
                self.list_of_producers[0].append(product)
        logging.info("Thread %s exited remove_from_cart()",
                     currentThread().getName())

    def place_order(self, cart_id):
        """
        Finalizes an order by returning the contents of the cart.
        @warning Not Thread-Safe: Returns a direct reference to the internal
                 cart list, which can be modified by other threads concurrently.
        """
        logging.info("place_order() called by Thread %s for the cart %s",
                     currentThread().getName(), str(cart_id))
        return_list = self.list_of_carts[cart_id]
        logging.info("Thread %s exited place_order()",
                     currentThread().getName())
        return return_list

class Producer(Thread):
    """A thread that continuously publishes products to the marketplace."""
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.id_producer = self.marketplace.register_producer()
        self.republish_wait_time = republish_wait_time

    def wait(self):
        """Sleeps for a configured time before retrying an action."""
        time.sleep(self.republish_wait_time)

    def run(self):
        """
        Continuously attempts to publish products, waiting and retrying if the
        marketplace queue for this producer is full.
        """
        while True:
            for product in self.products:
                while self.marketplace.publish(self.id_producer, product) is False:
                    self.wait()


class Consumer(Thread):
    """
    A thread that simulates a consumer shopping in the marketplace.
    It processes a predefined list of "add" and "remove" operations.
    """
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.id_cart = 0

    def wait(self):
        """Sleeps for a configured time before retrying an action."""
        time.sleep(self.retry_wait_time)

    def print_output(self):
        """Retrieves the final cart contents and prints them."""
        cart = self.marketplace.place_order(self.id_cart)
        for product in cart:
            print(self.name + ' bought ' + str(product))

    def run(self):
        """
        Executes the shopping simulation by creating new carts and performing
        add/remove operations as defined in its `self.carts` list. Uses a
        busy-wait polling loop to retry failed operations.
        """
        for cart in self.carts:
            self.id_cart = self.marketplace.new_cart()
            
            for operation in cart:
                quantity = operation['quantity']
                product = operation['product']
                
                if operation['type'] == "add":
                    while quantity > 0:
                        if self.marketplace.add_to_cart(self.id_cart, product) is False:
                            self.wait()
                        else:
                            quantity -= 1
                
                if operation['type'] == "remove":
                    while quantity > 0:
                        self.marketplace.remove_from_cart(self.id_cart, product)
                        quantity -= 1

            self.print_output()
