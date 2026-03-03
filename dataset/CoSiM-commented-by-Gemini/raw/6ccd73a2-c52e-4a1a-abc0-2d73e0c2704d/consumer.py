"""
This module implements a producer-consumer simulation for a marketplace.

This version is characterized by an attempt at fine-grained locking, using
separate locks for different operations within the Marketplace. However, the
implementation is deeply flawed, containing numerous race conditions and unsafe
state modifications. The data model is also different, with each producer
maintaining a list of their published products.

WARNING: This code is not thread-safe and is provided as an example of
incorrectly implemented concurrency controls.
"""
import time
from threading import Thread, Lock, currentThread


class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace.
    It processes a list of shopping operations and places an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        Main loop for the consumer. Processes each cart in its list.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for product_op in cart: # The item is here named 'product' but is an operation dict
                i = 0
                while i < product_op["quantity"]:
                    # Logic to handle 'add' or 'remove' operations.
                    if product_op["type"] == "remove":
                        res = self.marketplace.remove_from_cart(cart_id, product_op["product"])
                        if res == 1:
                            i += 1
                        else:
                            # Retry on failure.
                            time.sleep(self.retry_wait_time)
                    else:
                        res = self.marketplace.add_to_cart(cart_id, product_op["product"])
                        if res:
                            i += 1
                        else:
                            # Retry on failure.
                            time.sleep(self.retry_wait_time)

            self.marketplace.place_order(cart_id)

class Marketplace:
    """
    The central marketplace, managing producers, products, and carts.

    This implementation uses multiple locks in an attempt at fine-grained
    synchronization, but the logic is flawed with many race conditions.
    """
    def __init__(self, queue_size_per_producer):
        self.max_queue_size = queue_size_per_producer
        # Maps producer_id to a list of products they have published.
        self.producer_dictionary = {}
        self.current_producer_id = -1
        self.all_carts = {}
        # An attempt at fine-grained locking.
        self.add_lock = Lock()
        self.remove_lock = Lock()
        self.carts_lock = Lock()
        self.register_lock = Lock()

    def register_producer(self):
        """
        Registers a new producer.
        WARNING: This method is not thread-safe. It modifies the shared
        `producer_dictionary` after releasing the lock, creating a race condition.
        """
        with self.register_lock:
            self.current_producer_id += 1
        # This dictionary modification is unsynchronized.
        self.producer_dictionary[self.current_producer_id] = []
        return self.current_producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product for a given producer.
        WARNING: This method is not thread-safe. It reads and writes to the
        shared `producer_dictionary` without any locks.
        """
        p_id = int(producer_id)
        if len(self.producer_dictionary[p_id]) >= self.max_queue_size:
            return False
        self.producer_dictionary[p_id].append(product)
        return True

    def new_cart(self):
        """
        Creates a new cart.
        WARNING: This method is not thread-safe. It reads `len(self.all_carts)`
        outside the lock, which can lead to duplicate cart_ids.
        """
        with self.carts_lock:
            cart_id = len(self.all_carts) + 1
        self.all_carts[cart_id] = []
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart by searching all producers' stock.
        WARNING: This method is not thread-safe. It iterates over a shared
        dictionary without proper synchronization.
        """
        with self.add_lock:
            ok_add = False
            # The iteration over `producer_dictionary.items()` is not safe
            # if other threads are modifying it via `publish` or `register_producer`.
            for _, queue in self.producer_dictionary.items():
                if product in queue:
                    queue.remove(product)
                    self.all_carts[cart_id].append(product)
                    ok_add = True
                    break
        return ok_add

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart, returning it to any producer with space.
        WARNING: This method is not thread-safe and its logic is convoluted.
        """
        ok_remove = False
        # The iteration and modification of `producer_dictionary` are unsynchronized.
        for _, queue in self.producer_dictionary.items():
            if len(queue) < self.max_queue_size:
                queue.append(product)
                ok_remove = True
                break
        if ok_remove:
            # The lock only protects the removal from `all_carts`.
            with self.remove_lock:
                self.all_carts[cart_id].remove(product)
        return ok_remove

    def place_order(self, cart_id):
        """Prints the items in a cart to simulate an order."""
        for prod in self.all_carts[cart_id]:
            print(f"{currentThread().getName()} bought {prod}")

class Producer(Thread):
    """Represents a producer thread."""
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.product_id = self.marketplace.register_producer()

    def run(self):
        """
        Main loop for the producer. Publishes products to the marketplace.
        """
        while True:
            for elem in self.products:
                curr_prod = 0
                while curr_prod < elem[1]:
                    publish_ok = self.marketplace.publish(str(self.product_id), elem[0])
                    if publish_ok:
                        time.sleep(elem[2])
                        curr_prod += 1
                    else:
                        time.sleep(self.republish_wait_time)

from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a product."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass for Tea."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass for Coffee."""
    acidity: str
    roast_level: str
