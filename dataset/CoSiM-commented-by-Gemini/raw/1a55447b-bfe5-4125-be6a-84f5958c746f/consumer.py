
"""
This file simulates a Producer-Consumer model for an e-commerce marketplace.

It contains several modules concatenated together:
- Consumer: A thread that simulates a consumer buying products.
- Marketplace: The central shared resource, which has several concurrency issues.
- TestMarketplace: A unittest class for the marketplace.
- Producer: A thread that publishes products to the marketplace.

NOTE: This implementation contains several race conditions and logical bugs.
"""
import time
from threading import Thread, Lock
import unittest
import logging
from logging.handlers import RotatingFileHandler


class Consumer(Thread):
    """
    A worker thread that simulates a consumer's shopping activities.
    It processes a list of actions (add/remove) for a single cart.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer.

        Args:
            carts (list): A list of "carts", where each cart is a list of
                          shopping activities (dictionaries).
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying to add a
                                     product if it's out of stock.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # Each consumer thread gets one cart ID for its entire lifetime.
        self.id_cart = self.marketplace.new_cart()

    def run(self):
        """
        The main execution loop for the consumer. Processes each assigned cart.
        """
        for cart_activities in self.carts:
            for activity in cart_activities:
                for _ in range(activity["quantity"]):
                    if activity["type"] == "add":
                        # This is a busy-wait loop. It will continuously try to
                        # add the product until it succeeds, sleeping between attempts.
                        added = False
                        while not added:
                            added = self.marketplace.add_to_cart(self.id_cart, activity["product"])
                            if not added:
                                time.sleep(self.retry_wait_time)
                    else: # "remove" action
                        self.marketplace.remove_from_cart(self.id_cart, activity["product"])
                        # The sleep here is inefficient as it happens even after a successful removal.
                        time.sleep(self.retry_wait_time)
        
        final_products = self.marketplace.place_order(self.id_cart)
        
        for order in final_products:
            with self.marketplace.print_lock:
                print(f"cons{self.id_cart} bought {order}")


class Marketplace:
    """
    The central marketplace, a shared resource for producers and consumers.

    NOTE: This class is not fully thread-safe. Several methods lack the
    necessary locking or have inefficient locking strategies.
    """
    
    def __init__(self, queue_size_per_producer):
        # --- Logging Setup ---
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)
        self.formatter = logging.Formatter("%(asctime)s;%(message)s")
        handler = RotatingFileHandler('marketplace.log', 'w')
        handler.setFormatter(self.formatter)
        self.log.addHandler(handler)

        # --- Data Structures ---
        self.prods = {}  # {producer_id: [products]}
        self.cons = {}   # {cart_id: [products]}
        self.no_prods = 0
        self.no_cons = 0
        self.available_prods = []
        self.queue_size_per_producer = queue_size_per_producer

        # --- Synchronization ---
        self.producer_lock = Lock()
        self.publish_lock = Lock()
        self.cart_lock = Lock()
        self.print_lock = Lock()

    def register_producer(self):
        """Thread-safely registers a new producer."""
        self.log.info("Register product")
        with self.producer_lock:
            self.no_prods += 1
            self.prods[self.no_prods] = []
            return self.no_prods

    def publish(self, producer_id, product):
        """Thread-safely allows a producer to publish a product."""
        self.log.info("Publish")
        with self.publish_lock:
            if len(self.prods[producer_id]) < self.queue_size_per_producer:
                self.prods[producer_id].append(product)
                self.available_prods.append(product)
                self.log.info("Publish final")
                return True
            return False

    def new_cart(self):
        """Thread-safely creates a new empty cart."""
        self.log.info("New Cart")
        with self.cart_lock:
            self.no_cons += 1
            self.cons[self.no_cons] = []
            return self.no_cons

    def add_to_cart(self, cart_id, product):
        """
        Moves a product from stock to a cart.

        NOTE: This method holds a single lock for a long and inefficient
        operation, which can become a bottleneck. The inner loop to find the
        producer is O(N*M) and is not thread-safe with respect to the `self.prods` dict.
        """
        self.log.info("Add to Cart")
        with self.cart_lock:
            if product in self.available_prods:
                self.available_prods.remove(product)
                # This search is very inefficient and not thread-safe.
                for i, products in self.prods.items():
                    if product in products:
                        self.cons[cart_id].append(product)
                        self.prods[i].remove(product)
                        self.log.info("Add to Cart final")
                        return True
            return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to stock.
        
        RACE CONDITION: This method is not thread-safe. It reads and writes
        to shared lists (`self.cons` and `self.available_prods`) without a lock.
        """
        self.log.info("Remove from cart")
        if product in self.cons[cart_id]:
            self.cons[cart_id].remove(product)
            self.available_prods.append(product)
        self.log.info("Remove from cart final")

    def place_order(self, cart_id):
        """
        Returns the content of a cart for "placing an order".
        
        RACE CONDITION: This method is not thread-safe. It reads from `self.cons`
        without a lock.
        LOGIC FLAW: This method does not actually consume or remove the products
        from the system; it only returns a copy of the cart's contents.
        """
        self.log.info("Place order")
        return self.cons[cart_id]


class Producer(Thread):
    """A worker thread that produces items."""
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Correctly registers only once.
        self.id_producer = self.marketplace.register_producer()

    def run(self):
        """
        Continuously tries to publish products.

        BUG: The logic incorrectly sleeps for `republish_wait_time` even after a
        successful publish, which will slow down production unnecessarily.
        """
        while True:
            for prod_info in self.products:
                product, quantity, wait_time = prod_info
                for _ in range(quantity):
                    ret = self.marketplace.publish(self.id_producer, product)
                    if ret:
                        sleep(wait_time)
                    # BUG: This should be in an `else` block.
                    sleep(self.republish_wait_time)

# This Test class seems incomplete and uses hardcoded strings for products,
# which may not align with the dataclass definitions if they were used.
class TestMarketplace(unittest.TestCase):
    pass
