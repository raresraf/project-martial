"""
This module simulates an e-commerce marketplace with Producers and Consumers.

This implementation uses a central `Marketplace` with a single global pool of
products and coarse-grained locks for adding and removing items. This design
has several flaws, including a non-thread-safe `publish` method and locking
that creates unnecessary bottlenecks, serializing all 'add' and 'remove'
operations.
"""

from threading import Thread, Lock
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer that buys products from the marketplace.
    """
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.marketplace = marketplace
        self.carts = carts
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        """
        Processes each assigned cart by executing its 'add' or 'remove' actions.
        """
        for cart in self.carts:
            new_cart = self.marketplace.new_cart()

            for prod in cart:
                i = 0
                while i < prod["quantity"]:
                    if prod["type"] == "add":
                        # If adding to cart fails (product is unavailable), wait and retry.
                        if self.marketplace.add_to_cart(new_cart, prod["product"]):
                            i += 1
                        else:
                            sleep(self.retry_wait_time)
                    elif prod["type"] == "remove":
                        self.marketplace.remove_from_cart(new_cart, prod["product"])
                        i += 1
            
            order = self.marketplace.place_order(new_cart)
            for product in order:
                print(self.name, "bought", product)


class Marketplace:
    """
    The central marketplace. This implementation is flawed and not fully thread-safe.

    It uses a single global list for all products and coarse locks for add/remove
    operations, which unnecessarily serializes consumer actions and creates a
    performance bottleneck. The `publish` method is also not thread-safe.
    """
    def __init__(self, queue_size_per_producer):
        self.q_max_size_per_producer = queue_size_per_producer
        self.register_lock = Lock()
        self.new_cart_lock = Lock()
        
        # Coarse-grained lock for all 'add to cart' operations.
        self.add_lock = Lock()
        # Coarse-grained lock for all 'remove from cart' operations.
        self.remove_lock = Lock()

        self.no_producers = 0
        self.no_carts = 0
        
        # Tracks the number of items each producer has on the market.
        self.no_products = {}
        # Flawed: Maps a product to a single producer. If two producers publish
        # the same product, the last one wins.
        self.products = {}
        # A single, global list of all available products.
        self.market_products = []
        self.carts = {}

    def register_producer(self):
        """Thread-safely registers a new producer and returns a unique ID."""
        with self.register_lock:
            producer_id = self.no_producers
            self.no_producers += 1
            self.no_products[producer_id] = 0
        return producer_id

    def publish(self, producer_id, product):
        """
        Adds a product to the global market list.
        
        NOTE: This method is NOT thread-safe. Concurrent calls can lead to race
        conditions when modifying `self.products` and `self.market_products`.
        """
        if self.no_products[producer_id] >= self.q_max_size_per_producer:
            return False
        self.products[product] = producer_id
        self.no_products[producer_id] += 1
        self.market_products.append(product)
        return True

    def new_cart(self):
        """Thread-safely creates a new cart and returns its ID."""
        self.new_cart_lock.acquire()
        cart_id = self.no_carts
        self.no_carts += 1
        self.carts[cart_id] = []
        self.new_cart_lock.release()
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart. This method uses a global lock, serializing
        all 'add' operations across all consumers.
        """
        self.add_lock.acquire()
        if product not in self.market_products:
            self.add_lock.release()
            return False
        if cart_id not in self.carts.keys():
            self.add_lock.release()
            return False
        self.market_products.remove(product)
        self.no_products[self.products[product]] -= 1
        self.carts[cart_id].append(product)
        self.add_lock.release()
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart. This method uses a global lock,
        serializing all 'remove' operations.
        """
        self.remove_lock.acquire()
        if product not in self.carts[cart_id]:
            self.remove_lock.release()
            return False
        self.market_products.append(product)
        self.no_products[self.products[product]] += 1
        self.carts[cart_id].remove(product)
        self.remove_lock.release()
        return True

    def place_order(self, cart_id):
        """Finalizes the order by returning the list of items in the cart."""
        if cart_id not in self.carts.keys():
            return None
        return self.carts[cart_id]


class Producer(Thread):
    """
    Represents a producer that publishes products to the marketplace.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.daemon = kwargs['daemon']
        self.name = kwargs['name']
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """Continuously publishes products to the marketplace."""
        while True:
            for product in self.products:
                i = 0
                while i < product[1]:
                    # If publishing fails (e.g., queue is full), wait and retry.
                    if self.marketplace.publish(self.producer_id, product[0]):
                        time.sleep(product[2])
                        i += 1
                    else:
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base data class for a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A data class for a Tea product, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A data class for a Coffee product, inheriting from Product."""
    acidity: str
    roast_level: str
