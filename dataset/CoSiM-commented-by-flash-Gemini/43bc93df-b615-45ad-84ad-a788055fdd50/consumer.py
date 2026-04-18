
"""
@43bc93df-b615-45ad-84ad-a788055fdd50/consumer.py
@brief Synchronized Multi-threaded Marketplace Simulation.
This file implements a concurrent marketplace architecture where producers 
continuously supply products and consumers acquire them through shopping carts. 
It utilizes threading primitives and logging for state reconciliation in a 
multi-threaded environment.

Domain: Concurrent Programming, Synchronization Patterns.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Functional Utility: Represent a consumer agent that processes a set of shopping carts.
    Logic: For each cart, it sequentially adds or removes products in specified 
    quantities, retrying acquisitions if the marketplace is depleted.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Constructor: Binds the consumer to its carts and the shared marketplace.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def add_operation(self, quantity, cart_id, product):
        """
        Block Logic: Product acquisition loop with retry capability.
        Logic: Attempts to add 'quantity' of a product to the cart. If a request 
        fails (stock unavailable), it waits before retrying.
        """
        for _ in range(quantity):
            while not self.marketplace.add_to_cart(cart_id, product):
                time.sleep(self.retry_wait_time)

    def remove_operation(self, quantity, cart_id, product):
        """
        Block Logic: Product removal.
        Logic: Returns 'quantity' of a product from the cart back to the marketplace.
        """
        for _ in range(quantity):
            self.marketplace.remove_from_cart(cart_id, product)

    def run(self):
        """
        Execution Logic: Processes all assigned carts.
        Invariant: Each cart is processed atomically (all operations then order placement).
        """
        for cart in self.carts:
            id_cart = self.marketplace.new_cart()
            for operation in cart:
                if operation["type"] == "add":
                    self.add_operation(operation["quantity"], id_cart, operation["product"])
                else:
                    self.remove_operation(operation["quantity"], id_cart, operation["product"])

            self.marketplace.place_order(id_cart)


from threading import Lock, currentThread
import unittest
import logging
import time
from logging.handlers import RotatingFileHandler
from tema.product import Coffee, Tea

# Functional Utility: Configures a rotating log handler for marketplace events.
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

FORMATTER = logging.Formatter('$asctime : $levelname : $name : $message', style='$')
HANDLER = RotatingFileHandler('marketplace.log', maxBytes=20000, backupCount=5)
HANDLER.setFormatter(FORMATTER)

FORMATTER.converter = time.gmtime

LOGGER.addHandler(HANDLER)


class Marketplace:
    """
    Functional Utility: Thread-safe coordinator for product transactions.
    Logic: Tracks registered producers, available product stocks, and consumer 
    carts. Uses separate locks for consumer and producer operations to improve 
    concurrency.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Constructor: Initializes internal state and synchronization primitives.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.consumer_lock = Lock()
        self.producer_lock = Lock()
        self.products = {} 
        self.carts = {} 
        self.products_published = {} 
        self.producer_id = -1
        self.cart_id = -1

    def register_producer(self):
        """
        Functional Utility: Atomically registers a new producer and assigns an ID.
        """
        with self.producer_lock:
            LOGGER.info('[OLD]Last producer id:%d', self.producer_id)
            self.producer_id += 1
            self.products_published[self.producer_id] = 0
            self.products[self.producer_id] = {}
            LOGGER.info('[UPDATE]New producer id:%d', self.producer_id)
            return self.producer_id

    def publish(self, producer_id, product):
        """
        Functional Utility: Adds a product to a producer's stock.
        Logic: Enforces per-producer queue size limits and updates the stock mapping.
        """
        with self.producer_lock:
            LOGGER.info('[INPUT]Producer_id: %s and Product: %s', producer_id, product)

            res = False
            
            # Block Logic: Capacity check.
            if self.products_published[producer_id] <= self.queue_size_per_producer:
                self.products_published[producer_id] += 1
                if product in list(self.products[producer_id].keys()):
                    self.products[producer_id][product] += 1
                else:
                    self.products[producer_id][product] = 1
                res = True

            LOGGER.info('[OUTPUt]Method returns: %r', res)
            return res

    def new_cart(self):
        """
        Functional Utility: Initializes a new shopping cart for a consumer.
        """
        with self.consumer_lock:
            LOGGER.info('[OLD]:Cart_id: %d', self.cart_id)
            self.cart_id += 1
            self.carts[self.cart_id] = {}
            LOGGER.info('[UPDATE]:Cart_id: %d', self.cart_id)
            return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        Functional Utility: Transfers product ownership from marketplace to cart.
        Logic: Iterates through all producer stocks to find the requested product. 
        If found, decrements stock and increments cart quantity.
        """
        with self.consumer_lock:
            LOGGER.info('[INPUT]Cart_id: %d and Product: %s', cart_id, product)

            for producer in list(self.products.keys()):
                if product in list(self.products[producer].keys()):
                    self.products[producer][product] -= 1
                    if (product, producer) in list(self.carts[cart_id].keys()):
                        self.carts[cart_id][(product, producer)] += 1
                    else:
                        self.carts[cart_id][(product, producer)] = 1

                    if self.products[producer][product] == 0:
                        self.products[producer].pop(product, 0)
                    LOGGER.info('[OUTPUT]Method returns: True')
                    return True

        LOGGER.info('[OUTPUT]Method returns: False')
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Functional Utility: Reverses a product acquisition.
        Logic: Identifies the original producer of the product in the cart, 
        removes it, and restores it to the marketplace stock.
        """
        with self.consumer_lock:
            LOGGER.info('[INPUT]Cart_id: %d and Product: %s', cart_id, product)
            for (prod, producer_id) in list(self.carts[cart_id].keys()):
                if prod == product:
                    self.carts[cart_id][(product, producer_id)] -= 1
                    if self.carts[cart_id][(product, producer_id)] == 0:
                        self.carts[cart_id].pop((product, producer_id), 0)

                    if product in list(self.products[producer_id].keys()):
                        self.products[producer_id][product] += 1
                    else:
                        self.products[producer_id][product] = 1
                    break

    def place_order(self, cart_id):
        """
        Functional Utility: Finalizes the consumer order.
        Logic: Consolidates cart items, updates global published counts, 
        and prints confirmation.
        """
        with self.consumer_lock:
            LOGGER.info('[INPUT]Place order for cart_id: %d', cart_id)
            for (product, producer), quantity in self.carts[cart_id].items():
                for _ in range(quantity):
                    self.products_published[producer] -= 1
                    print(f"{currentThread().getName()} bought {product}")
            LOGGER.info('[OUTPUT]The cart was printed: %d', cart_id)



from threading import Thread
import time

class Producer(Thread):
    """
    Functional Utility: Represent a production agent that generates stock.
    Logic: Iterates through its product list, publishing items to the 
    marketplace and observing production times.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Constructor: Registers the producer and initializes its inventory.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()
        self.kwargs = kwargs

    def run(self):
        """
        Execution Logic: Infinite production loop.
        Logic: For each product, it attempts to publish. If successful, 
        it simulates production time. If rejected (full stock), it retries 
        after a wait period.
        """
        while True:
            for product, quantity, prod_time in self.products:
                for _ in range(quantity):
                    result = self.marketplace.publish(self.producer_id, product)

                    if result is True:
                        time.sleep(prod_time)
                    else:
                        # Block Logic: Congestion handling.
                        while not self.marketplace.publish(self.producer_id, product):
                            time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Functional Utility: Base immutable data carrier for marketplace items.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Functional Utility: Specialized product for tea types.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Functional Utility: Specialized product for coffee types.
    """
    acidity: str
    roast_level: str
