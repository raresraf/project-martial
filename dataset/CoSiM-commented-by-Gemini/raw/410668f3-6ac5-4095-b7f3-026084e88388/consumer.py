"""
This file models a multi-threaded producer-consumer simulation of a marketplace.

It appears to contain the contents of multiple files (`consumer.py`,
`marketplace.py`, `producer.py`, and `product.py`) concatenated together.
The code defines the core components of the simulation:
- Marketplace: The central shared resource where products are published and purchased.
- Consumer: A thread that simulates a user buying products.
- Producer: A thread that simulates a vendor publishing products.
- Product/Tea/Coffee: Data classes for the items being traded.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace.

    Each consumer runs in its own thread, processing a list of shopping carts.
    For each cart, it performs a series of 'add' and 'remove' operations
    before finally placing the order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of carts, where each cart is a list of
                          product operations.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying
                                     to add a product to the cart.
            **kwargs: Keyword arguments passed to the Thread constructor (e.g., name).
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.cart_id = -1
        self.marketplace = marketplace
        self.wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        """The main execution loop for the consumer."""
        # Process each shopping cart assigned to this consumer.
        for cart in self.carts:
            self.cart_id = self.marketplace.new_cart()

            # Execute the operations for the current cart (add/remove products).
            for cart_op in cart:
                op_type = cart_op["type"]
                quantity = cart_op["quantity"]
                prod = cart_op["product"]

                if op_type == "add":
                    # Attempt to add the specified quantity of a product.
                    while quantity > 0:
                        # Retry adding until successful.
                        while True:
                            ret = self.marketplace.add_to_cart(self.cart_id, prod)
                            if ret:
                                break
                            sleep(self.wait_time)
                        quantity -= 1
                else: # op_type == "remove"
                    # Remove the specified quantity of a product.
                    while quantity > 0:
                        self.marketplace.remove_from_cart(self.cart_id, prod)
                        quantity -= 1
            
            # Place the order and print the items "bought".
            lista = self.marketplace.place_order(self.cart_id)
            for cart_item in lista:
                print("{} bought {}".format(self.name, cart_item))

# The following classes appear to belong in separate files.

import logging
from logging import getLogger
from re import L
from threading import Lock
from logging.handlers import RotatingFileHandler

class Marketplace:
    """
    The central marketplace where producers and consumers interact.

    This class is intended to manage producer inventories and consumer shopping
    carts in a thread-safe manner.

    @warning The locking mechanism in this class is critically flawed. Each
             method creates a new, local `Lock()` object, which provides ZERO
             thread safety. Multiple threads can, and will, enter the critical
             sections simultaneously, leading to race conditions and data
             corruption. A single, shared instance lock (e.g., `self.lock = Lock()`)
             should be used across all methods.

    @note The ID generation for carts and producers is also not thread-safe and
          uses a questionable algorithm (`sum(list) + c`). Furthermore, the logic
          does not seem to remove products from a producer's inventory when an
          order is placed, effectively assuming infinite stock.
    """
    
    def __init__(self, queue_size_per_producer, ):
        """Initializes the marketplace."""
        self.queue_size_per_producer = queue_size_per_producer
        self.producatori = {} # {producer_id: {'produse': [...]}}
        self.cosuri = {} # {cart_id: {'produse_rezervate': [...]}}
        self.producatori_id = []
        self.cosuri_id = []
        self.prod_id = 1
        self.cos_id = 1
        self.logger = getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.handlers.RotatingFileHandler("marketplace.log"))

    def register_producer(self):
        """Intended to atomically register a new producer in the marketplace."""
        lock = Lock() # FLAW: This should be a shared instance lock.

        with lock:
            # FLAW: This ID generation is not thread-safe and incorrect.
            self.prod_id = sum(self.producatori_id)
            self.prod_id += 1
            self.producatori_id.append(self.prod_id)
            producator = {'produse': []}
            self.producatori[self.prod_id] = producator
            self.logger.info("Function  producator_id: %d",
                             "register_producer", self.prod_id)
            return self.prod_id

    def publish(self, producer_id, product):
        """
        Intended to allow a producer to publish a product to the marketplace.
        Returns True on success, False if the producer's queue is full.
        """
        lock = Lock() # FLAW: This should be a shared instance lock.
        with lock:
            for prod_id, produse_publicate in self.producatori.items():
                if prod_id == producer_id:
                    if len(produse_publicate['produse']) < self.queue_size_per_producer:
                        produse_publicate['produse'].append(product)
                        self.logger.info("Function %s called by:%d,product:%s ,return: TRUE\n",
                                         "publish", producer_id, product)
                        return True
            self.logger.info("Function  called by producer_id: %d, product:%s, return: FALSE\n",
                             "publish", producer_id, product)
            return False

    def new_cart(self):
        """Intended to create a new, unique shopping cart."""
        lock = Lock() # FLAW: This should be a shared instance lock.

        with lock:
            # FLAW: This ID generation is not thread-safe.
            self.cos_id = sum(self.cosuri_id)
            self.cos_id += 2
            self.cosuri_id.append(self.cos_id)
            cosuri = {'produse_rezervate': []}
            self.cosuri[self.cos_id] = cosuri
            return self.cos_id

    def add_to_cart(self, cart_id, product):
        """
        Intended to add a product to a cart if it's available from any producer.
        """
        lock = Lock() # FLAW: This should be a shared instance lock.

        with lock:
            for cos, continut in self.cosuri.items():
                if cos == cart_id:
                    # Check for product availability across all producers.
                    for producator, produse_publicate in self.producatori.items():
                        if product in produse_publicate['produse']:
                            continut['produse_rezervate'].append(product)
                            # NOTE: Product is NOT removed from producer's inventory.
                            self.logger.info("Function %s called by: %d, product:%s,return:TRUE\n",
                                             "add_to_cart", cart_id, product)
                            return True
        self.logger.info("Function  called by cart_id: %d, product:%s , return: FALSE\n",
                         "add_to_cart", cart_id, product)
        return False

    def remove_from_cart(self, cart_id, product):
        """Intended to remove a product from a specific cart."""
        lock = Lock() # FLAW: This should be a shared instance lock.
        with lock:
            for cos, continut in self.cosuri.items():
                if cos == cart_id:
                    continut['produse_rezervate'].remove(product)
        self.logger.info("Function %s called by cart_id:%d and product: %s\n", "remove_from-cart",
                         cart_id, product)

    def place_order(self, cart_id):
        """
        Intended to finalize an order, returning the products in the cart.
        NOTE: This does not clear the cart or fulfill the order from inventory.
        """
        lock = Lock() # FLAW: This should be a shared instance lock.
        with lock:
            for cos, continut in self.cosuri.items():
                if cos == cart_id:
                    self.logger.info("Function  cart_id: %d",
                                     "place_order", cart_id)
                    return continut['produse_rezervate']
            return None

# The line below indicates a file separation in the original source.
# >>>> file: producer.py

from threading import Thread
from time import sleep

class Producer(Thread):
    """
    Represents a producer that publishes products to the marketplace.
    
    Each producer runs in its own thread, iterating through its product list
    and publishing them according to the specified quantity and frequency.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of tuples, where each tuple contains
                             (product, quantity, wait_time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying a publish op.
            **kwargs: Keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.prod_id = self.marketplace.register_producer()
        self.republish_wait_time = republish_wait_time
        self.name = kwargs["name"]

    def run(self):
        """The main execution loop for the producer."""
        while True:
            for produs in self.products:
                product = produs[0]
                quantity = produs[1]
                waiting_time = produs[2]

                # Publish the specified quantity of the product.
                while quantity > 0:
                    # Retry publishing until successful.
                    while True:
                        ret = self.marketplace.publish(self.prod_id, product)
                        if ret:
                            break
                        sleep(self.republish_wait_time)
                    quantity -= 1
                    sleep(waiting_time)

# The lines below likely belong in a 'product.py' or similar module.
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
