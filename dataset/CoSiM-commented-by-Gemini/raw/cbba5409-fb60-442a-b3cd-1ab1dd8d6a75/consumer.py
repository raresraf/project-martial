"""
This module simulates a marketplace with producers, consumers, and products.

It uses a multi-threaded approach to simulate the concurrent actions of producers
creating and publishing products and consumers buying them. The Marketplace class
acts as a central hub, managing inventory and transactions. The script also
configures detailed logging to a file.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """Represents a consumer that purchases products from the marketplace."""

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a Consumer thread."""
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """The main loop for the consumer thread.
        
        Processes each cart by adding and removing products, then places the order.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for opperation in cart:
                if opperation["type"] == "add":
                    gotten_quantity = 0
                    # Continuously try to add the product until the desired quantity is met.
                    while gotten_quantity < opperation["quantity"]:
                        if self.marketplace.add_to_cart(cart_id, opperation["product"]):
                            gotten_quantity += 1
                        else:
                            # Wait before retrying if the product is not available.
                            sleep(self.retry_wait_time)

                elif opperation["type"] == "remove":
                    for _ in range(opperation["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, opperation["product"])

            self.marketplace.place_order(cart_id)


from threading import Lock
from threading import currentThread

import logging
from logging.handlers import RotatingFileHandler
import time

# --- Logging Configuration ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('marketplace.log', maxBytes=1000000, backupCount=3)
formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(funcName)s: %(message)s', \
    '%Y-%m-%d %H:%M:%S')
formatter.converter = time.gmtime
handler.setFormatter(formatter)
logger.addHandler(handler)
# --- End Logging Configuration ---

class Marketplace:
    """A central marketplace for producers to sell and consumers to buy products.
    
    This class manages product inventory, producer and consumer registration, and
    all buying and selling transactions in a thread-safe manner.
    """

    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace."""
        logging.info("enter: %s", queue_size_per_producer)
        self.queue_size_per_producer = queue_size_per_producer
        self.unused_producer_id = 0
        self.products = []  # List of lists, storing products per producer.
        self.register_lock = Lock()
        self.unused_cart_id = 0
        self.carts = []  # List of lists, storing products per cart.
        self.cart_lock = Lock()
        # A central inventory mapping a product to a list of producer IDs who have it.
        self.inventory = {}
        self.products_lock = Lock()
        self.consumer_print_lock = Lock()
        logging.info("exit")

    def register_producer(self):
        """Registers a new producer and returns a unique producer ID."""
        logging.info("enter")
        with self.register_lock:
            new_id = self.unused_producer_id
            self.unused_producer_id += 1
        self.products.append([])
        logging.info("exit: %s", new_id)
        return new_id

    def publish(self, producer_id, product):
        """Allows a producer to publish a product to the marketplace."""
        logging.info("enter: %s, %s", producer_id, product)
        if len(self.products[producer_id]) < self.queue_size_per_producer:
            self.products[producer_id].append(product)
            self.inventory.setdefault(product, [])
            self.inventory[product].append(producer_id)
            logging.info("exit: True")
            return True
        logging.info("exit: False")
        return False

    def new_cart(self):
        """Creates a new cart for a consumer and returns a unique cart ID."""
        logging.info("enter")
        with self.cart_lock:
            new_id = self.unused_cart_id
            self.unused_cart_id += 1
        self.carts.append([])
        logging.info("exit: %s", new_id)
        return new_id

    def add_to_cart(self, cart_id, product):
        """Adds a product to a consumer's cart.
        
        It finds a producer with the product in stock, moves the product from the
        producer's stock to the consumer's cart, and updates the central inventory.
        """
        logging.info("enter: %s, %s", cart_id, product)
        with self.products_lock:
            if product in self.inventory and len(self.inventory[product]) > 0:
                producer_id = self.inventory[product].pop()
                self.carts[cart_id].append((product, producer_id))
                self.products[producer_id].remove(product)
                logging.info("exit: True")
                return True
        logging.info("exit: False")
        return False

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a consumer's cart and returns it to the producer."""
        logging.info("enter: %s, %s", cart_id, product)
        cart_list = [tup for tup in self.carts[cart_id] if product in tup]
        if len(cart_list) > 0:
            (_, producer_id) = cart_list[0]
            self.carts[cart_id].remove(cart_list[0])
            self.inventory[product].append(producer_id)
            self.products[producer_id].append(product)
        logging.info("exit")

    def place_order(self, cart_id):
        """Finalizes an order and prints the items bought by the consumer."""
        logging.info("enter: %s", cart_id)
        with self.consumer_print_lock:
            for item in [product for (product, _) in self.carts[cart_id]]:
                print(f"{currentThread().getName()} bought {item}")
        logging.info("exit")


from threading import Thread
from time import sleep

class Producer(Thread):
    """Represents a producer that creates and publishes products."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes a Producer thread."""
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = marketplace.register_producer()

    def run(self):
        """The main loop for the producer thread.
        
        Continuously produces items and tries to publish them to the marketplace.
        Note: The sleep logic seems flawed. It sleeps `republish_wait_time` after a
        successful publish and the `production_time` after a failed publish, which
        is likely the reverse of the intended logic.
        """
        while True:
            for product in self.products:
                published_quantity = 0
                while published_quantity < product[1]:
                    if self.marketplace.publish(self.producer_id, product[0]):
                        published_quantity += 1
                        # This sleep likely belongs before the publish attempt.
                        sleep(self.republish_wait_time)
                    else:
                        # This should likely be a short retry-wait, not the production time.
                        sleep(product[2])


from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A generic product with a name and a price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A specialized product representing Tea."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A specialized product representing Coffee."""
    acidity: str
    roast_level: str
