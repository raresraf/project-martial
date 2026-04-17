"""
@cbba5409-fb60-442a-b3cd-1ab1dd8d6a75/consumer.py
@brief Threaded marketplace simulation with centralized logging and inventory management.
* Algorithm: Concurrent producer-consumer model with shared state managed via thread-safe collections and fine-grained locking.
* Functional Utility: Facilitates a virtual market where producers generate goods and consumers manage carts, with detailed audit logging for all operations.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    @brief Consumer entity that performs shopping operations (add/remove) across multiple carts.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes the consumer with its target shopping lists and market connection.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        @brief Main execution loop for the consumer.
        Algorithm: Iterative processing of carts with a retry loop for failed inventory acquisitions.
        """
        for cart in self.carts:
            # Logic: Initializes a new transaction context.
            cart_id = self.marketplace.new_cart()

            for opperation in cart:
                if opperation["type"] == "add":
                    gotten_quantity = 0

                    # Logic: Busy-wait with backoff until the requested quantity is secured.
                    while gotten_quantity < opperation["quantity"]:
                        if self.marketplace.add_to_cart(cart_id, opperation["product"]):
                            gotten_quantity += 1
                        else:
                            # Functional Utility: Throttles retry attempts during stock-outs.
                            sleep(self.retry_wait_time)

                elif opperation["type"] == "remove":
                    # Logic: Returns items from cart back to the marketplace inventory.
                    for _ in range(opperation["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, opperation["product"])

            # Post-condition: Completes the transaction and releases output artifacts.
            self.marketplace.place_order(cart_id)


from threading import Lock
from threading import currentThread

import logging
from logging.handlers import RotatingFileHandler
import time

# Block Logic: System-wide audit logging configuration.
# Strategy: Uses a rotating file handler to cap log size and preserve historical execution data.
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('marketplace.log', maxBytes=1000000, backupCount=3)
formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(funcName)s: %(message)s', \
    '%Y-%m-%d %H:%M:%S')
formatter.converter = time.gmtime
handler.setFormatter(formatter)
logger.addHandler(handler)


class Marketplace:
    """
    @brief Centralized inventory controller with multi-lock synchronization for high-throughput operations.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the marketplace and its various sub-locks to maximize concurrency.
        """
        logging.info("enter: %s", queue_size_per_producer)
        self.queue_size_per_producer = queue_size_per_producer

        self.unused_producer_id = 0
        self.products = [] # Intent: Stores active inventory partitioned by producer ID.
        self.register_lock = Lock() # Intent: Serializes producer onboarding.

        self.unused_cart_id = 0
        self.carts = [] # Intent: Registry of active consumer transactions.
        self.cart_lock = Lock() # Intent: Serializes cart creation.

        self.inventory = {} # Intent: Reverse-lookup map from product to a stack of producer IDs.
        self.products_lock = Lock() # Intent: Protects atomic inventory transitions (add/remove).

        self.consumer_print_lock = Lock() # Intent: Synchronizes shared output streams for purchase confirmation.

        logging.info("exit")

    def register_producer(self):
        """
        @brief Onboards a new producer and initializes its inventory buffer.
        """
        logging.info("enter")

        with self.register_lock:
            new_id = self.unused_producer_id
            self.unused_producer_id += 1

        self.products.append([])

        logging.info("exit: %s", new_id)
        return new_id

    def publish(self, producer_id, product):
        """
        @brief Adds a product to the market if the producer's quota is not exceeded.
        Algorithm: Direct append to producer list and update of the global reverse-lookup map.
        """
        logging.info("enter: %s, %s", producer_id, product)

        if len(self.products[producer_id]) < self.queue_size_per_producer:
            self.products[producer_id].append(product)

            # Logic: Maintains availability registry for fast consumer lookups.
            self.inventory.setdefault(product, [])
            self.inventory[product].append(producer_id)

            logging.info("exit: True")
            return True

        logging.info("exit: False")
        return False

    def new_cart(self):
        """
        @brief Allocates a new transaction identifier.
        """
        logging.info("enter")

        with self.cart_lock:
            new_id = self.unused_cart_id
            self.unused_cart_id += 1

        self.carts.append([])

        logging.info("exit: %s", new_id)
        return new_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Transfers an item from general inventory to a specific consumer cart.
        Invariant: Uses products_lock to ensure atomic ownership transition from producer to consumer.
        """
        logging.info("enter: %s, %s", cart_id, product)

        with self.products_lock:
            if product in self.inventory:
                if len(self.inventory[product]) > 0:
                    # Logic: Pops a producer ID that currently holds the product.
                    producer_id = self.inventory[product].pop()
                    
                    self.carts[cart_id].append((product, producer_id))
                    self.products[producer_id].remove(product)

                    logging.info("exit: True")
                    return True

        logging.info("exit: False")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Returns an item from a cart back to its original producer's inventory.
        """
        logging.info("enter: %s, %s", cart_id, product)

        # Logic: Linear search within the cart for the specific product unit.
        cart_list = [tup for tup in self.carts[cart_id] if product in tup]

        if len(cart_list) > 0:
            (_, producer_id) = cart_list[0]
            
            self.carts[cart_id].remove(cart_list[0])
            
            # Post-condition: Restores item to availability registry and producer buffer.
            self.inventory[product].append(producer_id)
            self.products[producer_id].append(product)

        logging.info("exit")

    def place_order(self, cart_id):
        """
        @brief Finalizes the shopping phase and outputs purchased items.
        """
        logging.info("enter: %s", cart_id)

        with self.consumer_print_lock:
            # Logic: Extracts products from tuples and prints acquisition log.
            for item in [product for (product, _) in self.carts[cart_id]]:
                print(f"{currentThread().getName()} bought {item}")

        logging.info("exit")


from threading import Thread
from time import sleep

class Producer(Thread):
    """
    @brief Producer agent that generates goods for the marketplace based on fixed quotas.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes the producer and registers it with the marketplace.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        self.producer_id = marketplace.register_producer()

    def run(self):
        """
        @brief Main production lifecycle.
        Algorithm: Iterative batch generation with individual item publication and backoff.
        """
        while True:
            for product in self.products:
                published_quantity = 0

                # Logic: Attempts to publish requested quantity for the current product type.
                while published_quantity < product[1]:
                    if self.marketplace.publish(self.producer_id, product[0]):
                        published_quantity += 1
                        # Domain: Intra-batch delay.
                        sleep(self.republish_wait_time)
                    else:
                        # Logic: Quota full; waits for market consumption before retrying.
                        sleep(product[2])


from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base schema for immutable marketplace products.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Specialized product type for tea.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Specialized product type for coffee.
    """
    acidity: str
    roast_level: str
