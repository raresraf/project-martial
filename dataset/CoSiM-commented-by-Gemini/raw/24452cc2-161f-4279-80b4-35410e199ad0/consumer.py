"""
A multi-threaded producer-consumer simulation of an e-commerce marketplace.

This implementation features a complex Marketplace with fine-grained locking using
multiple Semaphores, per-producer inventory management, and event logging to a file.
The 'add_to_cart' mechanism functions as a reservation system, marking items as
unavailable rather than removing them from the producer's stock.
"""
from threading import Thread, Semaphore
from time import sleep
import logging
from logging.handlers import RotatingFileHandler
import time

class Consumer(Thread):
    """
    Represents a consumer thread that simulates purchasing items.

    The consumer processes a predefined list of shopping actions (add/remove) for
    a series of carts.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of 'carts', where each cart is a list of commands.
            marketplace (Marketplace): The shared marketplace object.
            retry_wait_time (float): Seconds to wait before retrying a failed action.
            **kwargs: Catches the 'name' of the thread.
        """
        super().__init__()
        self.name = kwargs["name"]
        self.retry_wait_time = retry_wait_time
        self.id_cart = -1
        self.carts = carts
        self.marketplace = marketplace

    def run(self):
        """The main execution loop for the consumer."""
        for cart in self.carts:
            # Each cart from the list gets a new ID from the marketplace.
            self.id_cart = self.marketplace.new_cart()
            for command in cart:
                command_type = command["type"]
                product = command["product"]
                quantity = command["quantity"]

                # Pre-condition: Determine if the operation is 'add' or 'remove'.
                if command_type == "add":
                    for _ in range(quantity):
                        add_result = self.marketplace.add_to_cart(self.id_cart, product)
                        # Invariant: If adding to cart fails (product unavailable), wait and retry.
                        while True:
                            if not add_result:
                                sleep(self.retry_wait_time)
                                add_result = self.marketplace.add_to_cart(self.id_cart, product)
                            else:
                                break
                elif command_type == "remove":
                    for _ in range(quantity):
                        remove_result = self.marketplace.remove_from_cart(self.id_cart, product)
                        if not remove_result:
                            print("INVALID REMOVE RESULT; EXITING")
                            return
                else:
                    print("INVALID OPERATION; EXITING")
                    return
            
            cart_list = self.marketplace.place_order(self.id_cart)
            
            # Use a semaphore to ensure print statements are not interleaved.
            with self.marketplace.print_semaphore:
                for item in cart_list:
                    if item is not None:
                        print(f"{self.name} bought {item}")


class Marketplace:
    """
    A thread-safe marketplace using fine-grained locking and logging.

    Manages producer inventories and consumer carts with separate semaphores
    for different resources to allow for more parallelism.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace and sets up logging.

        Args:
            queue_size_per_producer (int): The maximum number of items each
                                           producer can have in their inventory.
        """
        # Setup for logging all marketplace operations.
        self.logger = logging.getLogger('my_logger')
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(levelname)8s: %(message)s')
        formatter.converter = time.gmtime
        
        handler = RotatingFileHandler('marketplace.log', maxBytes=1000000, backupCount=5)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.queues = {} # Stores producer inventories and their locks.
        self.capacity = queue_size_per_producer
        self.id_producer = -1
        self.id_cart = -1
        # Semaphores for controlling access to shared resources.
        self.print_semaphore = Semaphore(1)
        self.carts_semaphore = Semaphore(1)
        self.general_semaphore = Semaphore(1)
        self.carts = {}
        
        # The following log entry appears to be syntactically incorrect.
        # self.logger.info(, queue_size_per_producer)

    def register_producer(self):
        """
        Registers a new producer, creating an inventory entry for them.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        # self.logger.info()
        with self.general_semaphore:
            self.id_producer += 1
            # Each producer gets a dictionary with a product list and a dedicated semaphore.
            self.queues[self.id_producer] = {}
            self.queues[self.id_producer]["products"] = []
            self.queues[self.id_producer]["semaphore"] = Semaphore(1)
        
        # self.logger.info(, self.id_producer)
        return self.id_producer

    def publish(self, producer_id, product):
        """
        Adds a product to a producer's inventory.

        Returns:
            bool: True if the product was added, False if the inventory is full.
        """
        # self.logger.info(, producer_id, product)

        # Acquire the specific semaphore for this producer's inventory.
        self.queues[producer_id]["semaphore"].acquire()
        if len(self.queues[producer_id]["products"]) < self.capacity:
            # Adds the product as a tuple, with a boolean indicating availability.
            self.queues[producer_id]["products"].append((product, True))
            self.queues[producer_id]["semaphore"].release()
            
            # self.logger.info()
            return True
        self.queues[producer_id]["semaphore"].release()
        
        # self.logger.info()
        return False

    def new_cart(self):
        """
        Creates a new empty cart for a consumer.

        Returns:
            int: The unique ID for the new cart.
        """
        # self.logger.info()
        with self.carts_semaphore:
            self.id_cart += 1
            self.carts[self.id_cart] = []
            # self.logger.info(, self.id_cart)
        return self.id_cart

    def add_to_cart(self, cart_id, product):
        """
        Reserves a product for a cart by marking it as unavailable in the producer's inventory.

        Returns:
            bool: True if the product was found and reserved, False otherwise.
        """
        # self.logger.info(, cart_id, product)
        for id_producer, queue_producer in self.queues.items():
            queue_producer["semaphore"].acquire()
            for idx, queue_item in enumerate(queue_producer["products"]):
                # Logic: Finds a product that is marked as available.
                if product == queue_item[0] and queue_item[1] is True:
                    self.carts[cart_id].append((id_producer, product))
                    # Mark the product as unavailable (reserved) instead of removing it.
                    queue_producer["products"][idx] = (queue_item[0], False)
                    queue_producer["semaphore"].release()
                    return True # NOTE: The original corrupted log messages are omitted.
            queue_producer["semaphore"].release()
        return False

"""
NOTE: The remainder of this file appears to contain corrupted or extraneous text
fragments from other files or docstrings. They are not part of the executable
logic of the 'Consumer' and 'Marketplace' classes defined above.
"""
# --- Start of Corrupted/Extraneous Text ---
#         Removes a product from cart.
#
#         :type cart_id: Int
#         :param cart_id: id cart
#
#         :type product: Product
#         :param product: the product to remove from cart
#         Entering remove_from_cart function with parameters \
# cart_id = %s, product = %sReturning from add_to_cart function with value TrueReturning from add_to_cart function with value False
#         Return a list with all the products in the cart.
#         And removes those products from their producers' queue
#
#         :type cart_id: Int
#         :param cart_id: id cart
#         Entering place_order function with parameters \
# cart_id = %sReturning from place_order function with result = %s
# This module represents the Producer.
#
# Computer Systems Architecture Course
# Assignment 1
# March-April 2022
# IONITA Dragos 341 C1
#
#     Class that represents a producer.
#     
#         Constructor.
#
#         @type products: List()
#         @param products: a list of products that the producer will produce
#
#         @type marketplace: Marketplace
#         @param marketplace: a reference to the marketplace
#
#         @type republish_wait_time: Time
#         @param republish_wait_time: the number of seconds that a producer must
#         wait until the marketplace becomes available
#
#         @type kwargs:
#
#
#         @param kwargs: other arguments that are passed to the Thread's __init__()
#         
# This module offers the available Products.
#
# Computer Systems Architecture Course
# Assignment 1
# March-April 2022
# IONITA Dragos 341 C1
#
#     Class that represents a product.
#     
#     Tea products
#     
#     Coffee products
#     """
#     acidity: str
#     roast_level: str
# --- End of Corrupted/Extraneous Text ---