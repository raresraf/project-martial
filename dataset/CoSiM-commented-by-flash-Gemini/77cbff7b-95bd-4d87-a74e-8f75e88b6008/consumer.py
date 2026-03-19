"""
This module implements a multi-threaded e-commerce simulation, defining
`Consumer` and `Producer` roles that interact with a `Marketplace`.
It includes classes for managing product listings (`PublishedProduct`, `ProductsList`)
and shopping carts (`Cart`), demonstrating concurrent operations with
thread-safe mechanisms like locks and queues.
"""

import sys
import logging
import time
from threading import Thread, Lock, currentThread
from logging.handlers import RotatingFileHandler


class Consumer(Thread):
    """
    Represents a consumer in the e-commerce simulation.

    Each consumer operates as a separate thread, managing a list of shopping carts.
    It attempts to add/remove products to/from its carts via the marketplace
    and retries failed operations after a specified wait time.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a new Consumer instance.

        Args:
            carts (list): A list of shopping lists, where each list contains
                          operations (add/remove) for products.
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (int): The time in seconds to wait before retrying
                                   failed cart operations.
            **kwargs: Arbitrary keyword arguments to be passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs) # Initializes the base Thread class.
        self.carts = carts # Stores the list of shopping carts to process.
        self.marketplace = marketplace # Reference to the Marketplace instance.
        self.retry_wait_time = retry_wait_time # Time to wait before retrying failed operations.
        # Dictionary mapping operation types ("add", "remove") to corresponding marketplace methods.
        self.operations = {"add": marketplace.add_to_cart,
                           "remove": marketplace.remove_from_cart}

    def run(self):
        """
        The main execution loop for the Consumer thread.

        Iterates through each shopping list (cart) provided, attempts to perform
        the specified add/remove operations for products. If an operation fails,
        it retries after a specified wait time. Once all operations for a cart
        are complete, it places the order.
        """
        # Block Logic: Iterates through each shopping list (cart) assigned to this consumer.
        for cart in self.carts:
            cart_id = self.marketplace.new_cart() # Creates a new cart in the marketplace for the current shopping list.

            # Block Logic: Processes each operation (add/remove) within the current shopping list.
            for operation in cart:
                quantity = operation["quantity"] # The quantity of product for the current operation.

                # Block Logic: Continuously attempts to perform the operation until the required quantity is met.
                while quantity > 0:
                    operation_type = operation["type"] # The type of operation ("add" or "remove").
                    product = operation["product"] # The product involved in the operation.

                    # Block Logic: Calls the appropriate marketplace method based on the operation type.
                    if self.operations[operation_type](cart_id, product) is not False:
                        quantity -= 1 # Decrements the remaining quantity if the operation was successful.
                    else:
                        time.sleep(self.retry_wait_time) # Waits if the operation failed, then retries.

            self.marketplace.place_order(cart_id) # Places the order once all operations for the cart are complete.


class Marketplace:
    """
    Acts as the central hub for producers and consumers in the e-commerce simulation.

    It manages product listings from various producers, handles cart creation,
    adding/removing items from carts, and processing orders. All operations
    are designed to be thread-safe, and activities are logged.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes a new Marketplace instance.

        Args:
            queue_size_per_producer (int): The maximum number of products
                                           a single producer can have listed
                                           in the marketplace at any time.
        """
        self.carts_lock = Lock() # Lock for protecting access to the list of carts.
        self.carts = [] # List to store Cart objects, indexed by cart_id.

        self.producers_lock = Lock() # Lock for protecting access to producer-related data.
        self.producers_capacity = queue_size_per_producer # Max capacity for each producer's product list.
        self.producers_sizes = [] # List to track the current size of each producer's product list.
        self.products = [] # Global list of all products currently available in the marketplace (unreserved).

        # Block Logic: Configures logging for the marketplace operations.
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s : %(message)s')
        formatter.converter = time.gmtime # Use GMT for timestamps in logs.

        file_handler = RotatingFileHandler(
            "marketplace.log", maxBytes=4096, backupCount=0) # Log to a rotating file.
        file_handler.setFormatter(formatter) # Sets the format for the file handler.

        logger = logging.getLogger("marketplace") # Gets a logger instance named "marketplace".
        logger.setLevel(logging.INFO) # Sets the logging level to INFO.
        logger.addHandler(file_handler) # Adds the file handler to the logger.
        self.logger = logger

    def register_producer(self):
        """
        Registers a new producer with the marketplace and assigns a unique ID.

        Returns:
            int: The unique ID assigned to the registered producer.
        """
        self.logger.info("enter register_producer()")

        with self.producers_lock: # Ensures thread-safe access to producer-related data.
            self.producers_sizes.append(0) # Adds a new entry for the producer's queue size, initialized to 0.
            self.logger.info("leave register_producer")
            return len(self.producers_sizes) - 1 # Returns the newly assigned producer ID.

    def publish(self, producer_id, product):
        """
        Publishes a product from a specific producer to the marketplace.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Any): The product data to publish.

        Returns:
            bool: True if the product was published successfully, False if
                  the producer's queue is full.
        """
        self.logger.info(
            "enter publish(%d, %s)", producer_id, str(product))

        with self.producers_lock: # Ensures thread-safe access to producer-related data.
            # Block Logic: Checks if the producer's product queue has reached its maximum capacity.
            if self.producers_sizes[producer_id] == self.producers_capacity:
                self.logger.info("leave publish")
                return False # Cannot publish if the queue is full.

            self.producers_sizes[producer_id] += 1 # Increments the count of products for this producer.
            self.products.append((product, producer_id)) # Adds the product to the global list of available products.
            self.logger.info("leave publish")
            return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart and assigns it a unique ID.

        Returns:
            int: The unique ID of the newly created cart.
        """
        self.logger.info("enter new_cart()")
        with self.carts_lock: # Ensures thread-safe access to the carts list.
            self.carts.append([]) # Appends a new empty list to represent the new cart.
            self.logger.info("leave new_cart")
            return len(self.carts) - 1 # Returns the index of the new cart as its ID.

    def add_to_cart(self, cart_id, product):
        
        self.logger.info(
            "enter add_to_cart(%d, %s)", cart_id, str(product))



        self.producers_lock.acquire()
        for (prod, prod_id) in self.products:
            if prod == product:
                self.producers_sizes[prod_id] -= 1
                self.products.remove((prod, prod_id))
                self.producers_lock.release()
                self.carts[cart_id].append((prod, prod_id))
                self.logger.info("leave add_to_cart")
                return True

        self.producers_lock.release()
        self.logger.info("leave add_to_cart")
        return False

    def remove_from_cart(self, cart_id, product):
        
        self.logger.info("enter remove_from_cart(%d, %s)", cart_id, str(product))



        for (prod, prod_id) in self.carts[cart_id]:
            if prod == product:
                self.carts[cart_id].remove((prod, prod_id))
                self.producers_lock.acquire()
                self.products.append((prod, prod_id))
                self.producers_sizes[prod_id] += 1
                self.producers_lock.release()
                self.logger.info("leave remove_from_cart")
                return

    def place_order(self, cart_id):
        
        self.logger.info("enter place_order(%d)", cart_id)



        order = ""
        for (product, _) in self.carts[cart_id]:
            order += "{} bought {}
".format(
                currentThread().getName(), product)
        sys.stdout.write(order)
        self.logger.info("leave place_order")
        return self.carts[cart_id]


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time


        self.producer_id = marketplace.register_producer()

    def run(self):
        while True:
            for (product, quantity, wait_time) in self.products:
                while quantity > 0:
                    if self.marketplace.publish(self.producer_id, product):
                        quantity -= 1
                        time.sleep(wait_time / 20)
                    else:
                        time.sleep(self.republish_wait_time / 20)