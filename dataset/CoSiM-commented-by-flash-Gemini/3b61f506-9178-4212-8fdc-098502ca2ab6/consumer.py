"""
This module implements a multi-threaded producer-consumer simulation for a marketplace,
managing product inventory, producer and consumer interactions, and cart operations.
It includes a logging utility to track marketplace activities.
"""


import time
from threading import Thread
import logging
from logging.handlers import RotatingFileHandler
import functools
import inspect

def setup_logger():
    """
    Configures a logger for the marketplace.
    It sets up a rotating file handler to log messages to 'marketplace.log',
    with a maximum file size and backup count.
    Logging format includes timestamp, level, and message, using UTC time.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler('marketplace.log', maxBytes=500000, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.Formatter.converter = time.gmtime # Use UTC time for log timestamps
    logger.addHandler(handler)


def log_function(wrapped_function):
    """
    A decorator that logs the entry, arguments, and exit (with return value) of a function.
    This provides detailed tracing for marketplace operations.
    """

    @functools.wraps(wrapped_function)
    def wrapper(*args):
        logger = logging.getLogger(__name__)

        # Functional Utility: Logs the name of the function being entered.
        func_name = wrapped_function.__name__
        logger.info("Entering %s", {func_name})

        # Functional Utility: Extracts and logs the arguments passed to the function.
        func_args = inspect.signature(wrapped_function).bind(*args).arguments
        func_args_str = '\n\t'.join(
            f"{var_name} = {var_value}"
            for var_name, var_value
            in func_args.items()
        )
        logger.info("\t%s", func_args_str)

        # Functional Utility: Executes the wrapped function.
        out = wrapped_function(*args)

        # Functional Utility: Logs the return type and value of the function.
        logger.info("Return: %s - %s", type(out), out)
        logger.info("Done running %s", func_name)

        return out

    return wrapper


class Consumer(Thread):
    """
    The Consumer class represents a buyer in the marketplace.
    Each consumer runs as a separate thread, simulating the process of
    adding and removing items from a cart, and finally placing an order.
    Consumers handle retry logic for adding items to the cart if the marketplace
    is temporarily unable to fulfill the request.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        :param carts: A list of carts, where each cart is a list of operation dictionaries.
                      Each operation specifies "type" (add/remove), "product", and "quantity".
        :param marketplace: The shared marketplace instance to interact with.
        :param retry_wait_time: The time in seconds to wait before retrying an operation
                                 (e.g., adding a product to a full cart).
        :param kwargs: Additional keyword arguments. Expects 'name' for the consumer's identifier.
        """
        Thread.__init__(self) # Initializes the base Thread class.
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs.get("name") # Retrieves the consumer's name from kwargs.

    def run(self):
        """
        Executes the consumer's main logic.
        This method is called when the thread starts. It simulates the consumer's
        journey through the marketplace: creating a new cart, processing a series
        of add/remove commands for products, and finally placing the order.
        It handles retries for 'add' operations if the marketplace's cart
        addition fails.
        """
        # Block Logic: Iterates through each predefined cart (sequence of operations) for this consumer.
        for cart in self.carts:
            # Functional Utility: Requests a new cart ID from the marketplace.
            cart_id = self.marketplace.new_cart()
            # Block Logic: Processes each operation (add or remove) within the current cart's sequence.
            for operation in cart:
                product = operation.get("product") # Retrieves the product object for the current operation.
                quantity = operation.get("quantity") # Retrieves the quantity for the current operation.
                # Block Logic: Repeats the add/remove operation for the specified quantity.
                for _ in range(quantity):
                    # Conditional Logic: Handles "add" operations.
                    if operation.get("type") == "add":
                        res = False # Flag to track if the add operation was successful.
                        # Invariant: Continues to loop until the product is successfully added to the cart.
                        # Pre-condition: The marketplace's add_to_cart method might return False, indicating failure.
                        while not res:
                            res = self.marketplace.add_to_cart(cart_id, product)
                            time.sleep(self.retry_wait_time) # Pauses before retrying.
                    # Conditional Logic: Handles "remove" operations.
                    elif operation.get("type") == "remove":
                        self.marketplace.remove_from_cart(cart_id, product)

            # Functional Utility: Places the final order with all accumulated items in the cart.
            products = self.marketplace.place_order(cart_id)

            # Block Logic: Iterates through the items successfully ordered and prints a confirmation message.
            for product in products:
                print(f"{self.name} bought {product}")



from threading import Lock
# from .logger import setup_logger, log_function # Keeping the original import for context, but will define within this file for simplicity as it's a single file.

class Cart:
    """
    Represents a shopping cart for a consumer.
    It stores a list of products and the IDs of the producers who supplied them.
    """

    def __init__(self):
        """
        Initializes an empty shopping cart.
        """
        self.products = [] # Stores the actual product objects in the cart.
        self.producer_ids = [] # Stores the producer_id for each product, maintaining association.

    def add_to_cart(self, product, producer_id):
        """
        Adds a product to the cart, along with its producer's ID.

        :param product: The product object to add.
        :param producer_id: The ID of the producer from whom the product was taken.
        """
        self.products.append(product)
        self.producer_ids.append(producer_id)

    def remove_from_cart(self, product):
        """
        Removes a product from the cart and returns the ID of its producer.
        If multiple instances of the same product exist, only one is removed.

        :param product: The product object to remove.
        :return: The ID of the producer of the removed product, or None if the product was not found.
        """
        # Block Logic: Iterates through the products in the cart to find the specified product.
        for i in range(len(self.products)):
            # Conditional Logic: Checks if the current product in the cart matches the product to be removed.
            if self.products[i] == product:
                producer_id = self.producer_ids[i] # Retrieves the producer ID before removal.
                self.products.remove(product) # Removes the product from the list.
                self.producer_ids.remove(producer_id) # Removes the corresponding producer ID.
                return producer_id
        return None # Returns None if the product was not found in the cart.


class Marketplace:
    """
    The Marketplace class simulates a central hub where producers publish products
    and consumers can add/remove products from their carts to place orders.
    It manages product inventory, producer and consumer registration, and cart operations,
    ensuring thread-safe access to shared resources using various locks.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace with a specified queue size per producer.

        :param queue_size_per_producer: The maximum number of products a single producer
                                        can have available in the marketplace at any given time.
        """
        setup_logger() # Functional Utility: Initializes the logging system for the marketplace.
        self.queue_size_per_producer = queue_size_per_producer # Max products per producer in queues.

        # Data Structure: A dictionary to hold product queues for each producer.
        # Key: producer_id, Value: list of products.
        self.producer_queues = {}
        # Data Structure: A dictionary to hold locks for each producer's product queue.
        # This ensures thread-safe access to individual producer queues.
        self.producer_queues_locks = {}

        self.producer_id_counter = 0 # Counter for assigning unique producer IDs.
        self.producer_id_lock = Lock() # Lock to protect producer_id_counter and producer_queues/locks.

        # Data Structure: A dictionary to hold consumer carts.
        # Key: cart_id, Value: Cart object.
        self.carts = {}

        self.cart_id_counter = 0 # Counter for assigning unique cart IDs.
        self.cart_id_lock = Lock() # Lock to protect cart_id_counter and carts dictionary.

    @log_function
    def register_producer(self):
        """
        Registers a new producer with the marketplace, assigns a unique ID,
        and initializes an empty product queue and a dedicated lock for it.
        This method is thread-safe.

        :return: A unique integer ID for the newly registered producer.
        """
        # Synchronization: Acquires a lock to safely generate a new producer ID and initialize related data structures.
        with self.producer_id_lock:
            producer_id = self.producer_id_counter
            self.producer_queues[producer_id] = [] # Initializes an empty product queue for the new producer.
            self.producer_queues_locks[producer_id] = Lock() # Creates a dedicated lock for this producer's queue.
            self.producer_id_counter += 1 # Increments the counter for the next producer.
            return producer_id

    @log_function
    def publish(self, producer_id, product):
        """
        Publishes a product from a given producer to the marketplace.
        The product is added to the producer's queue only if the producer
        has not exceeded its maximum allowed products in the marketplace.
        This method is thread-safe for the specific producer's queue.

        :param producer_id: The ID of the producer publishing the product.
        :param product: The product object to be published.
        :return: True if the product was successfully published, False otherwise (e.g., queue full).
        """
        # Synchronization: Acquires the specific lock for this producer's queue to ensure thread safety.
        with self.producer_queues_locks[producer_id]:
            # Conditional Logic: Checks if the producer's queue has space for a new product.
            if len(self.producer_queues[producer_id]) <= self.queue_size_per_producer:
                self.producer_queues[producer_id].append(product) # Adds the product to the producer's queue.
                return True
        return False # Returns False if the queue is full.

    @log_function
    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer and assigns a unique ID.
        This method is thread-safe.

        :return: A unique integer ID for the new cart.
        """
        # Synchronization: Acquires a lock to safely generate a new cart ID and create a new Cart object.
        with self.cart_id_lock:
            cart_id = self.cart_id_counter
            self.carts[cart_id] = Cart() # Creates a new Cart object and stores it.
            self.cart_id_counter += 1 # Increments the counter for the next cart.
            return cart_id

    @log_function
    def add_to_cart(self, cart_id, product):
        """
        Adds a specified product to a consumer's cart by taking it from an available producer's queue.
        It iterates through all producer queues, finds the first occurrence of the product,
        removes it, and adds it to the consumer's cart. This method is thread-safe.

        :param cart_id: The ID of the consumer's cart.
        :param product: The product object to add to the cart.
        :return: True if the product was successfully added, False if not found or unavailable.
        """
        producers_no = 0
        # Synchronization: Acquires a lock to safely read the number of registered producers.
        with self.producer_id_lock:
            producers_no = self.producer_id_counter # Gets the total number of producers.

        # Block Logic: Iterates through each producer's queue to find the product.
        for i in range(producers_no):
            # Synchronization: Acquires the lock for the current producer's queue to safely check and modify it.
            with self.producer_queues_locks[i]:
                # Conditional Logic: Checks if the desired product is present in the current producer's queue.
                if product in self.producer_queues[i]:
                    self.producer_queues[i].remove(product) # Removes the product from the producer's queue.
                    self.carts[cart_id].add_to_cart(product, i) # Adds the product to the consumer's cart.
                    return True # Returns True as the product was successfully added.
        return False # Returns False if the product was not found in any producer's queue.

    @log_function
    def remove_from_cart(self, cart_id, product):
        """
        Removes a specified product from a consumer's cart and returns it to its original producer's queue.
        This method ensures that the product is returned to the correct producer and maintains thread safety.

        :param cart_id: The ID of the consumer's cart.
        :param product: The product object to remove from the cart.
        """
        # Functional Utility: Removes the product from the cart and retrieves its original producer's ID.
        producer_id = self.carts[cart_id].remove_from_cart(product)
        # Synchronization: Acquires the lock for the original producer's queue to safely add the product back.
        with self.producer_queues_locks[producer_id]:
            self.producer_queues[producer_id].append(product) # Appends the product back to its producer's queue.

    @log_function
    def place_order(self, cart_id):
        """
        Finalizes the order for a given cart.
        In this simulation, placing an order simply means returning the contents
        of the specified cart's products list. No further processing (e.g., payment, shipping)
        is simulated.

        :param cart_id: The ID of the cart to place the order for.
        :return: A list of product objects in the placed order.
        """
        return self.carts[cart_id].products


from threading import Thread
import time


class Producer(Thread):
    """
    The Producer class represents a seller in the marketplace.
    Each producer runs as a separate thread, continuously publishing products
    to the marketplace based on its predefined inventory, quantity, and timing.
    Producers will retry publishing if the marketplace's capacity for them is full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        :param products: A list of product operations. Each operation is a tuple
                         (product_object, quantity, sleep_time_after_each_batch).
        :param marketplace: The shared marketplace instance to interact with.
        :param republish_wait_time: The time in seconds to wait before retrying to publish
                                    a product if the marketplace is full for this producer.
        :param kwargs: Additional keyword arguments. Expects 'daemon' for thread daemon status.
        """
        thread_arg = kwargs["daemon"] # Retrieves the 'daemon' argument for the thread.
        Thread.__init__(self, daemon=thread_arg) # Initializes the base Thread class with daemon status.
        self.operations = products # The inventory and publishing schedule of products for this producer.
        self.marketplace = marketplace # Reference to the shared marketplace.
        self.republish_wait_time = republish_wait_time # Time to wait before retrying to publish.

    def run(self):
        """
        Executes the producer's main logic.
        This method is called when the thread starts. It first registers itself
        with the marketplace and then continuously iterates through its defined
        product operations, attempting to publish products. It includes retry logic
        if the marketplace refuses publication (e.g., due to capacity limits).
        """
        producer_id = self.marketplace.register_producer() # Registers the producer and gets its unique ID.
        # Invariant: The producer continuously attempts to publish products.
        while True:
            # Block Logic: Iterates through each defined product operation in the producer's schedule.
            for operation in self.operations:
                product = operation[0] # The product object to be published.
                quantity = operation[1] # The number of this product to publish in this batch.
                sleep_time = operation[2] # Time to sleep after publishing a batch of this product.
                time.sleep(sleep_time) # Functional Utility: Pauses before starting to publish a new batch of products.
                # Block Logic: Publishes the specified quantity of the current product type.
                for _ in range(quantity):
                    # Conditional Logic: Attempts to publish the product.
                    # If publishing fails (e.g., producer's queue is full), it retries after a wait.
                    if not self.marketplace.publish(producer_id, product):
                        time.sleep(self.republish_wait_time) # Pauses before retrying to publish.


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Represents a generic product with a name and price.
    This is a frozen dataclass, meaning its instances are immutable and hashable.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Represents a specific type of product: Tea.
    It inherits from Product and adds a 'type' attribute to specify tea variety.
    This is a frozen dataclass, meaning its instances are immutable and hashable.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Represents a specific type of product: Coffee.
    It inherits from Product and adds 'acidity' and 'roast_level' attributes
    to specify coffee characteristics.
    This is a frozen dataclass, meaning its instances are immutable and hashable.
    """
    acidity: str
    roast_level: str
