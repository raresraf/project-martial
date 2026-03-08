"""
@495b7d0f-3326-49a1-b0a4-7f1b8db42ad7/consumer.py
@brief This module simulates a multi-threaded e-commerce system, featuring
`Consumer` and `Producer` threads interacting with a `Marketplace`. It
includes logging, cart management, and product definitions using dataclasses.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    Represents a consumer in the e-commerce simulation.

    Consumers create shopping carts, add/remove products, and place orders
    within the marketplace. Each consumer operates as a separate thread.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping cart command lists for this consumer.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying an action if it fails.
            **kwargs: Keyword arguments passed to the Thread constructor,
                      e.g., `name` for the thread's name.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs.get("name") # Assign the thread's name from kwargs.

    def run(self):
        """
        The main execution method for the consumer thread.

        It processes a list of cart commands, creating new carts,
        adding/removing products, and finally placing orders.
        """
        # Block Logic: Process each predefined cart for this consumer.
        for cart in self.carts:
            cart_id = self.marketplace.new_cart() # Create a new cart in the marketplace.
            # Block Logic: Execute operations (add or remove products) for the current cart.
            for operation in cart:
                product = operation.get("product") # Get product to operate on.
                quantity = operation.get("quantity") # Get quantity of product.
                # Loop to perform the operation for the specified quantity.
                for _ in range(quantity):
                    if operation.get("type") == "add":
                        res = False
                        # Block Logic: Continuously attempt to add the product until successful.
                        while not res:
                            res = self.marketplace.add_to_cart(cart_id, product)
                            time.sleep(self.retry_wait_time) # Wait before retrying if addition failed.
                    elif operation.get("type") == "remove":
                        # Action: Remove the product from the cart.
                        self.marketplace.remove_from_cart(cart_id, product)

            # Functional Utility: Place the order for the current cart.
            products = self.marketplace.place_order(cart_id)

            # Functional Utility: Print the products bought by this consumer.
            for product in products:
                print(f"{self.name} bought {product}")


import logging
from logging.handlers import RotatingFileHandler
import functools
import inspect
import time

def setup_logger():
    """
    Configures the logger for the marketplace.

    Sets up a rotating file handler to log messages to 'marketplace.log',
    with a maximum file size and backup count. The formatter includes
    timestamp, log level, and message, using GMT for timestamps.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler('marketplace.log', maxBytes=500000, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.Formatter.converter = time.gmtime # Use GMT for log timestamps.
    logger.addHandler(handler)


def log_function(wrapped_function):
    """
    A decorator to log function calls, arguments, and return values.

    This provides an audit trail for important functions within the marketplace,
    aiding in debugging and understanding system behavior.
    """

    @functools.wraps(wrapped_function)
    def wrapper(*args):
        logger = logging.getLogger(__name__)

        # Log function entry.
        func_name = wrapped_function.__name__
        logger.info("Entering %s", {func_name})

        # Log function arguments.
        func_args = inspect.signature(wrapped_function).bind(*args).arguments
        func_args_str = '\n\t'.join(
            f"{var_name} = {var_value}"
            for var_name, var_value
            in func_args.items()
        )
        logger.info("\t%s", func_args_str)

        # Execute the original function.
        out = wrapped_function(*args)

        # Log function return value and exit.
        logger.info("Return: %s - %s", type(out), out)
        logger.info("Done running %s", func_name)

        return out

    return wrapper


from threading import Lock
from .logger import setup_logger, log_function

class Cart:
    """
    Represents a shopping cart, holding products and their originating producer IDs.

    This class provides functionality to add and remove products from the cart,
    keeping track of where each product came from for return purposes.
    """
    

    def __init__(self):
        """
        Initializes an empty shopping cart.
        """
        self.products = []      # List to store the product objects added to the cart.
        self.producer_ids = []  # List to store the producer ID corresponding to each product in `self.products`.

    def add_to_cart(self, product, producer_id):
        """
        Adds a product to the cart along with its originating producer ID.

        Args:
            product (object): The product object to add.
            producer_id (int): The ID of the producer from whom the product was acquired.
        """
        self.products.append(product)
        self.producer_ids.append(producer_id)

    def remove_from_cart(self, product):
        """
        Removes a specific product from the cart and returns the ID of its producer.

        Args:
            product (object): The product object to remove.

        Returns:
            int or None: The producer ID if the product was found and removed, otherwise None.
        """
        # Block Logic: Iterate through the products in the cart to find the one to remove.
        for i in range(len(self.products)):
            if self.products[i] == product:
                producer_id = self.producer_ids[i] # Store the producer ID before removal.
                self.products.remove(product)      # Remove the product from the list.
                self.producer_ids.remove(producer_id) # Remove the corresponding producer ID.
                return producer_id # Return the producer ID so the product can be returned to inventory.
        return None # Product not found in the cart.


class Marketplace:
    """
    Manages the overall e-commerce system, connecting producers and consumers.

    It handles product publication, cart creation, adding/removing products
    from carts, and placing orders, ensuring thread-safe operations.
    """
    

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace with a given queue size per producer.

        Args:
            queue_size_per_producer (int): The maximum number of products a producer
                                           can have in the marketplace's inventory.
        """
        setup_logger() # Setup logging for the marketplace operations.
        self.queue_size_per_producer = queue_size_per_producer

        # Dictionary to store product queues for each producer.
        # Key: producer_id, Value: list of products.
        self.producer_queues = {}

        # Dictionary to store locks for each producer's queue, ensuring thread-safe access.
        # Key: producer_id, Value: Lock object.
        self.producer_queues_locks = {}

        self.producer_id_counter = 0 # Counter for assigning unique producer IDs.
        self.producer_id_lock = Lock() # Lock for protecting producer ID counter.

        # Dictionary to store shopping carts. Key: cart_id, Value: Cart object.
        self.carts = {}

        self.cart_id_counter = 0 # Counter for assigning unique cart IDs.
        self.cart_id_lock = Lock() # Lock for protecting cart ID counter.

    @log_function
    def register_producer(self):
        """
        Registers a new producer with the marketplace.

        Assigns a unique ID to the producer and initializes an empty product queue
        and a dedicated lock for that producer.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        # Block Logic: Acquire lock to ensure atomic assignment of producer ID.
        with self.producer_id_lock:
            producer_id = self.producer_id_counter
            self.producer_queues[producer_id] = [] # Initialize an empty queue for the new producer.
            self.producer_queues_locks[producer_id] = Lock() # Create a new lock for the producer's queue.
            self.producer_id_counter += 1
            return producer_id

    @log_function
    def publish(self, producer_id, product):
        """
        Publishes a product from a producer to the marketplace.

        The product is added to the producer's queue only if the queue size
        is within the allowed limit.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (object): The product object to publish.

        Returns:
            bool: True if the product was successfully published, False otherwise.
        """
        # Block Logic: Acquire the producer's specific lock to ensure thread-safe queue modification.
        with self.producer_queues_locks[producer_id]:
            # Pre-condition: Check if the producer's queue is not full.
            if len(self.producer_queues[producer_id]) <= self.queue_size_per_producer:
                self.producer_queues[producer_id].append(product) # Add product to the queue.
                return True # Product successfully published.
        return False # Publication failed due to full queue.

    @log_function
    def new_cart(self):
        """
        Creates a new, empty shopping cart and returns its unique ID.

        Returns:
            int: The unique ID of the newly created cart.
        """
        # Block Logic: Acquire lock to ensure atomic assignment of cart ID.
        with self.cart_id_lock:
            cart_id = self.cart_id_counter
            self.carts[cart_id] = Cart() # Create a new Cart object.
            self.cart_id_counter += 1
            return cart_id

    @log_function
    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified cart.

        This involves searching through producers' queues for the product,
        removing it from the first available producer, and adding it to the cart.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (object): The product object to add.

        Returns:
            bool: True if the product was successfully added, False if not found.
        """
        producers_no = 0
        # Block Logic: Get the current number of registered producers in a thread-safe manner.
        with self.producer_id_lock:
            producers_no = self.producer_id_counter

        # Block Logic: Iterate through producers to find and retrieve the product.
        for i in range(producers_no):
            # Acquire lock for the producer's queue.
            with self.producer_queues_locks[i]:
                # Pre-condition: Check if the product is available in the current producer's queue.
                if product in self.producer_queues[i]:
                    self.producer_queues[i].remove(product) # Remove product from producer's queue.
                    self.carts[cart_id].add_to_cart(product, i) # Add product to cart with producer ID.
                    return True # Product successfully added.
        return False # Product not found in any producer's queue.

    @log_function
    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specified cart and returns it to its original producer's queue.

        Args:
            cart_id (int): The ID of the cart to remove the product from.
            product (object): The product object to remove.
        """
        # Functional Utility: Remove the product from the cart and get its original producer ID.
        producer_id = self.carts[cart_id].remove_from_cart(product)
        # Block Logic: Acquire the producer's lock and return the product to its queue.
        with self.producer_queues_locks[producer_id]:
            self.producer_queues[producer_id].append(product)

    @log_function
    def place_order(self, cart_id):
        """
        Retrieves the list of products in a specified cart, effectively placing an order.

        Args:
            cart_id (int): The ID of the cart to place the order for.

        Returns:
            list: A list of product objects in the ordered cart.
        """
        return self.carts[cart_id].products


from threading import Thread
import time


class Producer(Thread):
    """
    Represents a producer in the e-commerce simulation.

    Producers generate products and attempt to publish them to the marketplace,
    retrying if the marketplace's queue for that producer is full.
    Each producer operates as a separate daemon thread.
    """
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of product operations. Each operation is a tuple
                             containing (product object, quantity, sleep_time before next publish).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish a product
                                         if the marketplace queue is full.
            **kwargs: Keyword arguments passed to the Thread constructor, e.g., `daemon`.
        """
        thread_arg = kwargs["daemon"] # Extract daemon status for the thread.
        Thread.__init__(self, daemon=thread_arg)
        self.operations = products # List of product operations to perform.
        self.marketplace = marketplace # The marketplace instance to interact with.
        self.republish_wait_time = republish_wait_time # Delay for republishing.

    def run(self):
        """
        The main execution method for the producer thread.

        It registers itself with the marketplace, then continuously iterates
        through its defined product operations, publishing products and
        handling delays for both successful publications and retries.
        """
        producer_id = self.marketplace.register_producer() # Register with the marketplace to get a unique ID.
        while True:
            # Block Logic: Iterate through each product operation defined for this producer.
            for operation in self.operations:
                product = operation[0] # The product object to publish.
                quantity = operation[1] # The quantity of this product to publish.
                sleep_time = operation[2] # Time to wait after publishing this product.
                time.sleep(sleep_time) # Wait for the specified time before publishing.
                # Block Logic: Publish the product multiple times according to the specified quantity.
                for _ in range(quantity):
                    # Pre-condition: Attempt to publish the product. If fails, wait and retry.
                    if not self.marketplace.publish(producer_id, product):
                        time.sleep(self.republish_wait_time) # Wait before retrying if publication failed.


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Represents a generic product with a name and price.

    Attributes:
        name (str): The name of the product.
        price (int): The price of the product.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Represents a type of tea product, inheriting from Product.

    Attributes:
        type (str): The specific type or blend of tea.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Represents a type of coffee product, inheriting from Product.

    Attributes:
        acidity (str): Describes the acidity level of the coffee.
        roast_level (str): Describes the roast level of the coffee.
    """
    acidity: str
    roast_level: str
