"""
This module implements a multi-threaded producer-consumer marketplace simulation.

It defines:
- Consumer: A thread that simulates a buyer, adding and removing products from a cart, and placing orders.
- Marketplace: The central hub managing product inventories from Producers and handling consumer cart operations, ensuring thread safety.
- Producer: A thread that simulates a seller, continuously publishing products to the Marketplace.
- Product, Tea, Coffee: Dataclasses defining the structure of various product types.

The module also includes `MarketplaceTestCase` for unit testing the `Marketplace` class.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    Represents a consumer (buyer) in the marketplace simulation.

    Each consumer operates as a separate thread, executing a series of shopping tasks
    (adding and removing products from a cart) and eventually placing an order.
    It handles retries if a marketplace operation fails.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a new Consumer thread.

        Args:
            carts (list): A list of cart definitions. Each cart is a list of tasks,
                          where each task is a dictionary like
                          `{'type': 'add'/'remove', 'product': product_obj, 'quantity': int}`.
            marketplace (Marketplace): The shared marketplace instance to interact with.
            retry_wait_time (float): The time (in seconds) to wait before retrying a failed operation.
            **kwargs: Arbitrary keyword arguments passed to the base `Thread` constructor,
                      e.g., `name` for thread identification.
        """
        Thread.__init__(self, **kwargs) # Initialize the base Thread class with kwargs.
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs # Store kwargs for potential future use or debugging.

    def run(self):
        """
        The main execution method for the Consumer thread.

        This method simulates the shopping process for the consumer:
        1. Creates a new shopping cart in the marketplace.
        2. Iterates through a list of product tasks (add or remove products).
        3. For 'add' tasks, it attempts to add products to the cart, retrying if the operation fails.
        4. For 'remove' tasks, it removes products from the cart.
        5. Finally, it places the order and prints the purchased items to standard output.
        """
        cart_id = self.marketplace.new_cart() # Request a new cart ID from the marketplace.


        # Block Logic: Process each product task defined in the consumer's carts.
        for product_task_list in self.carts: # Renamed 'product' to 'product_task_list' for clarity.
            for attribute in product_task_list: # Each 'attribute' is a task dictionary.
                command = attribute.get("type") # 'add' or 'remove'
                product = attribute.get("product") # The product object
                quantity = attribute.get("quantity") # The quantity for this task

                if command == "remove":
                    i = 0
                    while i < quantity:
                        self.marketplace.remove_from_cart(cart_id, product)
                        i += 1
                elif command == "add":
                    i = 0
                    while i < quantity:
                        # Attempt to add product to cart. `no_wait` indicates if adding was successful.
                        no_wait = self.marketplace.add_to_cart(cart_id, product)
                        if no_wait:
                            i += 1 # Successfully added, proceed to next unit.
                        else:
                            # If adding failed (e.g., product not available), wait and retry.
                            time.sleep(self.retry_wait_time)
        
        # Block Logic: Place the order and print the items bought.
        order = self.marketplace.place_order(cart_id)
        for prod in order:
            print(self.name, "bought", prod)


from threading import Lock
from logging.handlers import RotatingFileHandler
import logging
import time

# --- Logging Setup ---
LOGGER = logging.getLogger('marketplace_logger') # Get a logger instance.
LOGGER.setLevel(logging.INFO) # Set the logging level to INFO.

FORMATTER = logging.Formatter('%(levelname)s:%(name)s:%(message)s') # Define log message format.
FORMATTER.converter = time.gmtime # Use GMT time for logging timestamps.

# Configure a RotatingFileHandler to manage log file size and rotation.
# MaxBytes=5000: Log file will rotate when it reaches 5KB.
# backupCount=10: Keep up to 10 old log files.
HANDLER = RotatingFileHandler('marketplace.log', maxBytes=5000, backupCount=10)
HANDLER.setFormatter(FORMATTER) # Apply the defined formatter to the handler.

LOGGER.addHandler(HANDLER) # Add the handler to the logger.


class Marketplace:
    """
    Manages the central logic for product exchange between producers and consumers.

    It handles producer registration, product publishing, shopping cart creation,
    and thread-safe addition/removal of products from carts. It also incorporates
    logging for tracing marketplace operations.
    """
    
    def __init__(self, queue_size_per_producer: int):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products each producer
                                           can have in its queue at any given time.
        """
        # Locks to ensure thread-safe access to shared resources within the marketplace.
        self.lock_consumer = Lock() # Protects consumer-related operations (carts, adding/removing).
        self.lock_producer = Lock() # Protects producer-related operations (producer queues, publishing).

        self.producers = [[]] # List of lists, where each inner list is a product queue for a producer.
                              # Initialized with an empty list at index 0, as producer IDs start from 1.
        self.carts = [[]]     # List of lists, where each inner list represents a consumer's shopping cart.
                              # Initialized with an empty list at index 0, as cart IDs start from 1.
        
        self.no_producers = 0 # Counter for the total number of registered producers.
        self.no_carts = 0     # Counter for the total number of created carts.

        self.queue_size_per_producer = queue_size_per_producer # Max products allowed per producer's queue.



    def register_producer(self):
        """
        Registers a new producer with the marketplace and assigns it a unique ID.

        Returns:
            int: The unique ID assigned to the registered producer.
        """
        LOGGER.info("A new producer is registered.")
        
        # Critical Section: Protecting modification of producer count and list.
        # This implementation does not use a lock specifically for `no_producers`
        # or `self.producers.append([])`, which could lead to race conditions
        # if multiple producers register simultaneously.
        self.no_producers += 1
        self.producers.append([]) # Appends a new empty queue for the new producer.
        
        LOGGER.info("Producer with id %s registered.", self.no_producers)

        return self.no_producers

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer to its respective queue in the marketplace.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Product): The product object to be published.

        Returns:
            bool: True if the product was successfully published (queue was not full), False otherwise.

        Raises:
            ValueError: If the `producer_id` is invalid.
        """
        LOGGER.info('Producer with id %d is publishing the product: %s', producer_id, product)
        # Validate producer ID.
        if producer_id > self.no_producers:
            LOGGER.error('Producer with id: %d does not exist', producer_id)
            raise ValueError("Producer does not exist!")

        product_list = self.producers[producer_id] # Get the specific producer's queue.
        with self.lock_producer: # Critical Section: Protects access to the producer's queue.
            if len(product_list) >= self.queue_size_per_producer:
                can_publish = False # Queue is full.
            else:
                product_list.append(product) # Add product to the queue.
                can_publish = True
        LOGGER.info("Producer published: %s", str(can_publish))

        return can_publish

    def new_cart(self):
        """
        Creates a new empty shopping cart and assigns it a unique ID.

        Returns:
            int: The unique ID assigned to the new cart.
        """
        LOGGER.info("New cart with id %d is being created.", self.no_carts + 1)
        # Critical Section: Protecting modification of cart count and list.
        # This implementation does not use a lock specifically for `no_carts`
        # or `self.carts.append([])`, which could lead to race conditions
        # if multiple carts are created simultaneously.
        self.no_carts += 1
        self.carts.append([]) # Appends a new empty cart.

        return self.no_carts

    def add_to_cart(self, cart_id, product):
        """
        Attempts to add a specified product to a consumer's cart.

        It searches all producer queues for the product. If found, it removes
        the product from the producer's queue and adds it to the consumer's cart.

        Args:
            cart_id (int): The ID of the cart to which the product should be added.
            product (Product): The product object to add.

        Returns:
            bool: True if the product was found and added to the cart, False otherwise.
        """
        LOGGER.info("Cart with id %d is adding %s.", cart_id, product)
        can_add = False
        index = -1 # Used to store the producer_id if the product is found.
        
        with self.lock_consumer: # Critical Section: Protects access to consumer carts and producer queues.
                                 # This lock might be too broad, potentially blocking other add/remove ops.
            # Block Logic: Search all producer queues for the product.
            for i in range(0, self.no_producers): # Iterate through active producer queues.
                for prod_in_list in self.producers[i]: # Iterate through products in the current producer's queue.
                    if prod_in_list == product:
                        index = i # Found the product, record the producer's ID.
                        break # Exit inner loop (found product).
            
            if index >= 0: # If the product was found in a producer's queue.
                self.carts[cart_id].append(product) # Add product to the consumer's cart.
                # BUG: The product is added to the cart but NOT removed from the producer's queue.
                # This leads to products being available multiple times even after being "bought".
                can_add = True

        if can_add:
            LOGGER.info("Product was added to the cart.")
        else:
            LOGGER.info("Product could not be added to the cart.")

        return can_add

    def remove_from_cart(self, cart_id, product):
        """
        Removes a specified product from a consumer's cart.

        Args:
            cart_id (int): The ID of the cart from which the product should be removed.
            product (Product): The product object to remove.

        Returns:
            None: This function does not explicitly return a value, but modifies the cart.
                  It also doesn't return the product to any producer's queue.
        """
        LOGGER.info("Cart with id %d is removing product %s.", cart_id, product)
        found = False
        with self.lock_consumer: # Critical Section: Protects access to consumer carts.
            if product in self.carts[cart_id]: # Check if the product exists in the cart.
                found = True
            if found:
                self.carts[cart_id].remove(product) # Remove the product from the cart.
                # BUG: The removed product is NOT returned to any producer's queue.
                # This leads to a permanent loss of the product from the marketplace.

    def place_order(self, cart_id):
        """
        Processes a consumer's order by returning the contents of their cart.

        This method also removes the ordered products from the original producers'
        queues.

        Args:
            cart_id (int): The ID of the cart to place an order for.

        Returns:
            list: A copy of the list of products that were in the placed order.

        Raises:
            ValueError: If the `cart_id` is invalid.
        """
        LOGGER.info("Cart with id %d placed an order.", cart_id)
        # Validate cart ID.
        if cart_id > self.no_carts:
            LOGGER.error("Cart with id %d is invalid!", cart_id)
            raise ValueError("Cart does not exist!")

        # Block Logic: Remove products from producer queues after an order is placed.
        # BUG: This logic is flawed. It removes *any* instance of the product from *any* producer
        # queue it finds, without tracking which producer originally supplied the product to the cart.
        # This can lead to incorrect inventory management and race conditions.
        for prod in self.carts[cart_id]:
            for producer_queue in self.producers: # Iterate through all producer queues.
                if prod in producer_queue:
                    producer_queue.remove(prod) # Remove the product from the producer's queue.
                    break # Break after finding and removing the first instance.

        LOGGER.info("Product list: %s.", self.carts[cart_id])

        return self.carts[cart_id].copy() # Return a copy of the ordered products.

# --- Unit Testing Section ---
# This part of the code is typically found in a separate test file.
import unittest
# from marketplace import Marketplace # This import would be used if Marketplace was in a separate file.
# from product import Product # This import would be used if Product was in a separate file.


class MarketplaceTestCase(unittest.TestCase):
    """
    Unit tests for the Marketplace class functionalities.
    """

    # Setup for test cases.
    product_test = Product("coffee", 10)
    marketplace = Marketplace(4) # Initialize marketplace with a queue size.

    def test_place_order_exception(self):
        """
        Tests that `place_order` raises a ValueError for an invalid cart ID.
        """
        marketplace = Marketplace(2)
        self.assertRaises(ValueError, marketplace.place_order, 1)

    def test_place_order(self):
        """
        Tests the `place_order` method with a valid scenario.
        """
        self.marketplace.carts = [[self.product_test]] # Manually set up a cart.
        self.marketplace.no_carts = 1 # Manually set cart count.
        response = self.marketplace.place_order(0) # Cart ID 0 is used for internal testing array access.
        expected = [self.product_test]
        self.assertEqual(response, expected)

    def test_register_producer(self):
        """
        Tests the `register_producer` method for correct producer ID assignment.
        """
        marketplace = Marketplace(10)
        result = marketplace.register_producer()
        self.assertEqual(result, 1) # Expected first producer ID to be 1.

    def test_publish_exception(self):
        """
        Tests that `publish` raises a ValueError for an invalid producer ID.
        """
        marketplace = Marketplace(5)
        self.assertRaises(ValueError, marketplace.publish, 2, Product("coffee", 10))

    def test_publish_method(self):
        """
        Tests the `publish` method for successful product publishing.
        """
        self.marketplace.register_producer() # Register a producer.
        result = self.marketplace.publish(1, self.product_test) # Publish a product from producer ID 1.
        self.assertEqual(result, True)

    def test_publish_method_false(self):
        """
        Tests the `publish` method when the producer's queue is full.
        """
        self.marketplace.register_producer()
        for _ in range(0, 4): # Fill the queue up to its capacity (4).
            self.marketplace.publish(1, self.product_test)
        response = self.marketplace.publish(1, self.product_test) # Attempt to publish when full.
        self.assertEqual(response, False)

    def test_new_cart(self):
        """
        Tests the `new_cart` method for correct cart ID assignment.
        """
        marketplace = Marketplace(2)
        result = marketplace.new_cart()
        self.assertEqual(result, 1) # Expected first cart ID to be 1.

    def test_add_cart(self):
        """
        Tests the `add_to_cart` method for successful product addition.
        """
        self.marketplace.register_producer() # Register a producer.
        self.marketplace.publish(0, self.product_test) # Publish product to producer 0.
                                                      # Note: producer 0 might not be the registered one.
        self.marketplace.new_cart() # Create a new cart.
        result = self.marketplace.add_to_cart(1, self.product_test) # Add product to cart 1.
        self.assertEqual(result, True)

    def test_add_cart_false(self):
        """
        Tests the `add_to_cart` method when the product is not available.
        """
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        result = self.marketplace.add_to_cart(1, self.product_test) # Attempt to add unavailable product.
        self.assertEqual(result, False)

    def test_remove_from_cart(self):
        """
        Tests the `remove_from_cart` method.
        """
        self.marketplace.register_producer()
        self.marketplace.publish(0, self.product_test)
        self.marketplace.new_cart()
        # BUG in test: `remove_from_cart` does not return anything explicitly, so assertIsNone is not ideal.
        # It also calls `publish` with producer ID 0 which is not the one registered above if
        # register_producer() is called multiple times.
        self.assertIsNone(self.marketplace.remove_from_cart(0, self.product_test))

if __name__ == '__main__':
    unittest.main() # Run all unit tests when the script is executed directly.


class Producer(Thread):
    """
    Represents a producer (seller) in the marketplace simulation.

    Each producer operates as a separate thread, continuously publishing products
    to the marketplace's queues. It handles delays for republishing.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a new Producer thread.

        Args:
            products (list): A list of product definitions. Each element is a tuple
                             `(product_obj, quantity_to_publish, publish_wait_time_per_unit)`.
            marketplace (Marketplace): The shared marketplace instance to interact with.
            republish_wait_time (float): The time (in seconds) to wait before retrying
                                         to publish if the queue is full.
            **kwargs: Arbitrary keyword arguments passed to the base `Thread` constructor,
                      e.g., `daemon` status and `name`.
        """
        # Initialize the base Thread class. Set as daemon if specified.
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs # Store kwargs for potential future use or debugging.

    def run(self):
        """
        The main execution method for the Producer thread.

        It registers itself with the marketplace to get a unique producer ID.
        Then, it continuously attempts to publish its defined products using
        a helper `publish` method. The loop is `while True` which means it will
        run indefinitely unless explicitly stopped or daemon threads terminate.
        """
        # BUG: `register_producer()` should ideally be called once, not in a loop.
        # If called repeatedly, it will register a new producer in every iteration.
        while True:
            prod_id = self.marketplace.register_producer() # Registers a new producer each loop iteration.
            for product_details in self.products: # Renamed 'product' to 'product_details' for clarity.
                i = 0 # Counter for the number of units of the current product published.
                self.publish(i, prod_id, product_details) # Calls helper method to publish product.

    def publish(self, i, prod_id, product_details):
        """
        Helper method to publish products from this producer to the marketplace.

        It attempts to publish the specified quantity of a product, with delays
        between successful publishes and retries for failed attempts (e.g., if queue is full).

        Args:
            i (int): Initial quantity counter (appears to be always 0 from `run` method).
            prod_id (int): The ID of this producer in the marketplace.
            product_details (tuple): A tuple `(product_obj, quantity_to_publish, publish_wait_time_per_unit)`.
        """
        # product_details[1] is quantity_to_publish, product_details[0] is product_obj.
        while i < product_details[1]: # Loop until the desired quantity for this product type is published.
            # Attempt to publish one unit of the product. `no_wait` indicates if publishing was successful.
            no_wait = self.marketplace.publish(prod_id, product_details[0])
            if no_wait:
                i += 1 # Successfully published, increment counter.
                time.sleep(product_details[2]) # Wait for specified time after successful publish.
            else:
                # If publishing failed (e.g., producer's queue is full), wait and retry.
                time.sleep(self.republish_wait_time)


# --- Product Dataclasses ---
# These dataclasses define the structure for various product types.
from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Base dataclass representing a generic product.

    Attributes:
        name (str): The name of the product.
        price (int): The price of the product.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Dataclass representing a type of tea, inheriting from Product.

    Attributes:
        type (str): The type of tea (e.g., "Green", "Black", "Herbal").
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Dataclass representing a type of coffee, inheriting from Product.

    Attributes:
        acidity (str): The acidity level of the coffee.
        roast_level (str): The roast level of the coffee.
    """
    acidity: str
    roast_level: str
