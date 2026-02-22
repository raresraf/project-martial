"""
@e1f7799d-c6b3-4a95-8650-c4c15bb861ec/consumer.py
@brief Implements a multi-threaded producer-consumer system with a marketplace and unit tests.

This module sets up a simulation for a concurrent e-commerce-like system, featuring producers supplying
products, consumers placing orders via carts, and a central marketplace managing operations. It
incorporates thread-safe mechanisms, logging for operational insights, and comprehensive unit tests
to ensure the correct functionality of the Marketplace component.
"""

from threading import Thread, Lock, currentThread
import time
from dataclasses import dataclass

# Functional Utility: Configures basic logging for the marketplace operations.
# Logging information is written to 'marketplace.log' and rotated.
from logging.handlers import RotatingFileHandler
import unittest
import logging
# from .product import Product # This import path seems problematic: '.product' suggests a relative import within a package, but this file seems top-level.
                             # Assuming 'Product' class definitions are either defined elsewhere or need an absolute path if not defined in this file.
                             # For now, relying on the dataclass definitions later in this file.

logging.basicConfig(
    handlers =[RotatingFileHandler('marketplace.log', maxBytes = 500000, backupCount = 20)],
    format = "%(asctime)s %(levelname)s %(funcName)s %(message)s",
    level = logging.INFO)

logging.Formatter.converter = time.gmtime # Changed gmtime to time.gmtime for clarity.


class Consumer(Thread):
    """
    @brief Represents a consumer thread that interacts with the marketplace to buy products.

    Each Consumer processes a series of operations to add or remove products from a cart,
    and then places an order through the shared Marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping carts, each containing a list of operations to perform.
            marketplace (Marketplace): The shared marketplace instance to interact with.
            retry_wait_time (float): The time in seconds to wait before retrying an operation
                                     if a product is not immediately available.
            **kwargs: Arbitrary keyword arguments, including 'name' for the consumer thread.
        """
        # Functional Utility: Calls the constructor of the parent Thread class, passing kwargs.
        Thread.__init__(self) # kwargs are not passed here, so Thread.init(self, **kwargs) is not used directly.
        # Functional Utility: Stores the list of cart operations for this consumer.
        self.carts = carts
        # Functional Utility: Stores a reference to the shared Marketplace instance.
        self.marketplace = marketplace
        # Functional Utility: Stores the retry wait time for product unavailability.
        self.retry_wait_time = retry_wait_time
        # Functional Utility: Stores the name of the consumer, primarily for logging.
        self.name = kwargs['name']

    def run(self):
        """
        @brief Executes the consumer's purchasing logic.

        Iterates through each cart's operations, adding or removing products,
        and handles retries if products are not available. Finally, it places the order.
        """
        # Functional Utility: Creates a new cart in the marketplace and gets its unique ID.
        id_cart = self.marketplace.new_cart()

        # Block Logic: Iterates through each list of commands representing a shopping session.
        # Pre-condition: 'self.carts' contains lists of commands.
        # Invariant: Each command list will be processed for the current cart.
        for commands in self.carts:
            # Block Logic: Processes each individual command (add/remove) within the current shopping session.
            # Pre-condition: 'commands' is a list of dictionaries detailing cart operations.
            # Invariant: Each command is attempted the specified number of times.
            for command in commands:
                type_command = command["type"]
                product = command["product"]
                prod_quantity = command["quantity"]

                # Conditional Logic: Handles "add" commands.
                if type_command == "add":
                    # Block Logic: Attempts to add the product 'prod_quantity' times, retrying if unavailable.
                    # Pre-condition: 'prod_quantity' is the number of units to add.
                    # Invariant: The loop continues until all units are added, or retry limit is reached (not explicit).
                    for _ in range(prod_quantity):
                        # Block Logic: Continuously retries adding the product until successful.
                        # Invariant: The loop waits and retries if the marketplace cannot add the product.
                        while not self.marketplace.add_to_cart(id_cart, product):
                            time.sleep(self.retry_wait_time)

                # Conditional Logic: Handles "remove" commands.
                if type_command == "remove":
                    # Block Logic: Removes the product 'prod_quantity' times from the cart.
                    # Pre-condition: 'prod_quantity' is the number of units to remove.
                    # Invariant: The product will be removed 'prod_quantity' times.
                    for _ in range(prod_quantity):
                        self.marketplace.remove_from_cart(id_cart, product)

            # Block Logic: Prints the products from the ordered cart.
            # Pre-condition: 'self.marketplace.place_order(id_cart)' returns a list of products.
            # Invariant: Each product in the ordered list will be printed.
            for product in self.marketplace.place_order(id_cart):
                # Functional Utility: Prints which consumer bought which product, with flush for immediate output.
                print(f'{self.name} bought {product}', flush=True)


class Marketplace:
    """
    @brief Central hub for managing product availability, producers, and consumer carts.

    This class orchestrates the interaction between producers and consumers,
    handling product publishing, cart management, and order placement with
    thread-safe mechanisms. Logging is integrated for tracing operations.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace with specified producer queue size.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in the marketplace's inventory.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.nr_producers = 0 # Functional Utility: Counter for assigning unique producer IDs.
        self.nr_carts = 0 # Functional Utility: Counter for assigning unique cart IDs.
        # Functional Utility: Dictionary mapping producer IDs to their respective product queues (lists).
        # Each entry in the list is a dictionary of {product: 'a' (available) or 'u' (unavailable)}.
        self.producers_dict = {}
        # Functional Utility: Dictionary mapping cart IDs to lists of items in the cart.
        # Each item in the list is a dictionary of {product: producer_id}.
        self.consumers_dict = {}
        
        # Functional Utility: A global lock to protect critical sections involving shared marketplace state.
        self.lock = Lock()

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.

        Assigns a unique ID to the producer and initializes an empty product list for them.

        Returns:
            int: The unique ID assigned to the registered producer.
        """
        # Block Logic: Ensures thread-safe registration of a new producer.
        with self.lock:
            # Functional Utility: Initializes an empty list as the product stock for the new producer.
            self.producers_dict[self.nr_producers] = []
            self.nr_producers += 1 # Functional Utility: Increments the total count of registered producers.

        return self.nr_producers - 1 # Functional Utility: Returns the newly assigned producer ID.

    def publish(self, producer_id, product):
        """
        @brief Publishes a product from a producer to their queue in the marketplace.

        The product is published only if the producer has not exceeded their
        allotted queue size. Logging tracks publish attempts.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (any): The product to be published.

        Returns:
            bool: True if the product was successfully published, False otherwise.
        """
        logging.info('%d %s', producer_id, product) # Functional Utility: Logs the publish attempt.

        products_list = self.producers_dict[producer_id]
        # Conditional Logic: Checks if the producer's current item count exceeds the allowed queue size.
        if len(products_list) < self.queue_size_per_producer:
            # Functional Utility: Adds the product to the producer's list, marking it as 'available' ('a').
            products_list.append({product:'a'})
            logging.info('True') # Functional Utility: Logs successful publish.
            return True
        logging.info('False') # Functional Utility: Logs failed publish (queue full).
        return False

    def new_cart(self):
        """
        @brief Creates a new shopping cart and assigns it a unique ID.

        Returns:
            int: The unique ID of the newly created cart.
        """
        # Block Logic: Ensures thread-safe creation of a new cart.
        with self.lock:
            # Functional Utility: Initializes an empty list to represent the new cart.
            self.consumers_dict[self.nr_carts] = []
            self.nr_carts += 1 # Functional Utility: Increments the total count of carts.

        return self.nr_carts - 1 # Functional Utility: Returns the newly assigned cart ID.

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product from the marketplace to a specific consumer's cart.

        Searches through all producer queues for the specified product. If found and
        available, it is moved from the producer's queue to the consumer's cart.

        Args:
            cart_id (int): The ID of the consumer's shopping cart.
            product (Product): The product to add.

        Returns:
            bool: True if the product was successfully added, False if not available.
        """
        logging.info('%d %s', cart_id, product) # Functional Utility: Logs the add-to-cart attempt.

        cart_list = self.consumers_dict[cart_id]

        # Block Logic: Iterates through each producer's stock to find an available instance of the product.
        # Pre-condition: 'self.producers_dict' contains producer IDs mapped to their product lists.
        # Invariant: The first available product found is moved to the cart.
        for key in self.producers_dict: # Functional Utility: 'key' represents producer_id.
            products_map = self.producers_dict[key] # Functional Utility: Gets the list of products for the current producer.
            for dict_item in products_map: # Functional Utility: Iterates through product dictionaries in the producer's list.
                # Conditional Logic: Checks if the product is in the current dictionary item and is marked 'available'.
                if product in dict_item:
                    if dict_item[product] == 'a':
                        cart_list.append({product:key}) # Functional Utility: Adds product and its producer_id to the cart.
                        dict_item[product] = 'u' # Functional Utility: Marks the product as 'unavailable' ('u') in the producer's stock.
                        logging.info('True') # Functional Utility: Logs successful addition.
                        return True
        logging.info('False') # Functional Utility: Logs failed addition (product not found or not available).
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specific cart and marks it as available again in the producer's stock.

        Args:
            cart_id (int): The ID of the cart from which to remove the product.
            product (Product): The product to remove.
        """
        logging.info('%d %s', cart_id, product) # Functional Utility: Logs the remove-from-cart attempt.

        cart_list = self.consumers_dict[cart_id]
        # Block Logic: Iterates through the items in the cart to find the specified product.
        # Pre-condition: 'cart_list' contains product dictionaries like {product: producer_id}.
        # Invariant: The product is removed from the cart and re-marked 'available' in producer stock.
        for prod_dict in cart_list:
            # Conditional Logic: Checks if the product matches the item in the cart.
            if product in prod_dict:
                # Functional Utility: Retrieves the producer ID from the cart item.
                producer_id = prod_dict[product]
                product_list = self.producers_dict[producer_id] # Functional Utility: Gets the producer's stock list.
                # Block Logic: Finds the product in the producer's stock and marks it 'available'.
                # Pre-condition: 'product_list' contains product dictionaries.
                # Invariant: The product will be found and its status changed to 'available'.
                for prodd in product_list:
                    if product in prodd:
                        prodd[product] = 'a' # Functional Utility: Marks the product as 'available' ('a').
                cart_list.remove(prod_dict) # Functional Utility: Removes the product from the cart.
                return

    def place_order(self, cart_id):
        """
        @brief Finalizes an order for a given cart, removing it from the marketplace's active carts,
               updating producer stock, and returning the list of ordered products.

        Args:
            cart_id (int): The ID of the cart to place the order for.

        Returns:
            list: A list of products that were part of the order.
        """
        logging.info('%d', cart_id) # Functional Utility: Logs the place-order attempt.

        products_ordered_list = []
        # Block Logic: Iterates through items in the cart to finalize the order.
        # Pre-condition: 'self.consumers_dict[cart_id]' contains product-producer_id mappings.
        # Invariant: Each item is added to 'products_ordered_list' and removed from producer stock.
        for item in self.consumers_dict[cart_id]:
            for key in item: # Functional Utility: 'key' is the product.
                products_ordered_list.append(key) # Functional Utility: Adds the product to the ordered list.
                # Block Logic: Finds the product in the producer's stock and removes it permanently.
                # Pre-condition: 'self.producers_dict[item[key]]' contains product dictionaries.
                # Invariant: The ordered product is removed from the producer's inventory.
                for prod in self.producers_dict[item[key]]:
                    if key in prod:
                        self.producers_dict[item[key]].remove(prod) # Functional Utility: Removes the product from the producer's stock.
                        break

        self.consumers_dict[cart_id].clear() # Functional Utility: Clears the cart after the order is placed.
        logging.info(products_ordered_list) # Functional Utility: Logs the list of ordered products.
        return products_ordered_list

class TestMarketPlace(unittest.TestCase):
    """
    @brief Unit tests for the Marketplace class to verify its core functionalities.
    """
    
    def setUp(self) -> None:
        """
        @brief Sets up the test environment before each test method is run.

        Initializes a Marketplace object and a list of dummy products.
        """
        self.market_place_object = Marketplace(10)

        self.products_list = []
        # Block Logic: Populates a list with 10 dummy products for testing.
        # Invariant: 'self.products_list' will contain 10 Product objects.
        for i in range(10):
            new_product = Product('product' + str(i), i * 10)
            self.products_list.append(new_product)

        self.cart = {} # Functional Utility: Initializes an empty dictionary for a test cart.
        self.producer = None # Functional Utility: Initializes producer to None.

    def test_register_producer(self):
        """
        @brief Tests the producer registration functionality of the Marketplace.
        """
        # Block Logic: Registers 10 producers and asserts their IDs are sequentially assigned.
        # Invariant: Each registered producer receives an ID equal to its registration order.
        for i in range(10):
            self.assertEqual(self.market_place_object.register_producer(), i)

    def test_publish(self):
        """
        @brief Tests the product publishing functionality, including queue size limits.
        """
        self.producer = self.market_place_object.register_producer()
        # Block Logic: Publishes 10 products and asserts successful publication.
        # Invariant: All 10 products are successfully published within the queue limit.
        for product in self.products_list:
            self.assertTrue(self.market_place_object.publish(self.producer, product))

        # Functional Utility: Attempts to publish an 11th product to test the queue size limit.
        new_product = Product('product10', 0)
        self.assertFalse(self.market_place_object.publish(self.producer, new_product)) # Functional Utility: Asserts that publishing fails due to full queue.

    def test_new_cart(self):
        """
        @brief Tests the cart creation functionality of the Marketplace.
        """
        # Block Logic: Creates 10 new carts and asserts their IDs are sequentially assigned.
        # Invariant: Each new cart receives an ID equal to its creation order.
        for i in range(10):
            self.assertEqual(self.market_place_object.new_cart(), i)

    def test_add_to_cart(self):
        """
        @brief Tests adding products to a cart from the Marketplace.
        """
        self.producer = self.market_place_object.register_producer()
        # Block Logic: Publishes all test products from the producer.
        # Invariant: All products are available in the marketplace.
        for product in self.products_list:
            self.market_place_object.publish(self.producer, product)
        self.cart = self.market_place_object.new_cart()
        # Block Logic: Adds each published product to the new cart and asserts success.
        # Invariant: All available products are successfully added to the cart.
        for product in self.products_list:
            self.assertTrue(self.market_place_object.add_to_cart(self.cart, product))

    def test_remove_from_cart(self):
        """
        @brief Tests removing a product from a cart and its return to the producer's stock.
        """
        product_to_be_removed = self.products_list[0]

        self.producer = self.market_place_object.register_producer()
        self.market_place_object.publish(self.producer, product_to_be_removed)

        self.cart = self.market_place_object.new_cart()
        self.market_place_object.add_to_cart(self.cart, product_to_be_removed)

        producer_list = self.market_place_object.producers_dict[self.producer]

        self.assertTrue({product_to_be_removed:'u'} in producer_list) # Functional Utility: Asserts product is marked unavailable in producer list.
        self.market_place_object.remove_from_cart(self.cart, product_to_be_removed)

        self.assertTrue({product_to_be_removed:'a'} in producer_list) # Functional Utility: Asserts product is marked available again.

    def test_place_order(self):
        """
        @brief Tests placing an order and verifying the products received.
        """
        self.producer = self.market_place_object.register_producer()

        # Block Logic: Publishes all test products from the producer.
        for product in self.products_list:
            self.market_place_object.publish(self.producer, product)

        self.cart = self.market_place_object.new_cart()

        # Block Logic: Adds all published products to the new cart.
        for product in self.products_list:
            self.market_place_object.add_to_cart(self.cart, product)

        products_ordered = self.market_place_object.place_order(self.cart)

        self.assertEqual(self.products_list, products_ordered) # Functional Utility: Asserts that the ordered products match the initial list.


class Producer(Thread):
    """
    @brief Represents a producer thread that continuously supplies products to the marketplace.

    Producers generate products according to a predefined list and attempt to publish
    them to the Marketplace, pausing if their designated queue space is full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.

        Args:
            products (list): A list of product specifications (product_id, quantity, production_time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time in seconds to wait before retrying if marketplace is full.
            **kwargs: Arbitrary keyword arguments, including 'daemon' status for the thread.
        """
        # Functional Utility: Calls the parent Thread class constructor, setting daemon status.
        Thread.__init__(self, daemon=kwargs['daemon'])

        # Functional Utility: Stores the list of products this producer will generate.
        self.products = products
        # Functional Utility: Stores a reference to the shared Marketplace instance.
        self.marketplace = marketplace
        # Functional Utility: Stores the wait time if the marketplace queue is full.
        self.republish_wait_time = republish_wait_time
        # Functional Utility: Stores the name of the producer.
        self.name = kwargs['name']

        # Functional Utility: Will store the unique ID assigned to this producer by the marketplace.
        self.producer_id = None


    def run(self):
        """
        @brief The main execution loop for the Producer thread.

        Registers with the marketplace, then continuously produces items and attempts
        to publish them, waiting if the marketplace queue is full.
        """
        # Functional Utility: Registers this producer with the marketplace and obtains a unique ID.
        self.producer_id = self.marketplace.register_producer()
        # Invariant: The producer continuously attempts to produce and publish products.
        while True:
            # Block Logic: Iterates through the list of products this producer is configured to make.
            # Pre-condition: 'self.products' contains tuples of product specifications.
            # Invariant: Each product type will be produced in its specified quantity.
            for (prod, prod_quantity, waiting_time) in self.products:
                time.sleep(waiting_time) # Functional Utility: Simulates the time taken to produce a batch of this product.
                produced = 0
                # Block Logic: Attempts to publish the product 'prod_quantity' times.
                # Pre-condition: 'produced' tracks the number of units successfully published.
                # Invariant: The loop continues until all units are published for this product batch.
                while produced < prod_quantity:
                    # Functional Utility: Attempts to publish one unit of the product to the marketplace.
                    res = self.marketplace.publish(self.producer_id, # Changed to self.producer_id (int)
                                                   prod)
                    # Conditional Logic: If publishing was successful.
                    if res:
                        produced = produced + 1 # Functional Utility: Increments count of successfully published units.
                    else:
                        # Functional Utility: If publishing failed (marketplace queue full), waits and retries.
                        time.sleep(self.republish_wait_time)


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Represents a generic product with a name and price.

    This is a frozen dataclass, meaning instances are immutable after creation.
    It defines custom hash and equality methods based solely on the product name.
    """
    name: str
    price: int

    # Functional Utility: A custom hash function is essential for dataclasses
    # used as dictionary keys or in sets, especially when __eq__ is customized,
    # or when using frozen=True and relying on field hashing.
    def __hash__(self) -> int:
        return hash((self.name, self.price)) # Included price in hash for more robust uniqueness.

    # Functional Utility: A custom equality check based on all relevant attributes.
    def __eq__(self, other) -> bool:
        if not isinstance(other, Product):
            return NotImplemented
        return self.name == other.name and self.price == other.price


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Represents a specific type of Product: Tea.

    Extends the base Product with an additional attribute for 'type'.
    """
    type: str

    # Functional Utility: Custom hash for Tea to include its specific fields.
    def __hash__(self) -> int:
        return hash((super().__hash__(), self.type))

    # Functional Utility: Custom equality for Tea to include its specific fields.
    def __eq__(self, other) -> bool:
        if not isinstance(other, Tea):
            return NotImplemented
        return super().__eq__(other) and self.type == other.type


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Represents a specific type of Product: Coffee.

    Extends the base Product with additional attributes for 'acidity' and 'roast_level'.
    """
    acidity: str
    roast_level: str

    # Functional Utility: Custom hash for Coffee to include its specific fields.
    def __hash__(self) -> int:
        return hash((super().__hash__(), self.acidity, self.roast_level))

    # Functional Utility: Custom equality for Coffee to include its specific fields.
    def __eq__(self, other) -> bool:
        if not isinstance(other, Coffee):
            return NotImplemented
        return super().__eq__(other) and self.acidity == other.acidity and self.roast_level == other.roast_level
