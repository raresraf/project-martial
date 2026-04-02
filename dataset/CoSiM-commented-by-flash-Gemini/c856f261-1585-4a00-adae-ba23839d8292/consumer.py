"""
This module implements a multi-threaded marketplace simulation, including producer and consumer logic,
a central marketplace for product management, and unit tests for the marketplace functionality.
It demonstrates concurrent operations using Python's `threading` module and utilizes locks for
synchronization to ensure thread safety in shared data structures.
"""

from threading import Thread, Lock
from time import sleep
import time
import unittest
import logging
import logging.handlers
from dataclasses import dataclass


# --- Logging Configuration ---
LOGGER = logging.getLogger("marketlogger")
LOGGER.setLevel(logging.INFO)

# Configures a rotating file handler for logging marketplace events.
# Logs are written to 'marketplace.log', with a maximum size of 20KB and 5 backup files.
HANDLER = logging.handlers.RotatingFileHandler(
    "marketplace.log", maxBytes=20000, backupCount=5)

# Defines the format for log messages, including timestamp and message content.
FORMATTER = logging.Formatter("%(asctime)s;%(message)s")
HANDLER.setFormatter(FORMATTER)
logging.Formatter.converter = time.gmtime
LOGGER.addHandler(HANDLER)


# --- Global Constants for Producer ---
# Represents the index of the product name within the product definition tuple.
PRODUCT_POS = 0
# Represents the index of the number of products to publish within the product definition tuple.
NUMBER_OF_PRODUCTS_POS = 1
# Represents the index of the waiting time before republishing within the product definition tuple.
WAITING_TIME_POS = 2

# Defines the maximum number of products a single producer can publish to the marketplace.
QUEUE_SIZE = 3


# --- Data Classes for Products ---

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Represents a generic product with a name and price.
    This is an immutable data class.

    Attributes:
        name (str): The name of the product.
        price (int): The price of the product.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Represents a specific type of product: Tea.
    Inherits from Product and adds a 'type' attribute.
    This is an immutable data class.

    Attributes:
        type (str): The specific type of tea (e.g., "Green", "Black").
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Represents a specific type of product: Coffee.
    Inherits from Product and adds 'acidity' and 'roast_level' attributes.
    This is an immutable data class.

    Attributes:
        acidity (str): The acidity level of the coffee.
        roast_level (str): The roast level of the coffee.
    """
    acidity: str
    roast_level: str


# --- Consumer Class ---

class Consumer(Thread):
    """
    Represents a consumer entity in the marketplace simulation.
    Consumers attempt to add products to their carts and place orders.
    Each consumer runs as a separate thread.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a new Consumer instance.

        Args:
            carts (list): A list of cart configurations, where each configuration
                          is a list of commands (add/remove product).
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): The time (in seconds) to wait before retrying
                                     an 'add_to_cart' operation if it initially fails.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor,
                      e.g., 'name' for thread identification.
        """
        self.carts = carts
        self.marketplace = marketplace
        self.wait_time = retry_wait_time
        self.curr_cart = []  # Not actively used in the provided run method
        self.cart_id = -1    # Not actively used in the provided run method
        self.kwargs = kwargs # Stores additional arguments like thread name
        super().__init__(**kwargs) # Correct way to call parent constructor


    def print_cart(self, cart):
        """
        Prints the contents of a shopping cart, indicating which products the consumer bought.
        Ensures thread-safe printing using a lock obtained from the marketplace.

        Args:
            cart (dict): A dictionary representing the cart, where keys are products
                         and values are lists of producer IDs.
        """
        # Acquire a lock to ensure that print statements from different threads do not interleave.
        lock = self.marketplace.get_print_lock()
        lock.acquire()
        # Iterate through each product and its associated producer IDs in the cart.
        for prod in cart:
            # For each instance of a product (identified by its producer ID).
            for _ in range(len(cart[prod])):
                # Print a message indicating the consumer (by name) bought the product.
                print(self.kwargs['name'] + " bought " + str(prod))
        # Release the lock after printing is complete.
        lock.release()


    def add_to_cart(self, product, cart_id):
        """
        Attempts to add a product to a specific cart.
        Retries the operation if it fails (e.g., product not available) after a wait period.

        Args:
            product (Product): The product to add to the cart.
            cart_id (int): The ID of the cart to which the product should be added.
        """
        res = False
        # Keep retrying until the product is successfully added to the cart.
        while res is False:
            # Attempt to add the product to the cart via the marketplace.
            res = self.marketplace.add_to_cart(cart_id, product)
            # If the addition was not successful.
            if res is False:
                # Wait for a predefined time before retrying to avoid busy-waiting.
                sleep(self.wait_time)


    def run(self):
        """
        The main execution method for the consumer thread.
        It iterates through a list of predefined carts, creates a new cart in the marketplace,
        performs add/remove operations for products, and finally places the order and prints the cart.
        """
        # Iterate over each cart configuration defined for this consumer.
        for cart_config in self.carts:
            # Request a new cart ID from the marketplace for the current shopping session.
            cart_id = self.marketplace.new_cart()
            # Process each command (add/remove) within the current cart configuration.
            for cmd in cart_config:
                cmd_type = cmd['type']
                product = cmd['product']
                quantity = cmd['quantity']
                # Perform the add or remove operation for the specified quantity of the product.
                for _ in range(0, quantity):
                    # If the command type is "add".
                    if cmd_type == "add":
                        # Attempt to add the product to the cart, retrying if necessary.
                        self.add_to_cart(product, cart_id)
                    # Otherwise (if the command type is "remove").
                    else:
                        # Remove the product from the cart directly.
                        self.marketplace.remove_from_cart(cart_id, product)
            # After processing all commands for the cart, place the order.
            # The marketplace returns the final state of the cart.
            final_cart = self.marketplace.place_order(cart_id)
            # Print the contents of the finalized order.
            self.print_cart(final_cart)


# --- Marketplace Class ---

class Marketplace:
    """
    Represents the central marketplace where producers publish products and consumers place orders.
    It manages product inventory, producer queues, shopping carts, and ensures thread safety
    using multiple locks for different resources.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes a new Marketplace instance.

        Args:
            queue_size_per_producer (int): The maximum number of products a single producer
                                           can have in the marketplace's inventory at any time.
        """
        self.queue_size = queue_size_per_producer
        self.producer_id_indexer = 0
        self.cart_id_indexer = 0
        # Stores the current count of published products for each producer.
        self.producers_dict = {}
        # Stores all available products, mapped to a list of producer IDs that supplied them.
        self.all_products = {}
        # Lock for protecting producer-related data (producer_id_indexer, producers_dict, all_products).
        self.lock = Lock()
        # Lock for protecting cart-related data (cart_id_indexer, carts).
        self.cart_lock = Lock()
        # Lock for ensuring exclusive access to print statements.
        self.print_lock = Lock()
        # Stores all active shopping carts, mapped by cart ID.
        self.carts = {}

    def get_print_lock(self):
        """
        Returns the lock used for thread-safe printing.

        Returns:
            threading.Lock: The lock instance for printing.
        """
        LOGGER.info("print_lock returned")
        return self.print_lock

    def register_producer(self):
        """
        Registers a new producer with the marketplace, assigning a unique ID.
        Ensures thread safety during ID assignment and producer dictionary update.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        LOGGER.info(" (register) started")
        # Acquire the main lock to protect shared producer indexing and dictionary.
        self.lock.acquire()
        # Assign a new producer ID and increment the indexer.
        new_id = self.producer_id_indexer
        self.producer_id_indexer += 1
        # Initialize the product count for the new producer.
        self.producers_dict[new_id] = 0
        # Release the lock.
        self.lock.release()
        LOGGER.info("(register) id %d", new_id)
        return new_id

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace by a given producer.
        Checks if the producer has reached their queue size limit.
        Ensures thread safety when updating product inventory.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Product): The product to be published.

        Returns:
            bool: True if the product was successfully published, False otherwise.
        """
        LOGGER.info("(publish) %d, %s", producer_id, str(product))
        # Acquire the main lock to protect shared product inventory and producer product counts.
        self.lock.acquire()
        # Pre-condition: Check if the producer has reached their maximum allowed published items.
        if self.producers_dict[producer_id] == self.queue_size:
            self.lock.release()
            LOGGER.info(" (publish) producer has too many items")
            # Invariant: If the limit is reached, no product is published.
            return False

        # Increment the count of products published by this producer.
        self.producers_dict[producer_id] += 1

        # If the product is new to the marketplace.
        if product not in self.all_products:
            # Initialize its entry with the current producer's ID.
            self.all_products[product] = [producer_id]
        # If the product already exists.
        else:
            # Add the current producer's ID to the list of suppliers for this product.
            self.all_products[product].append(producer_id)

        self.lock.release()
        LOGGER.info(" (publish) the product was published")
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart and assigns it a unique ID.
        Ensures thread safety during cart ID assignment and cart dictionary update.

        Returns:
            int: The unique ID of the newly created cart.
        """
        LOGGER.info("(new cart) started")
        # Acquire the cart lock to protect shared cart indexing and dictionary.
        self.cart_lock.acquire()

        # Assign a new cart ID and increment the indexer.
        new_id = self.cart_id_indexer
        self.cart_id_indexer += 1
        # Initialize an empty dictionary for the new cart.
        self.carts[new_id] = {}

        self.cart_lock.release()
        LOGGER.info(" (new_cart) created cart with id %d", new_id)
        return new_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specific shopping cart.
        Removes one instance of the product from the marketplace's available inventory.
        Ensures thread safety when modifying global product inventory and a specific cart.

        Args:
            cart_id (int): The ID of the cart to which the product should be added.
            product (Product): The product to add.

        Returns:
            bool: True if the product was successfully added, False if the product
                  is not available in the marketplace.
        """
        LOGGER.info(" (add_to_cart) params %d, %s", cart_id, str(product))
        # Acquire the main lock first, as it covers the global product inventory (`self.all_products`).
        self.lock.acquire()
        # Pre-condition: Check if the product is available in the marketplace.
        if product not in self.all_products or len(self.all_products[product]) == 0:
            LOGGER.info(
                "(add_to_cart) no product %s is published", str(product))
            self.lock.release()
            # Invariant: If product is not available, cart remains unchanged.
            return False

        # Remove one instance of the product from the marketplace's inventory.
        # This also retrieves the producer ID for later accounting.
        producer_id = self.all_products[product].pop(0)

        # Acquire the cart lock to protect the specific cart being modified.
        self.cart_lock.acquire()

        # If the product is not yet in this specific cart.
        if product not in self.carts[cart_id]:
            # Initialize its entry with the producer ID.
            self.carts[cart_id][product] = [producer_id]
        # If the product is already in the cart.
        else:
            # Add the producer ID to the list of suppliers for this product in the cart.
            self.carts[cart_id][product].append(producer_id)

        self.lock.release()
        self.cart_lock.release()
        LOGGER.info(" (add_to_cart) product %s was added", str(product))
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specific shopping cart and returns it to the marketplace's inventory.
        Ensures thread safety during these operations.

        Args:
            cart_id (int): The ID of the cart from which the product should be removed.
            product (Product): The product to remove.
        """
        LOGGER.info('(remove from cart) params %s %d', str(product), cart_id)
        # Pre-condition: Check if the product exists in the cart and there are instances to remove.
        if product in self.carts[cart_id] and len(self.carts[cart_id][product]) != 0:
            # Acquire both locks as both cart and global product inventory are modified.
            self.cart_lock.acquire()
            self.lock.acquire()
            # Remove one instance of the product from the cart and retrieve its producer ID.
            producer_id = self.carts[cart_id][product].pop(0)
            # Return the product instance to the marketplace's available inventory.
            self.all_products[product].append(producer_id)
            self.cart_lock.release()
            self.lock.release()
        LOGGER.info(" (remove_from_cart) finished %s was removed %d",
                    str(product), cart_id)

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.
        Decrements the count of products from each producer based on the items in the cart.
        Ensures thread safety during producer product count updates.

        Args:
            cart_id (int): The ID of the cart for which the order is being placed.

        Returns:
            dict: The contents of the cart that was placed as an order.
        """
        LOGGER.info(" (place order) param %d", cart_id)
        # Acquire the main lock to protect the `producers_dict` which tracks producer's active products.
        self.lock.acquire()

        products = self.carts[cart_id]
        # For each product in the cart and its associated producer IDs.
        for _, ids in products.items():
            # For each producer who supplied an item in this order.
            for producer_id in ids:
                # Decrement the count of active products for that producer.
                self.producers_dict[producer_id] -= 1
        self.lock.release()
        LOGGER.info(" (place order) %d was placed", cart_id)
        # Invariant: The marketplace state reflects the reduction in producer inventory.
        return self.carts[cart_id]


# --- TestMarketplace Class ---

class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace class, verifying its core functionalities
    like producer registration, product publishing, cart management, and order placement.
    """

    def publish_products(self, prod_id, number_of_products):
        """
        Helper method to publish a specified number of generic products for a given producer.

        Args:
            prod_id (int): The ID of the producer.
            number_of_products (int): The quantity of products to publish.
        """
        for i in range(number_of_products):
            self.marketplace.publish(prod_id, str("prod_" + str(i)))

    def setUp(self):
        """
        Set up method called before each test to initialize a fresh Marketplace instance.
        """
        self.marketplace = Marketplace(QUEUE_SIZE)

    def test_register_producer(self):
        """
        Tests the `register_producer` method to ensure unique producer IDs are assigned sequentially.
        """
        val = self.marketplace.register_producer()
        self.assertEqual(val, 0, "First producer ID should be 0")
        val = self.marketplace.register_producer()
        self.assertEqual(val, 1, "Second producer ID should be 1")

    def test_publish(self):
        """
        Tests the `publish` method, including successful publication and the queue size limit.
        """
        prod_id = self.marketplace.register_producer()
        # Publish products up to the QUEUE_SIZE limit.
        for i in range(QUEUE_SIZE):
            val = self.marketplace.publish(prod_id, str("prod_" + str(i)))
            self.assertEqual(val, True, f"Publishing prod_{i} should succeed")
        # Attempt to publish beyond the QUEUE_SIZE limit.
        val = self.marketplace.publish(prod_id, "some_product")
        self.assertEqual(val, False, "Publishing beyond queue size should fail")

    def test_publish_with_2_publishers(self):
        """
        Tests publishing with multiple producers to ensure products are tracked correctly
        by different suppliers.
        """
        id_1 = self.marketplace.register_producer()
        id_2 = self.marketplace.register_producer()

        # Both producers publish the same product.
        self.marketplace.publish(id_1, "prod")
        self.marketplace.publish(id_2, "prod")
        # Check that both producer IDs are associated with the "prod" product.
        no_of_producers = len(self.marketplace.all_products["prod"])
        self.assertEqual(no_of_producers, 2, "Expected 2 producers for 'prod'")

    def test_new_cart(self):
        """
        Tests the `new_cart` method to ensure unique cart IDs are assigned sequentially.
        """
        val = self.marketplace.new_cart()
        self.assertEqual(val, 0, "First cart ID should be 0")
        val = self.marketplace.new_cart()
        self.assertEqual(val, 1, "Second cart ID should be 1")

    def test_add_to_cart(self):
        """
        Tests the `add_to_cart` method for successful additions and attempts to add
        non-existent products.
        """
        prod_id = self.marketplace.register_producer()
        self.publish_products(prod_id, 2) # Publish "prod_0", "prod_1"
        cart_id = self.marketplace.new_cart()

        # Test adding an available product.
        val = self.marketplace.add_to_cart(cart_id, "prod_1")
        self.assertEqual(val, True, "Adding an available product should succeed")
        # Test adding a non-existent product.
        val = self.marketplace.add_to_cart(cart_id, "non_existent_prod")
        self.assertEqual(val, False, "Adding a non-existent product should fail")
        # Verify that the correct producer ID is associated with the added product in the cart.
        returned_prod_id = self.marketplace.carts[cart_id]["prod_1"][0]
        self.assertEqual(returned_prod_id, prod_id, "Producer ID in cart should match original producer")

    def test_add_to_cart_same_prod_twice(self):
        """
        Tests adding the same product multiple times from different producers to a single cart.
        """
        id1 = self.marketplace.register_producer()
        id2 = self.marketplace.register_producer()

        # Both producers publish the same product.
        self.marketplace.publish(id1, "prod")
        self.marketplace.publish(id2, "prod")
        cart_id = self.marketplace.new_cart()
        # Add the product twice to the cart.
        self.marketplace.add_to_cart(cart_id, "prod")
        self.marketplace.add_to_cart(cart_id, "prod")
        # Verify that two instances of "prod" are now in the cart.
        no_of_producers = len(self.marketplace.carts[cart_id]["prod"])
        self.assertEqual(no_of_producers, 2, "Expected 2 instances of 'prod' in cart")

    def test_remove_from_cart(self):
        """
        Tests the `remove_from_cart` method to ensure products are correctly removed
        and returned to the marketplace inventory.
        """
        prod_id = self.marketplace.register_producer()
        self.publish_products(prod_id, 2) # Publish "prod_0", "prod_1"
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "prod_1")

        # Before removal, there should be one "prod_1" in the cart.
        self.assertEqual(len(self.marketplace.carts[cart_id]["prod_1"]), 1)
        # Remove the product.
        self.marketplace.remove_from_cart(cart_id, "prod_1")

        # After removal, the list for "prod_1" in the cart should be empty.
        self.assertTrue(len(self.marketplace.carts[cart_id]["prod_1"]) == 0,
                        "Product 'prod_1' should be removed from cart")
        # The product should also be back in the marketplace's all_products, but its original
        # producer count will not be incremented until order is placed.
        self.assertTrue(prod_id in self.marketplace.all_products["prod_1"])


    def test_place_order(self):
        """
        Tests the `place_order` method to verify that product counts for producers
        are correctly decremented after an order is placed.
        """
        prod_id = self.marketplace.register_producer()
        self.publish_products(prod_id, 3) # Publish "prod_0", "prod_1", "prod_2"

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "prod_1")
        self.marketplace.add_to_cart(cart_id, "prod_0")
        self.marketplace.add_to_cart(cart_id, "prod_2")
        self.marketplace.remove_from_cart(cart_id, "prod_1") # Remove one instance, returning it to inventory

        # Before placing order, `producers_dict[prod_id]` should be 3 (all published products).
        # One "prod_1" was removed, meaning it's back in marketplace inventory, but producer still
        # "owns" it. The decrements happen only upon placing an order.

        products_in_order = self.marketplace.place_order(cart_id)
        # Verify the contents of the returned order.
        self.assertTrue(len(products_in_order["prod_0"]) == 1, "Expected 1 'prod_0' in placed order")
        self.assertTrue(len(products_in_order["prod_1"]) == 0, "Expected 0 'prod_1' in placed order (it was removed)")
        self.assertTrue(len(products_in_order["prod_2"]) == 1, "Expected 1 'prod_2' in placed order")
        # Verify that the producer's product count has been decremented correctly.
        # Two products were ultimately ordered (prod_0, prod_2), so count should be 3 - 2 = 1.
        number_of_products_of_producer = self.marketplace.producers_dict[prod_id]
        self.assertEqual(number_of_products_of_producer, 1, "Producer's product count should be 1 after order")


    def test_get_print_lock(self):
        """
        Tests the `get_print_lock` method to ensure it returns a valid Lock object.
        """
        lock = self.marketplace.get_print_lock()
        self.assertTrue(isinstance(lock, type(Lock())), "get_print_lock should return a Lock instance")


# --- Producer Class ---

class Producer(Thread):
    """
    Represents a producer entity in the marketplace simulation.
    Producers register with the marketplace and continuously publish products.
    Each producer runs as a separate thread.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a new Producer instance.

        Args:
            products (list): A list of products to be published by this producer.
                             Each product entry is a tuple: (product_object, number_of_products, wait_time_before_next_publish).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): The time (in seconds) to wait before retrying
                                         a 'publish' operation if it initially fails
                                         (e.g., due to marketplace queue limits).
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor,
                      e.g., 'name' for thread identification.
        """
        self.products = products
        self.marketplace = marketplace
        self.wait_time = republish_wait_time
        super().__init__(**kwargs) # Correct way to call parent constructor


    def run(self):
        """
        The main execution method for the producer thread.
        It registers with the marketplace, then enters an infinite loop to
        continuously publish its defined products, respecting queue limits and wait times.
        """
        # Register the producer with the marketplace to get a unique producer ID.
        prod_id = self.marketplace.register_producer()
        # The producer continuously tries to publish products.
        while True:
            # Iterate through each product defined for this producer.
            for prod_def in self.products:
                product = prod_def[PRODUCT_POS]
                no_prods = prod_def[NUMBER_OF_PRODUCTS_POS]
                pause_time = prod_def[WAITING_TIME_POS]
                # Attempt to publish the specified number of instances for the current product.
                for _ in range(0, no_prods):
                    res = False
                    # Keep retrying to publish until successful.
                    while res is False:
                        # Attempt to publish the product to the marketplace.
                        res = self.marketplace.publish(prod_id, product)
                        # If publication was not successful (e.g., marketplace queue full).
                        if res is False:
                            # Wait for a predefined time before retrying.
                            sleep(self.wait_time)
                    # Pause for a specified time after successfully publishing an instance of a product.
                    sleep(pause_time)
