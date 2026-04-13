
"""
This module simulates a marketplace with a producer-consumer model.

It defines the core components of the simulation:
- Product: A dataclass representing an item to be sold.
- Producer: A thread that creates products and adds them to the marketplace.
- Consumer: A thread that browses products and buys them from the marketplace.
- Marketplace: The central class that manages products, producers, and consumers,
             handling the logic for publishing, adding to carts, and placing orders.
- TestMarketplace: A suite of unit tests to verify the functionality of the Marketplace.

The simulation uses threading to run producers and consumers concurrently and employs
locks to manage safe access to shared resources within the marketplace.
"""

from threading import Thread, Lock
from time import sleep
from dataclasses import dataclass
import logging
from logging.handlers import RotatingFileHandler
import time
import unittest
# The original file has a relative import, which is kept as-is.
# Assuming 'tema' is a package available in the execution environment.
import tema.product as p

# --- Constants ---
ADD_OPTION = "add"
NAME_ARG = "name"
QUANTITY = "quantity"
PRODUCT_ARG = "product"
TYPE = "type"


class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace.

    A consumer performs a series of actions defined in a `carts` data structure,
    such as adding items to and removing items from their cart, and finally
    placing an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        Args:
            carts: A list of shopping actions to be performed.
            marketplace: The shared Marketplace instance.
            retry_wait_time (float): Time to wait before retrying to add a product.
            **kwargs: Keyword arguments for the Thread base class, must include 'name'.
        """
        super().__init__(**kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.cons_name = kwargs[NAME_ARG]

    def run(self):
        """
        The main execution logic for the consumer thread.

        Iterates through the predefined cart actions, interacting with the
        marketplace to add or remove products. After processing a set of actions,
        it places an order and prints the items bought.
        """
        cons_id = self.marketplace.new_cart()

        for entries in self.carts:
            for cart in entries:
                for _ in range(cart[QUANTITY]):
                    # Block Logic: Add or remove an item from the cart based on the action type.
                    if cart[TYPE] == ADD_OPTION:
                        # Invariant: Loop continues until the product is successfully added to the cart.
                        # This simulates a persistent consumer who waits until an item is available.
                        while True:
                            done = self.marketplace.add_to_cart(cons_id, cart[PRODUCT_ARG])
                            if done:
                                break
                            # Inline: Wait for a specified time before retrying to avoid busy-waiting.
                            sleep(self.retry_wait_time)
                    else:
                        # Assumes the item is in the cart for the remove operation.
                        self.marketplace.remove_from_cart(cons_id, cart[PRODUCT_ARG])

            # After processing a batch of cart actions, place the order.
            groceries = self.marketplace.place_order(cons_id)

            for product in groceries:
                print(f"{self.cons_name} bought {product}")


# --- Marketplace and Supporting Classes ---

# Set up logging for the marketplace operations.
MARKETPLACE_LOGGER = logging.getLogger("market_logger")
MARKETPLACE_LOGGER.setLevel(logging.INFO)
HANDLER = RotatingFileHandler("marketplace.log", maxBytes=100000, backupCount=16)
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
FORMATTER.converter = time.gmtime
HANDLER.setFormatter(FORMATTER)
MARKETPLACE_LOGGER.addHandler(HANDLER)

TEST_QUEUE_SIZE = 3


class ProductEntry:
    """
    A wrapper class for a product in the marketplace inventory.

    It associates a product with its producer and tracks its availability state
    (i.e., whether it's available to be added to a cart or has already been claimed).
    """

    def __init__(self, product, producer_id):
        """
        Initializes a ProductEntry.

        Args:
            product: The product object.
            producer_id: The ID of the producer who published this product.
        """
        self.product = product
        self.producer_id = producer_id
        self.is_available = True

    def __str__(self):
        return f"prod = {self.product}, id = {self.producer_id}, av = {self.is_available}"


class Marketplace:
    """
    The central marketplace simulation.

    Manages the inventory of products from multiple producers and handles
    interactions from consumers (carts). It uses locks to ensure thread-safe
    operations on its shared data structures.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in the marketplace at one time.
        """
        MARKETPLACE_LOGGER.info("start __init__ with args: %d", queue_size_per_producer)

        self.queue_size = queue_size_per_producer
        self.producers = {}  # {producer_id: [ProductEntry, ...]}
        self.consumers = {}  # {cart_id: [ProductEntry, ...]}
        self.curr_prod_id = 0
        self.curr_cons_id = 0
        self.reg_prod_lock = Lock()  # Lock for registering new producers.
        self.reg_cart_lock = Lock()  # Lock for creating new carts.
        self.producer_locks = {}  # {producer_id: Lock()} for fine-grained inventory control.

        MARKETPLACE_LOGGER.info("exit __init__")

    def register_producer(self):
        """
        Atomically registers a new producer, returning a unique ID.
        """
        MARKETPLACE_LOGGER.info("start register_producer")

        # Block Logic: Atomically increment the producer ID and initialize their inventory.
        # A lock ensures that each producer receives a unique ID.
        with self.reg_prod_lock:
            new_prod_id = self.curr_prod_id
            self.producers[new_prod_id] = []
            self.producer_locks[new_prod_id] = Lock()
            self.curr_prod_id += 1

            MARKETPLACE_LOGGER.info("exit register_producer")

            return new_prod_id

    def publish(self, producer_id, product):
        """
        Adds a product to a producer's public inventory.

        Fails if the producer's inventory is already at its maximum capacity.

        Args:
            producer_id: The ID of the publishing producer.
            product: The product to publish.

        Returns:
            bool: True if the product was successfully published, False otherwise.
        """
        MARKETPLACE_LOGGER.info("start publish with args: %d; %s", producer_id, product)

        # Pre-condition: Check if the producer's queue has space.
        if len(self.producers[producer_id]) < self.queue_size:
            self.producers[producer_id].append(ProductEntry(product, producer_id))
            MARKETPLACE_LOGGER.info("exit publish")
            return True

        MARKETPLACE_LOGGER.info("exit publish")
        return False

    def new_cart(self):
        """
        Atomically creates a new shopping cart for a consumer, returning a unique ID.
        """
        MARKETPLACE_LOGGER.info("start new_cart")

        # Block Logic: Atomically increment the consumer/cart ID and initialize their cart.
        with self.reg_cart_lock:
            new_cart_id = self.curr_cons_id
            self.consumers[new_cart_id] = []
            self.curr_cons_id += 1

            MARKETPLACE_LOGGER.info("exit new_cart")
            return new_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Finds an available product and adds it to a consumer's cart.

        It iterates through all producers' inventories to find a matching and
        available product. The operation is atomic on a per-producer basis.

        Args:
            cart_id: The ID of the consumer's cart.
            product: The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        MARKETPLACE_LOGGER.info("start add_to_cart with args: %id; %s", cart_id, product)

        # Block Logic: Iterate through all producers to find the requested product.
        for id_producer, product_entries in self.producers.items():
            # Invariant: A fine-grained lock is held for each producer's inventory
            # while it is being checked and modified. This prevents race conditions
            # where two consumers might grab the same item.
            with self.producer_locks[id_producer]:
                for product_entry in product_entries:
                    if product == product_entry.product and product_entry.is_available:
                        # Mark the product as unavailable and add it to the consumer's cart.
                        product_entry.is_available = False
                        self.consumers[cart_id].append(product_entry)
                        return True

        MARKETPLACE_LOGGER.info("exit add_to_cart")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a consumer's cart, making it available again.

        Args:
            cart_id: The ID of the consumer's cart.
            product: The product to remove.
        """
        MARKETPLACE_LOGGER.info("start remove_from_cart with args: %d; %s", cart_id, product)
        to_remove = None

        # Block Logic: Find the specified product within the consumer's cart.
        for product_entry in self.consumers[cart_id]:
            if product == product_entry.product:
                # Block Logic: Atomically update the product's state.
                # A lock on the original producer's inventory is required to safely
                # mark the item as available again.
                with self.producer_locks[product_entry.producer_id]:
                    to_remove = product_entry
                    product_entry.is_available = True
                    break
        
        if to_remove:
            self.consumers[cart_id].remove(to_remove)

        MARKETPLACE_LOGGER.info("exit remove_from_cart")

    def place_order(self, cart_id):
        """
        Finalizes an order, permanently removing items from the marketplace.

        Args:
            cart_id: The ID of the cart to be ordered.

        Returns:
            list: A list of product objects that were in the cart.
        """
        MARKETPLACE_LOGGER.info("start place_order with args: %d", cart_id)
        groceries = []

        # Block Logic: Iterate through the items in the cart to finalize the purchase.
        for product_entry in self.consumers[cart_id]:
            # Block Logic: Lock the producer's inventory to safely remove the product.
            # This ensures that the product is permanently removed from the system.
            with self.producer_locks[product_entry.producer_id]:
                groceries.append(product_entry.product)
                self.producers[product_entry.producer_id].remove(product_entry)

        # Clear the consumer's cart after the order is placed.
        self.consumers[cart_id].clear()

        MARKETPLACE_LOGGER.info("exit place_order")
        return groceries


class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace class.
    
    These tests verify the core functionalities of the marketplace, including
    producer/consumer registration, publishing products, cart operations,
    and order placement, ensuring thread-safety and logical correctness.
    """

    def setUp(self):
        """Initializes a test environment before each test."""
        self.marketplace = Marketplace(TEST_QUEUE_SIZE)
        self.coffee_product_0 = p.Coffee("coffee0", 0, "0", "l")
        self.coffee_product_1 = p.Coffee("coffee1", 1, "1", "m")
        self.coffee_product_2 = p.Coffee("coffee2", 2, "2", "h")
        self.tea_product_0 = p.Tea("tea0", 3, "t0")
        self.tea_product_1 = p.Tea("tea1", 4, "t1")
        self.producer_0 = self.marketplace.register_producer()
        self.producer_1 = self.marketplace.register_producer()
        self.consumer_0 = self.marketplace.new_cart()
        self.consumer_1 = self.marketplace.new_cart()

    def test_register_producer(self):
        """Tests that producer registration yields unique, sequential IDs."""
        self.assertEqual(0, self.producer_0, "wrong id for producer_0")
        self.assertEqual(1, self.producer_1, "wrong id for producer_1")

        id_test = []
        for _ in range(1024):
            id_test.append(self.marketplace.register_producer())

        for i in range(1024):
            self.assertEqual(i + 2, id_test[i], f"wrong id: got {id_test[i]}, should be {i + 2}")

    def test_publish(self):
        """Tests that products can be published and that the queue size limit is enforced."""
        # Test successful publications.
        result = self.marketplace.publish(self.producer_0, self.coffee_product_0)
        self.assertEqual(result, True, "should be True")
        self.assertEqual(self.marketplace.producers[self.producer_0][0].product,
                         self.coffee_product_0,
                         "wrong product")

        result = self.marketplace.publish(self.producer_0, self.coffee_product_1)
        self.assertEqual(result, True, "should be True")
        self.assertEqual(self.marketplace.producers[self.producer_0][1].product,
                         self.coffee_product_1,
                         "wrong product")

        result = self.marketplace.publish(self.producer_0, self.coffee_product_2)
        self.assertEqual(result, True, "should be True")
        self.assertEqual(self.marketplace.producers[self.producer_0][2].product,
                         self.coffee_product_2,
                         "wrong product")

        # Test that publishing fails when the queue is full.
        result = self.marketplace.publish(self.producer_0, self.tea_product_0)
        self.assertEqual(result, False, "should be False")

    def test_new_cart(self):
        """Tests that new carts are created with unique, sequential IDs."""
        self.assertEqual(0, self.consumer_0, "wrong id for consumer_0")
        self.assertEqual(1, self.consumer_1, "wrong id for consumer_1")

        id_test = []
        for _ in range(1024):
            id_test.append(self.marketplace.new_cart())

        for i in range(1024):
            self.assertEqual(i + 2, id_test[i], f"wrong id: got {id_test[i]}, should be {i + 2}")

    def test_add_to_cart(self):
        """Tests adding available and unavailable items to a cart."""
        # Test adding an available item.
        self.marketplace.publish(self.producer_1, self.tea_product_1)
        result = self.marketplace.add_to_cart(self.consumer_0, self.tea_product_1)
        self.assertEqual(result, True, "should be True")

        # Test adding a non-existent item.
        result = self.marketplace.add_to_cart(self.consumer_0, self.coffee_product_0)
        self.assertEqual(result, False, "should be False")

        # Test adding an already claimed item.
        result = self.marketplace.add_to_cart(self.consumer_1, self.tea_product_1)
        self.assertEqual(result, False, "should be False")

        # Test adding multiple items to a cart.
        self.marketplace.publish(self.producer_0, self.coffee_product_0)
        self.marketplace.publish(self.producer_1, self.coffee_product_1)
        self.marketplace.publish(self.producer_1, self.coffee_product_2)
        result = self.marketplace.add_to_cart(self.consumer_1, self.coffee_product_0)
        self.assertEqual(result, True, "should be True")
        result = self.marketplace.add_to_cart(self.consumer_1, self.coffee_product_2)
        self.assertEqual(result, True, "should be True")
        result = self.marketplace.add_to_cart(self.consumer_1, self.coffee_product_1)
        self.assertEqual(result, True, "should be True")
        self.assertEqual(3,
                         len(self.marketplace.consumers[self.consumer_1]),
                         "wrong number of products in cart")

    def test_remove_from_cart(self):
        """Tests that items can be removed from a cart and made available again."""
        # Test basic removal.
        self.marketplace.publish(self.producer_0, self.coffee_product_0)
        self.marketplace.add_to_cart(self.consumer_0, self.coffee_product_0)
        self.marketplace.remove_from_cart(self.consumer_0, self.coffee_product_0)
        self.assertEqual(0,
                         len(self.marketplace.consumers[self.consumer_0]),
                         "wrong number of products in cart")

        # Test that a removed item becomes available for another consumer.
        self.marketplace.publish(self.producer_1, self.tea_product_0)
        self.marketplace.add_to_cart(self.consumer_0, self.tea_product_0)
        result = self.marketplace.add_to_cart(self.consumer_1, self.tea_product_0)
        self.assertEqual(result, False, "should be False")
        self.marketplace.remove_from_cart(self.consumer_0, self.tea_product_0)
        result = self.marketplace.add_to_cart(self.consumer_1, self.tea_product_0)
        self.assertEqual(result, True, "should be True")

    def test_place_order(self):
        """Tests the entire workflow from publishing to placing an order."""
        self.marketplace.publish(self.producer_0, self.tea_product_0)
        self.marketplace.publish(self.producer_1, self.coffee_product_1)
        self.marketplace.publish(self.producer_1, self.coffee_product_2)
        self.marketplace.publish(self.producer_0, self.coffee_product_2)
        self.marketplace.publish(self.producer_1, self.tea_product_1)
        self.marketplace.add_to_cart(self.consumer_0, self.coffee_product_2)
        self.marketplace.add_to_cart(self.consumer_0, self.coffee_product_2)
        self.marketplace.add_to_cart(self.consumer_0, self.coffee_product_1)
        self.marketplace.add_to_cart(self.consumer_0, self.tea_product_1)
        self.marketplace.remove_from_cart(self.consumer_0, self.coffee_product_1)
        self.marketplace.add_to_cart(self.consumer_0, self.tea_product_0)

        # Verify cart contents before ordering.
        self.assertEqual(4,
                         len(self.marketplace.consumers[self.consumer_0]),
                         "wrong number of products in cart")

        groceries_list = self.marketplace.place_order(self.consumer_0)

        # Verify the final list of purchased items.
        self.assertEqual(groceries_list,
                         [self.coffee_product_2, self.coffee_product_2,
                          self.tea_product_1, self.tea_product_0],
                         "wrong products in cart")

        # Verify the cart is empty after the order.
        self.assertEqual(0, len(self.marketplace.producers[self.consumer_0]), "should be empty")


# --- Producer Class ---

# Constants for the Producer's product list structure.
PRODUCTS = 0
QUANTITY = 1
PRODUCE_TIME = 2


class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer instance.

        Args:
            products: A list of products that the producer will create.
            marketplace: The shared Marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish.
            **kwargs: Keyword arguments for the Thread base class.
        """
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        The main execution logic for the producer thread.

        Registers with the marketplace and then enters an infinite loop to
        continuously produce and publish its assigned products.
        """
        prod_id = self.marketplace.register_producer()

        # This infinite loop simulates a producer that continuously restocks items.
        while True:
            for product in self.products:
                for _ in range(product[QUANTITY]):
                    # Invariant: Loop until the product is successfully published.
                    # This ensures the producer will eventually publish its product,
                    # waiting if its inventory space in the marketplace is full.
                    while True:
                        done = self.marketplace.publish(prod_id, product[PRODUCTS])
                        if done:
                            break
                        # Inline: Wait before retrying to avoid busy-waiting.
                        sleep(self.republish_wait_time)

                    # Simulate the time it takes to produce the item.
                    sleep(product[PRODUCE_TIME])


# --- Product Dataclasses ---

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a product with a name and price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass representing Coffee, inheriting from Product."""
    acidity: str
    roast_level: str
