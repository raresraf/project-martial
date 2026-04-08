"""
This module implements a multi-threaded producer-consumer marketplace simulation.

It contains four main classes:
- Marketplace: A central, thread-safe class that manages inventory and carts.
- Producer: A thread that adds products to the marketplace.
- Consumer: A thread that simulates customer actions like adding to a cart and placing orders.
- TestMarketplace: A unittest suite for verifying the functionality.

The simulation uses locks to ensure data consistency in a concurrent environment.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer thread that simulates a user's shopping process.

    Each consumer is initialized with a set of carts, each containing a list of
    actions (add/remove products). The consumer processes these actions against
    the shared marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping carts, where each cart is a list of actions.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying a failed action.
            **kwargs: Keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """The main execution loop for the consumer thread.

        Iterates through each assigned cart, creates it in the marketplace,
        processes all add/remove actions, and finally places the order.
        """
        # Invariant: Process each shopping cart scenario sequentially.
        for i in self.carts:
            cart_id = self.marketplace.new_cart()

            # Invariant: Process each action within a cart.
            for j in i:
                if j['type'] == "add":
                    # Block Logic: Attempt to add the specified quantity of a product to the cart.
                    for elem in range(j['quantity']):
                        add_val = self.marketplace.add_to_cart(cart_id, j['product'])
                        # Pre-condition: If adding the product fails (e.g., out of stock),
                        # wait and retry until successful.
                        while add_val is False:
                            sleep(self.retry_wait_time)
                            add_val = self.marketplace.add_to_cart(cart_id, j['product'])
                elif j['type'] == "remove":
                    # Block Logic: Remove the specified quantity of a product from the cart.
                    for elem in range(j['quantity']):
                        self.marketplace.remove_from_cart(cart_id, j['product'])

            # Finalize the transaction for the current cart.
            self.marketplace.place_order(cart_id)

# --- Start of Marketplace and Testing section ---
# Note: This appears to be a separate file concatenated with the Consumer class.

import logging
from logging.handlers import RotatingFileHandler
import threading
import time
import unittest
from threading import Lock
# These imports suggest the classes were originally in a 'tema' package.
from tema.consumer import Consumer
from tema.producer import Producer

# --- Logging Setup ---
logging.basicConfig(filename="marketplace.log", level=logging.INFO)

LOGGER = logging.getLogger('mktplace_logger')
LOGGER.setLevel(logging.INFO)

HANDLER = RotatingFileHandler('marketplace.log', maxBytes=7500000, backupCount=10)
LOGGER.addHandler(HANDLER)

logging.Formatter.converter = time.gmtime()


class Marketplace:
    """
    A thread-safe marketplace that manages producers, products, and customer carts.

    This class acts as the central shared resource, using locks to coordinate
    concurrent access from multiple producer and consumer threads.
    """
    def __init__(self, queue_size_per_producer):
        """Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products each
                producer can have in their published queue at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer

        self.id_for_producer = 0
        self.id_for_cart = 0
        self.products_queue = []  # A list of lists, one for each producer's products.
        self.mark_booked_products = []  # Parallel structure to track if a product is in a cart.
        self.list_of_carts = []  # A list of all active shopping carts.

        # --- Synchronization Primitives ---
        self.lock_for_id = Lock() # Protects producer ID generation.
        self.lock_for_cart_id = Lock() # Protects cart ID generation.
        self.lock_for_add_cart = Lock() # Protects adding/removing from carts and product availability.
        self.lock_for_publish = Lock() # Protects producer queues during publishing and order placement.

        LOGGER.info("Constructor with queue_size_per_producer %s", queue_size_per_producer)

    def register_producer(self):
        """
        Allocates a unique ID for a new producer and sets up their inventory space.

        Returns:
            int: The unique ID for the registered producer.
        """
        LOGGER.info("Register producer, asking for an id")
        with self.lock_for_id:
            temp_id = self.id_for_producer
            # Create a new product queue and booking status list for this producer.
            self.products_queue.append([])
            self.mark_booked_products.append([])
            self.id_for_producer += 1
        LOGGER.info("Register producer, he will get the id %s", self.id_for_producer)

        return temp_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        The operation is thread-safe and respects the producer's queue size limit.

        Args:
            producer_id (int): The ID of the producer.
            product: The product to be published.

        Returns:
            bool: True if the product was published successfully, False if the
                  producer's queue was full.
        """
        LOGGER.info("Producer with id %s wants to publish the following product: %s"
                    , producer_id, product)
        
        with self.lock_for_publish:
            # Pre-condition: Check if there is space in the producer's queue.
            if (len(self.products_queue[producer_id]) + 1) <= self.queue_size_per_producer:
                self.products_queue[producer_id].append(product)
                self.mark_booked_products[producer_id].append(0) # 0 means available.
                LOGGER.info("Producer published the product!")
                return True
            else:
                LOGGER.info("Producer did not publish the product!")
                return False


    def new_cart(self):
        """
        Creates a new shopping cart and returns its unique ID.

        Returns:
            int: The unique ID for the new cart.
        """
        LOGGER.info("A new cart is created for a consumer")
        with self.lock_for_cart_id:
            temp_id = self.id_for_cart
            self.list_of_carts.append(list([]))
            self.id_for_cart += 1
        LOGGER.info("The cart was created and has the id %s", temp_id)

        return temp_id

    def add_to_cart(self, cart_id, product):
        """
        Adds an available product to a specified shopping cart.

        This method searches through all producer inventories for an available
        (not booked) item of the requested product type. If found, it marks the
        item as booked and adds it to the cart.

        Args:
            cart_id (int): The ID of the cart to add to.
            product: The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        LOGGER.info("A cosumer wants to add product %s to the cart with id %s", product, cart_id)
        found_product = False

        # Block Logic: This entire search-and-book operation is a critical section
        # to prevent race conditions where two consumers might grab the same item.
        with self.lock_for_add_cart:
            for l_elem in range(len(self.products_queue)):
                for p_elem in range(len(self.products_queue[l_elem])):
                    # Pre-condition: Find a matching product that is not already booked.
                    if (self.products_queue[l_elem][p_elem] == product and
                            self.mark_booked_products[l_elem][p_elem] == 0):
                        
                        # Add to cart and mark as booked (1).
                        self.list_of_carts[cart_id].append(self.products_queue[l_elem][p_elem])
                        self.mark_booked_products[l_elem][p_elem] = 1
                        found_product = True
                        break
                if found_product:
                    break
        
        if found_product:
            LOGGER.info("The consumer added the product to the cart")
        else:
            LOGGER.info("The consumer did not add the product to the cart")
        return found_product


    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart, making it available again.

        Args:
            cart_id (int): The ID of the cart.
            product: The product to remove.
        """
        LOGGER.info("A cosumer wants to remove product %s from the cart %s", product, cart_id)
        with self.lock_for_add_cart:
            removed_product = False
            for l_elem in range(len(self.products_queue)):
                # This search logic assumes the product exists and is booked.
                # It finds the first instance and un-books it.
                for p_elem in range(len(self.products_queue[l_elem])):
                    if (self.products_queue[l_elem][p_elem] == product and
                            self.mark_booked_products[l_elem][p_elem] == 1):
                        
                        self.list_of_carts[cart_id].remove(product)
                        self.mark_booked_products[l_elem][p_elem] = 0 # Mark as available again.
                        removed_product = True
                        break
                if removed_product:
                    break
        LOGGER.info("The consumer removed the product from the cart")


    def place_order(self, cart_id):
        """
        Finalizes an order, permanently removing items from the marketplace inventory.

        Args:
            cart_id (int): The ID of the cart being ordered.

        Returns:
            list: A list of the products that were in the cart.
        """
        LOGGER.info("A consumer wants to place the order for the cart with id %s", cart_id)
        with self.lock_for_publish:
            elem_list = []
            # Invariant: Iterate through a copy of the cart items to avoid modification issues.
            for i in list(self.list_of_carts[cart_id]):
                print(threading.current_thread().name, "bought", i)
                elem_list.append(i)
                found = False
                # Block Logic: Find the corresponding booked item in the master product list and remove it.
                for l_elem in range(len(self.products_queue)):
                    for p_elem in range(len(self.products_queue[l_elem])):
                        if (self.products_queue[l_elem][p_elem] == i and
                                self.mark_booked_products[l_elem][p_elem] == 1):
                            self.products_queue[l_elem].pop(p_elem)
                            self.mark_booked_products[l_elem].pop(p_elem)
                            found = True
                            break
                    if found:
                        break
        
        LOGGER.info("The consumer placed the order")
        return elem_list


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    
    def setUp(self):
        """Set up a new marketplace and test data for each test."""
        self.marketplace = Marketplace(15)
        self.carts = [["add", "id2", 1], ["add", "id1", 3], ["remove", "id1", 1]]
        self.products = [["id2", 2, 0.18], ["id1", 1, 0.23]]
        self.consumer = Consumer(self.carts, self.marketplace, 0.31)
        self.producer = Producer(self.products, self.marketplace, 0.15)

    def test_register_producer(self):
        """Tests that producer registration returns sequential IDs starting from 0."""
        expected_register_id = 0
        self.assertEqual(expected_register_id, self.producer.marketplace.register_producer())

    def test_publish(self):
        """Tests that products can be published up to the queue limit."""
        expected_queue = [["id2", "id2", "id1", "id2", "id2", "id1", "id2", "id2",
                           "id1", "id2", "id2", "id1", "id2", "id2", "id1"]]
        self.marketplace.products_queue.append([])
        self.marketplace.mark_booked_products.append([])
        pos = 0
        while pos < self.marketplace.queue_size_per_producer:
            if pos % 3 == 2:
                self.producer.marketplace.publish(0, "id1")
            else:
                self.producer.marketplace.publish(0, "id2")
            pos += 1
        self.assertEqual(expected_queue, self.marketplace.products_queue)

    def test_new_cart(self):
        """Tests that new cart creation returns sequential IDs starting from 0."""
        expected_cart_id = 0
        self.assertEqual(expected_cart_id, self.consumer.marketplace.new_cart())

    def test_add_to_cart(self):
        """Tests that products are correctly added to a cart."""
        self.marketplace.products_queue.append([])
        self.marketplace.mark_booked_products.append([])
        pos = 0
        while pos < self.marketplace.queue_size_per_producer:
            if pos % 3 == 2:
                self.producer.marketplace.publish(0, "id1")
            else:
                self.producer.marketplace.publish(0, "id2")
            pos += 1
        self.consumer.marketplace.list_of_carts.append([])
        expected_cart = ["id2", "id1", "id1", "id1"]
        for i in self.carts:
            if i[0] == "add":
                for j in range(i[2]):
                    self.consumer.marketplace.add_to_cart(0, i[1])
        self.assertEqual(expected_cart, self.consumer.marketplace.list_of_carts[0])

    def test_remove_from_cart(self):
        """Tests that removing a product correctly updates the cart."""
        expected_cart = ["id2", "id1", "id1"]
        self.marketplace.products_queue.append([])
        self.marketplace.mark_booked_products.append([])
        pos = 0
        while pos < self.marketplace.queue_size_per_producer:
            if pos % 3 == 2:
                self.producer.marketplace.publish(0, "id1")
            else:
                self.producer.marketplace.publish(0, "id2")
            pos += 1
        self.consumer.marketplace.list_of_carts.append([])
        for i in self.carts:
            if i[0] == "add":
                for j in range(i[2]):
                    self.consumer.marketplace.add_to_cart(0, i[1])
        for i in self.carts:
            if i[0] == "remove":
                for j in range(i[2]):
                    self.consumer.marketplace.remove_from_cart(0, i[1])
        self.assertEqual(expected_cart, self.consumer.marketplace.list_of_carts[0])

    def test_place_order(self):
        """Tests that placing an order correctly finalizes the cart and returns the items."""
        expected_output = ["id2", "id1", "id1"]
        self.marketplace.products_queue.append([])
        self.marketplace.mark_booked_products.append([])
        pos = 0
        while pos < self.marketplace.queue_size_per_producer:
            if pos % 3 == 2:
                self.producer.marketplace.publish(0, "id1")
            else:
                self.producer.marketplace.publish(0, "id2")
            pos += 1
        self.consumer.marketplace.list_of_carts.append([])
        for i in self.carts:
            if i[0] == "add":
                for j in range(i[2]):
                    self.consumer.marketplace.add_to_cart(0, i[1])
        for i in self.carts:
            if i[0] == "remove":
                for j in range(i[2]):
                    self.consumer.marketplace.remove_from_cart(0, i[1])
        self.assertEqual(expected_output, self.consumer.marketplace.place_order(0))


class Producer(Thread):
    """
    Represents a producer thread that continuously publishes products.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes a Producer thread.

        Args:
            products (list): A list of products to publish, where each item contains
                the product, quantity, and sleep time.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish.
            **kwargs: Keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace


        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """The main execution loop for the producer thread.

        Registers with the marketplace and then enters an infinite loop,
        publishing its assigned products.
        """
        register_id = self.marketplace.register_producer()
        while 1:
            for i in self.products:
                for j in range(i[1]):
                    value = self.marketplace.publish(register_id, i[0])
                    # Pre-condition: If the marketplace queue is full, wait and retry.
                    while value is False:
                        time.sleep(self.republish_wait_time)
                        value = self.marketplace.publish(register_id, i[0])
                    time.sleep(i[2])
