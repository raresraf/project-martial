# -*- coding: utf-8 -*-
"""
This module simulates a multi-producer, multi-consumer marketplace.

It defines the core components of a concurrent marketplace simulation, including:
- Marketplace: The central, thread-safe hub for all transactions.
- Consumer: A thread that simulates a customer creating a cart and buying products.
- Producer: A thread that simulates a producer publishing products to the marketplace.
- Product, Tea, Coffee: Dataclasses representing items for sale.
- ProductDict: A thread-safe dictionary for managing product inventory.
- TestMarketplace, TestProductDict: Unit tests for the core components.

The simulation relies on Python's `threading` module for concurrency, using `Lock`
objects to protect shared data structures from race conditions.
"""
from threading import Thread
import time


class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace.

    Each consumer is initialized with a set of carts, where each cart contains
    a list of operations (add/remove products). The consumer processes these
    operations, places an order, and prints the products it "bought".
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of carts, where each cart is a list of
                          product operations (dictionaries).
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): The time in seconds to wait before
                                     retrying to add a product that is
                                     currently unavailable.
            **kwargs: Accepts a 'name' keyword argument for the thread name.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        """
        The main execution method for the consumer thread.

        Iterates through its assigned carts, executes the add/remove operations
        for each product, and finally places the order. If an attempt to add a
        product fails (because it's not in the marketplace), it will wait and
        retry.
        """
        market = self.marketplace
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for cart_ops in cart:
                # Pre-condition: Loop to perform the requested quantity of operations.
                for _ in range(0, cart_ops['quantity']):
                    if cart_ops['type'] == 'add':
                        # Attempt to add the product to the cart.
                        is_product_in_market = market.add_to_cart(cart_id, cart_ops['product'])
                        # Block Logic: If the product is not available, enter a retry loop.
                        # This simulates a consumer waiting for a product to be restocked.
                        # Invariant: The loop continues until `add_to_cart` succeeds.
                        while not is_product_in_market:
                            time.sleep(self.retry_wait_time)
                            is_product_in_market = market.add_to_cart(cart_id, cart_ops['product'])
                    else:
                        # Remove the product from the cart.
                        self.marketplace.remove_from_cart(cart_id, cart_ops['product'])

            # After all operations, place the order to finalize the purchase.
            product_list = self.marketplace.place_order(cart_id)
            for product in product_list:
                print(self.name, "bought", product)


import logging
import time
import unittest


from threading import Lock
from logging.handlers import RotatingFileHandler
from tema.product_dict import ProductDict
from tema.product import Tea
from tema.product import Coffee


class Marketplace:
    """
    A thread-safe marketplace that manages producers, products, and carts.

    This class acts as the central shared resource in the simulation. It uses
    locks to protect access to its internal data structures (product inventory,
    producer queues, consumer carts), ensuring safe concurrent access from
    multiple producer and consumer threads.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in the marketplace at
                                           any given time.
        """
        self.queue_size_per_producer = queue_size_per_producer

        # Lock-protected counter for assigning unique producer IDs.
        self.next_producer_id = 1
        self.next_producer_id_lock = Lock()

        # Lock-protected counter for assigning unique cart IDs.
        self.next_cart_id = 1
        self.next_cart_id_lock = Lock()

        # The main inventory of products available in the marketplace.
        self.market_products = ProductDict()

        # Tracks the number of products each producer has published.
        self.producer_queue_sizes = {}
        self.producer_queue_sizes_lock = Lock()

        # Stores the contents of each active consumer cart.
        self.consumer_carts = {}
        self.consumer_carts_lock = Lock()
        handler = RotatingFileHandler(
            'marketplace.log',
            mode='w',
            maxBytes=1000000,
            backupCount=1000,
            delay=True
        )

        logging.basicConfig(
            handlers=[handler],
            level=logging.INFO,
            format='%(asctime)s %(levelname)s : %(message)s'
        )

        logging.Formatter.converter = time.gmtime

    def register_producer(self):
        """
        Registers a new producer, assigning it a unique ID.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        logging.info('Entering register_producer')
        with self.next_producer_id_lock:
            curr_producer_id = self.next_producer_id
            self.next_producer_id += 1

        with self.producer_queue_sizes_lock:
            self.producer_queue_sizes[curr_producer_id] = 0

        logging.info('Leaving register_producer')
        return curr_producer_id

    def publish(self, producer_id, product):
        """
        Adds a product from a producer to the marketplace inventory.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Product): The product to be published.

        Returns:
            bool: True if the product was successfully published, False if the
                  producer's queue is full.
        """
        logging.info('Entering publish with producer_id=%d product=%s', producer_id, repr(product))

        # Pre-condition: Check if the producer has reached its publishing limit.
        with self.producer_queue_sizes_lock:
            if self.producer_queue_sizes[producer_id] >= self.queue_size_per_producer:
                logging.info('Leaving publish')
                return False

        # Add the product to the central inventory.
        self.market_products.put(product, producer_id)

        # Increment the producer's published count.
        with self.producer_queue_sizes_lock:
            self.producer_queue_sizes[producer_id] += 1

        logging.info('Leaving publish')
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.

        Returns:
            int: The unique ID assigned to the new cart.
        """
        logging.info('Entering new_cart')
        with self.next_cart_id_lock:
            curr_cart_id = self.next_cart_id
            self.next_cart_id += 1

        # Associate the new cart ID with an empty ProductDict.
        with self.consumer_carts_lock:
            self.consumer_carts[curr_cart_id] = ProductDict()

        logging.info('Leaving new_cart')
        return curr_cart_id

    def get_cart(self, cart_id) -> ProductDict:
        """Retrieves the contents of a specific cart."""
        logging.info('Entering get_cart with cart_id=%d', cart_id)
        with self.consumer_carts_lock:
            logging.info('Leaving get_cart')
            return self.consumer_carts[cart_id]

    def add_to_cart(self, cart_id, product):
        """
        Moves a product from the marketplace inventory to a consumer's cart.

        Args:
            cart_id (int): The ID of the target cart.
            product (Product): The product to add.

        Returns:
            bool: True if the product was successfully added, False if the
                  product was not available in the marketplace.
        """
        logging.info('Entering add_to_cart with cart_id=%d product=%s', cart_id, repr(product))
        # Atomically removes the product from the main inventory.
        producer_id = self.market_products.remove(product)

        # Pre-condition: If remove() returns None, the product was not available.
        if not producer_id:
            logging.info('Leaving add_to_cart')
            return False

        # Add the retrieved product to the consumer's personal cart.
        consumer_cart = self.get_cart(cart_id)
        consumer_cart.put(product, producer_id)

        logging.info('Leaving add_to_cart')
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Moves a product from a consumer's cart back to the marketplace inventory.
        """
        log_message = 'Entering remove_from_cart with cart_id=%d product=%s'
        logging.info(log_message, cart_id, repr(product))
        consumer_cart = self.get_cart(cart_id)
        # Remove the product from the consumer's cart to get its original producer ID.
        producer_id = consumer_cart.remove(product)

        # Return the product to the main marketplace inventory.
        self.market_products.put(product, producer_id)
        logging.info('Leaving remove_from_cart')

    def place_order(self, cart_id):
        """
        Finalizes a purchase, consuming the products in a cart.

        This action decrements the queue size for the original producers of the
        purchased items.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: A list of the products that were in the cart.
        """
        logging.info('Entering place_order with cart_id=%d', cart_id)
        consumer_cart = self.get_cart(cart_id)
        product_list = []
        for product in consumer_cart.dict:
            quantity_dict = consumer_cart.dict[product]

            # Block Logic: Iterate through all items in the cart to finalize them.
            for producer_id in quantity_dict:
                quantity = quantity_dict[producer_id]
                for _ in range(0, quantity):
                    product_list.append(product)

                # Decrement the producer's active product count, as the items
                # have now been sold.
                with self.producer_queue_sizes_lock:
                    self.producer_queue_sizes[producer_id] -= quantity

        logging.info('Leaving place_order')
        return product_list


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self) -> None:
        """Sets up a new Marketplace and products for each test."""
        self.marketplace = Marketplace(3)
        self.product1 = Tea("Linden", 9, "Linden")
        self.product2 = Coffee("Indonezia", 1, 5.05, 'MEDIUM')

    def test_register_producer(self):
        """Tests that producer registration assigns sequential, unique IDs."""
        for i in range(1, 100):
            self.assertEqual(self.marketplace.register_producer(), i)

    def test_publish(self):
        """Tests the logic for publishing products, including queue limits."""
        self.marketplace.register_producer()


        self.assertEqual(self.marketplace.publish(1, self.product1), True)
        self.assertEqual(self.marketplace.publish(1, self.product1), True)
        self.assertEqual(self.marketplace.publish(1, self.product2), True)
        self.assertEqual(self.marketplace.producer_queue_sizes[1], 3)

        # Test that publishing fails when the producer's queue is full.
        self.assertEqual(self.marketplace.publish(1, self.product2), False)
        self.assertEqual(self.marketplace.producer_queue_sizes[1], 3)
        market_products = {self.product1: {1: 2}, self.product2: {1: 1}}
        self.assertEqual(self.marketplace.market_products.dict, market_products)

        self.marketplace.register_producer()
        for _ in range(0, 10):
            self.marketplace.publish(2, self.product2)

        market_products[self.product2][2] = 3
        self.assertEqual(self.marketplace.market_products.dict, market_products)

    def test_new_cart(self):
        """Tests that new carts are created with sequential, unique IDs."""
        for i in range(1, 100):
            self.assertEqual(self.marketplace.new_cart(), i)

    def test_get_cart(self):
        """Tests retrieving a cart and verifying its contents."""
        self.marketplace.new_cart()
        self.assertEqual(self.marketplace.get_cart(1).dict, {})
        self.marketplace.register_producer()
        for i in range(1, 4):
            self.marketplace.publish(1, self.product1)
            self.marketplace.add_to_cart(1, self.product1)
            cart = {self.product1: {1: i}}
            self.assertEqual(self.marketplace.get_cart(1).dict, cart)

    def test_add_to_cart(self):
        """Tests moving products from the marketplace to a consumer's cart."""
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.product1)
        self.marketplace.publish(2, self.product1)
        self.marketplace.publish(2, self.product2)

        self.marketplace.new_cart()
        self.marketplace.add_to_cart(1, self.product1)
        self.marketplace.add_to_cart(1, self.product1)
        cart = {self.product1: {1: 1, 2: 1}}
        self.assertEqual(self.marketplace.get_cart(1).dict, cart)

        self.marketplace.add_to_cart(1, self.product2)
        self.assertEqual(self.marketplace.market_products.dict, {})
        cart = {self.product1: {1: 1, 2: 1}, self.product2: {2: 1}}
        self.assertEqual(self.marketplace.get_cart(1).dict, cart)

    def fill_cart(self):
        """Helper method to populate a cart with products for testing."""
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        for _ in range(1, 4):
            self.marketplace.publish(1, self.product1)
            self.marketplace.publish(2, self.product2)
            self.marketplace.add_to_cart(1, self.product2)
            self.marketplace.add_to_cart(1, self.product1)

    def test_remove_from_cart(self):
        """Tests moving products from a cart back to the marketplace."""
        self.fill_cart()
        for i in range(0, 3):
            cart = {self.product1: {1: 3 - i}, self.product2: {2: 3}}
            self.assertEqual(self.marketplace.get_cart(1).dict, cart)
            self.marketplace.remove_from_cart(1, self.product1)
            market_products = {self.product1: {1: i + 1}}
            self.assertEqual(self.marketplace.market_products.dict, market_products)

        for i in range(0, 3):
            cart = {self.product2: {2: 3 - i}}
            self.assertEqual(self.marketplace.get_cart(1).dict, cart)
            self.marketplace.remove_from_cart(1, self.product2)
            market_products = {self.product1: {1: 3}, self.product2: {2: i + 1}}
            self.assertEqual(self.marketplace.market_products.dict, market_products)

        self.assertEqual(self.marketplace.get_cart(1).dict, {})

    def test_place_order(self):
        """Tests the finalization of an order and producer queue updates."""
        self.fill_cart()
        self.marketplace.remove_from_cart(1, self.product1)
        self.marketplace.remove_from_cart(1, self.product2)
        products = self.marketplace.place_order(1)
        product1_count = 0
        product2_count = 0
        for product in products:
            if product == self.product1:
                product1_count += 1

            if product == self.product2:
                product2_count += 1

        self.assertEqual(product1_count, 2)
        self.assertEqual(product2_count, 2)
        market_products = {self.product1: {1: 1}, self.product2: {2: 1}}
        self.assertEqual(self.marketplace.market_products.dict, market_products)
        self.assertEqual(self.marketplace.producer_queue_sizes[1], 1)
        self.assertEqual(self.marketplace.producer_queue_sizes[2], 1)


from threading import Thread
import time


class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.

    The producer runs in an infinite loop, continuously attempting to publish
    a predefined list of products.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of tuples, where each tuple contains
                             (product, quantity, processing_time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): The time in seconds to wait before
                                         retrying to publish products.
            **kwargs: Accepts 'daemon' and 'name' keyword arguments.
        """
        Thread.__init__(self)
        self.setDaemon(kwargs['daemon'])
        self.product_infos = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()
        self.name = kwargs['name']

    def run(self):
        """
        The main execution method for the producer thread.

        Continuously loops through its product list, publishing each one.
        A processing time is simulated after each publication. If publishing
        fails (due to a full queue), it waits before retrying.
        """
        while True:
            for product_info in self.product_infos:
                (product, quantity, processing_time) = product_info
                for _ in range(0, quantity):
                    can_i_republish = self.marketplace.publish(self.producer_id, product)
                    time.sleep(processing_time)
                    if not can_i_republish:
                        time.sleep(self.republish_wait_time)

            time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A simple, immutable data class representing a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A data class representing a type of Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A data class representing a type of Coffee, inheriting from Product."""
    acidity: float
    roast_level: str


from threading import Lock
from threading import Thread
import unittest
from tema.product import Tea
from tema.product import Coffee


class ProductDict:
    """
    A thread-safe dictionary for managing product inventory.

    It stores products and tracks their quantities on a per-producer basis.
    The structure is a nested dictionary: {product: {producer_id: quantity}}.
    All access is protected by a lock.
    """
    def __init__(self):
        """Initializes the dictionary and its lock."""
        self.dict = {}
        self.dict_lock = Lock()

    def put(self, product, producer_id):
        """
        Adds one unit of a product from a specific producer to the inventory.
        """
        with self.dict_lock:
            if product in self.dict:
                quantity_dict = self.dict[product]
                # Increment the quantity for an existing producer.
                if producer_id in quantity_dict:
                    quantity_dict[producer_id] += 1
                # Add a new producer for an existing product.
                else:
                    quantity_dict[producer_id] = 1
            else:
                # Add a new product to the inventory.
                self.dict[product] = {producer_id: 1}

    def remove(self, product):
        """
        Removes one unit of a product from the inventory.

        It removes a unit from the first available producer for that product.

        Args:
            product (Product): The product to remove.

        Returns:
            int or None: The ID of the producer from whom the product was
                         taken, or None if the product is not available.
        """
        with self.dict_lock:
            if product not in self.dict:
                return None

            # Find the first producer for this product and decrement their count.
            quantity_dict = self.dict[product]
            for producer_id in quantity_dict:
                quantity_dict[producer_id] -= 1
                producer_id_return = producer_id
                break

            # If the producer's count for this product drops to zero, remove them.
            if quantity_dict[producer_id_return] == 0:
                quantity_dict.pop(producer_id_return)

            # If no producers are left for this product, remove the product entry.
            if not quantity_dict:
                self.dict.pop(product)

            return producer_id_return


class TestProductDict(unittest.TestCase):
    """
    Unit tests for the ProductDict class, focusing on its concurrency-handling.
    """
    def setUp(self) -> None:
        """
        Sets up a multi-threaded test environment to check for race conditions.
        """
        self.product_dict = ProductDict()
        self.product1 = Tea("Linden", 9, "Herbal")
        self.product2 = Coffee("Indonezia", 1, 5.05, 'MEDIUM')

        def thread_run():
            for _ in range(0, 5):
                for j in range(1, 6):
                    self.product_dict.put(self.product1, j)
                    self.product_dict.put(self.product2, j + 1)

        threads = []
        for _ in range(0, 10):
            thread = Thread(target=thread_run)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def test_put(self):
        """
        Tests that `put` operations from multiple threads are correctly aggregated.
        """
        quantity_dict1 = {1: 50, 2: 50, 3: 50, 4: 50, 5: 50}
        quantity_dict2 = {2: 50, 3: 50, 4: 50, 5: 50, 6: 50}
        product_dict = {self.product1: quantity_dict1, self.product2: quantity_dict2}
        self.assertEqual(self.product_dict.dict, product_dict)

    def test_remove(self):
        """
        Tests that `remove` operations from multiple threads correctly
        decrement the inventory without race conditions.
        """
        product_ids1 = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        product_ids2 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

        def thread_run():
            for _ in range(0, 5):
                for _ in range(0, 5):
                    product_id1 = self.product_dict.remove(self.product1)
                    product_ids1[product_id1] += 1
                    product_id2 = self.product_dict.remove(self.product2)
                    product_ids2[product_id2] += 1

        threads = []
        for _ in range(0, 10):
            thread = Thread(target=thread_run)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        self.assertEqual(self.product_dict.dict, {})
        product_ids1_correct = {1: 50, 2: 50, 3: 50, 4: 50, 5: 50}
        product_ids2_correct = {2: 50, 3: 50, 4: 50, 5: 50, 6: 50}
        self.assertEqual(product_ids1, product_ids1_correct)
        self.assertEqual(product_ids2, product_ids2_correct)
        self.assertEqual(self.product_dict.remove(self.product1), None)
