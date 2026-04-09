"""
This module simulates a multi-threaded producer-consumer marketplace.

It contains the core components for the simulation:
- Marketplace: A thread-safe central hub with integrated logging where products
  are published and purchased.
- Producer: A thread that publishes products to the marketplace.
- Consumer: A thread that adds products to a cart and places orders.
- TestMarketplace: A suite of unit tests to validate the marketplace functionality.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer thread that simulates purchasing products.

    Each consumer processes a list of predefined shopping actions, interacting
    with the marketplace to fill a cart and place an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping carts, where each cart is a list
                          of dictionaries specifying add/remove actions.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait in seconds before retrying
                                     to add an unavailable product.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution logic for the consumer thread.

        For each assigned cart, it creates a new cart in the marketplace,
        executes all add/remove actions, and finally places the order.
        """
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()

            
            # Invariant: Process all actions defined for the current cart.
            for action in cart:
                curr_quantity = 0

                
                # Pre-condition: Ensure the action is performed for the specified quantity.
                while curr_quantity < action["quantity"]:
                    if action["type"] == "add":
                        
                        # Attempt to add product; if it fails, wait and retry.
                        if self.marketplace.add_to_cart(cart_id, action["product"]):
                            curr_quantity += 1
                        else:
                            
                            sleep(self.retry_wait_time)
                    elif action["type"] == "remove":
                        
                        self.marketplace.remove_from_cart(cart_id, action["product"])
                        curr_quantity += 1

            
            self.marketplace.place_order(cart_id)

import time
import unittest
import logging
from logging.handlers import RotatingFileHandler
from threading import Lock, currentThread


from tema.consumer import Consumer
from tema.producer import Producer
from tema.product import Coffee, Tea


class Marketplace:
    """
    A thread-safe marketplace for producers and consumers.

    This class acts as the central coordinator, managing product inventory,
    producer queues, and customer carts. It uses locks to ensure that all
    operations are atomic and can be safely called from multiple threads.
    It also features integrated logging for all significant events.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products any
                                           single producer can have listed at
                                           one time.
        """
        self.queue_size_per_producer = queue_size_per_producer

        


        self.nr_producers = -1

        
        self.nr_carts = -1

        
        self.queues = []

        
        self.products = []

        
        self.carts = []

        self.producer_lock = Lock()
        self.cart_lock = Lock()
        self.cart_add_lock = Lock()
        self.place_order_lock = Lock()

        
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
        
        logging.Formatter.converter = time.gmtime

        self.logger = logging.getLogger("marketplace_logger")

        
        handler = RotatingFileHandler("file.log", maxBytes=5000, backupCount=15)
        self.logger.addHandler(handler)

        
        self.logger.propagate = False

    def register_producer(self):
        """
        Registers a new producer with the marketplace.

        This method is thread-safe. It assigns a unique ID to the producer
        and initializes an empty queue for them.

        Returns:
            int: The unique ID for the registered producer.
        """
        
        
        with self.producer_lock:
            self.nr_producers += 1

            
            self.queues.append([])

            self.logger.info(f'register_producer output: producer_id={self.nr_producers}')

        
        return self.nr_producers

    def publish(self, producer_id, product):
        """
        Publishes a product from a specific producer to the marketplace.

        This method is thread-safe.

        Args:
            producer_id (int): The ID of the producer publishing the item.
            product: The product to be published.

        Returns:
            bool: True if publishing was successful, False if the producer's
                  queue is full.
        """
        self.logger.info(f'publish input: producer_id={producer_id}, product={product}')

        p_id = int(producer_id)

        
        # Pre-condition: Check if the producer's queue has space.
        if len(self.queues[p_id]) == self.queue_size_per_producer:
            self.logger.info("publish output: FALSE")
            return False

        
        self.products.append(product)
        self.queues[p_id].append(product)

        self.logger.info("publish output: TRUE")
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart.

        This method is thread-safe.

        Returns:
            int: The unique ID for the new cart.
        """
        
        
        with self.cart_lock:
            self.nr_carts += 1

            
            self.carts.append([])

            self.logger.info(f'new_cart output: cart_id={self.nr_carts}')

        
        return self.nr_carts

    def add_to_cart(self, cart_id, product):
        """
        Adds an available product from the marketplace into a shopping cart.

        This operation is thread-safe. It atomically moves a product from the
        general availability pool into a specific user's cart.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product: The product to add.

        Returns:
            bool: True if the product was available and added, False otherwise.
        """
        self.logger.info(f'add_to_cart input: cart_id={cart_id}, product={product}')

        
        
        with self.cart_add_lock:
            
            # Pre-condition: Check if the product is in the list of available products.
            if product not in self.products:
                self.logger.info("add_to_cart output: FALSE")
                
                return False

            
            self.products.remove(product)

            
            self.carts[cart_id].append(product)

            self.logger.info("add_to_cart output: TRUE")

        
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart and returns it to the marketplace.

        Args:
            cart_id (int): The ID of the cart.
            product: The product to remove from the cart.
        """
        self.logger.info(f'remove_from_cart input: cart_id={cart_id}, product={product}')

        
        self.products.append(product)

        
        self.carts[cart_id].remove(product)

    def remove_from_queue(self, product):
        """
        Removes a purchased product from its original producer's queue.

        This completes the consumption cycle, ensuring a producer's queue slot
        is freed up after their product is successfully sold.

        Args:
            product: The product that has been sold.
        """
        self.logger.info(f'remove_from_queue input: product={product}')

        
        # Invariant: Search all producer queues to find and remove the product.
        for producer_queue in self.queues:
            if product in producer_queue:
                producer_queue.remove(product)
                break

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.

        This thread-safe method iterates through the items in the cart,
        removes them from their respective producer queues to signal consumption,
        and prints the purchase information.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: The final list of products that were in the cart.
        """
        
        self.logger.info(f'place_order input: cart_id={cart_id}')

        with self.place_order_lock:
            
            
            for product in self.carts[cart_id]:
                self.remove_from_queue(product)
                print(currentThread().name, "bought", product)

        self.logger.info(f'place_order output: cart_list={self.carts[cart_id]}')

        
        return self.carts[cart_id]


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self):
        """Sets up the test fixture before each test method."""
        self.marketplace = Marketplace(15)
        self.product1 = Coffee("Indonezia", 1, "5.05", "MEDIUM")
        self.product2 = Tea("Linden", 9, "Herbal")


        self.producer = Producer([[self.product1, 2, 0.18],
                                  [self.product2, 1, 0.23]],
                                 self.marketplace,
                                 0.15)
        self.consumer = Consumer([[{"type": "add", "product": self.product2, "quantity": 2},
                                   {"type": "add", "product": self.product1, "quantity": 2},
                                   {"type": "remove", "product": self.product1, "quantity": 1}
                                   ]],
                                 self.marketplace,
                                 0.31)

        self.cart_id = self.marketplace.new_cart()

    def test_register_producer(self):
        """Tests that producer registration returns the correct initial ID."""
        self.assertEqual(self.producer.producer_id, "0")

    def test_publish(self):
        """Tests that products are correctly published and stored."""
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product2)

        self.assertEqual(self.marketplace.products,
                         self.marketplace.queues[int(self.producer.producer_id)])

    def test_new_cart(self):
        """Tests that new cart creation returns the correct initial ID."""
        self.assertEqual(self.cart_id, 0)

    def test_add_to_cart(self):
        """Tests adding products to a cart."""
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product2)

        self.marketplace.add_to_cart(self.cart_id, self.product1)
        self.marketplace.add_to_cart(self.cart_id, self.product2)

        self.assertEqual(self.marketplace.carts[self.cart_id], [self.product1, self.product2])

    def test_remove_from_cart(self):
        """Tests removing a product from a cart."""
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product2)

        self.marketplace.add_to_cart(self.cart_id, self.product1)
        self.marketplace.add_to_cart(self.cart_id, self.product1)
        self.marketplace.add_to_cart(self.cart_id, self.product2)

        self.marketplace.remove_from_cart(self.cart_id, self.product1)
        self.assertEqual(self.marketplace.carts[self.cart_id], [self.product1, self.product2])

    def test_place_order(self):
        """Tests the entire flow from publishing to placing an order."""
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product1)
        self.marketplace.publish(self.producer.producer_id, self.product2)
        self.marketplace.publish(self.producer.producer_id, self.product2)

        self.consumer.run()

        self.assertEqual(self.marketplace.queues[int(self.producer.producer_id)],
                         [self.product1])


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.

    The producer runs in an infinite loop, continuously attempting to publish
    a predefined list of products according to specified quantities and timings.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products to publish. Each element is a
                             list of [product, quantity, interval].
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying a failed
                                         publish attempt.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products


        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = str(self.marketplace.register_producer())

    def run(self):
        """
        The main execution logic for the producer thread.

        Enters an infinite loop, iterating through its product list and
        publishing them to the marketplace.
        """
        
        while True:
            
            for (product, max_products, success_wait_time) in self.products:
                curr_products = 0

                
                # Invariant: Publish the product `max_products` times.
                while curr_products < max_products:
                    
                    # If publishing is successful, wait and continue.
                    if self.marketplace.publish(self.producer_id, product):
                        
                        curr_products += 1
                        sleep(success_wait_time)
                    else:
                        # If publishing fails (e.g., queue is full), wait and retry.
                        sleep(self.republish_wait_time)
