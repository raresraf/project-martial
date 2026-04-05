

"""
@5071e982-e301-4cca-ac79-1b061d216f34/consumer.py
@brief Simulates a marketplace system with producers and consumers, managing product availability, cart operations, and orders.
Functional Utility: This module provides the core components for a multi-threaded simulation of an e-commerce marketplace, demonstrating concurrent access and synchronization for managing product inventory and consumer purchases.
Domain: Concurrency, Producer-Consumer Problem, E-commerce Simulation.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    @brief Represents a consumer that interacts with the marketplace to add and remove products from a cart and place orders.
    Functional Utility: Manages the lifecycle of a consumer's shopping activities, from creating a cart to placing an order, including handling product availability.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.
        @param carts: A list of shopping carts, where each cart is a list of operations (add/remove product).
        @param marketplace: Reference to the shared `Marketplace` instance.
        @param retry_wait_time: Time to wait before retrying an `add_to_cart` operation.
        @param **kwargs: Additional keyword arguments, including the thread name.
        Functional Utility: Sets up the consumer with its assigned shopping intentions, access to the marketplace, and retry logic.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        @brief Executes the consumer's shopping logic.
        Functional Utility: Orchestrates the consumer's interaction with the marketplace, processing each cart's operations and finalizing orders.
        Block Logic: Iterates through each `cart` assigned to the consumer, performing add/remove operations and finally placing the order.
        """
        for cart in self.carts:
            # Functional Utility: Requests a new, unique shopping cart from the marketplace.
            cart_id = self.marketplace.new_cart()
            # Block Logic: Processes each operation (add or remove) within the current cart.
            for ops in cart:
                # Block Logic: Handles adding a product to the cart.
                if ops['type'] == "add":
                    for _ in range(0, ops['quantity']):
                        # Invariant: Product not successfully added to cart.
                        # Functional Utility: Continuously attempts to add the product to the cart until successful.
                        while self.marketplace.add_to_cart(cart_id, ops['product']) is not True:
                            # Functional Utility: Pauses execution before retrying to prevent busy-waiting.
                            sleep(self.retry_wait_time)
                else:
                    for _ in range(0, ops['quantity']):
                        # Functional Utility: Handles removing a product from the cart.
                        self.marketplace.remove_from_cart(cart_id, ops['product'])
            # Functional Utility: Finalizes the order for the current cart.
            products = self.marketplace.place_order(cart_id)

            # Functional Utility: Acquires the consumer-specific lock to ensure exclusive access to shared console output.
            lock = self.marketplace.get_consumer_lock()

            # Pre-condition: Ensures only one consumer can print at a time.
            lock.acquire()
            # Functional Utility: Prints the list of products successfully purchased by the consumer.
            for product in products:
                print(self.kwargs['name'] + " bought " + str(product))
            # Functional Utility: Releases the consumer lock.
            lock.release()


import logging
from logging.handlers import RotatingFileHandler
import time
from threading import Lock
import unittest

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO,
                    handlers=[RotatingFileHandler('marketplace.log',
                                                  maxBytes=20000, backupCount=10)])
logging.Formatter.converter = time.gmtime

class Marketplace:
    """
    @brief Manages products, producers, and consumer carts in a thread-safe manner.
    Functional Utility: Provides the central logic for product flow within the e-commerce simulation, ensuring data consistency and handling interactions between producers and consumers.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the marketplace.
        @param queue_size_per_producer: The maximum number of products a producer can have in the marketplace at any given time.
        Functional Utility: Sets up internal data structures for managing products, carts, producer and consumer IDs, and associated locks for thread safety.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_lock = Lock()
        self.consumer_lock = Lock()
        self.producer_id = -1
        self.cart_id = -1
        self.size_per_producer = {}
        self.carts = {}
        self.products_dict = {}

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace, assigning it a unique ID.
        Functional Utility: Atomically increments the producer ID and initializes the product count for the new producer.
        Synchronization: Uses `self.producer_lock` to protect shared state during registration.
        @return: The newly registered producer's ID.
        """
        self.producer_lock.acquire()
        logging.info("New producer entered register_producer method")
        self.producer_id += 1
        self.size_per_producer[self.producer_id] = 0
        self.producer_lock.release()
        logging.info("New producer registered with id %d", self.producer_id)
        return self.producer_id

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to publish a product to the marketplace.
        @param producer_id: The ID of the producer publishing the product.
        @param product: The product to be published.
        Functional Utility: Adds the product to the marketplace, associating it with the producer, if the producer's queue size limit is not exceeded.
        Synchronization: Uses `self.producer_lock` to ensure thread-safe updates to product and producer inventories.
        Pre-condition: `self.size_per_producer[producer_id] < self.queue_size_per_producer`.
        @return: `True` if the product was successfully published, `False` otherwise.
        """
        logging.info("Producer with id %d entered publish method", producer_id)

        self.producer_lock.acquire()
        # Block Logic: Checks if the producer has reached its maximum allowed products in the marketplace.
        if self.size_per_producer[producer_id] == self.queue_size_per_producer:
            logging.info(f"Producer with id {producer_id} failed to publish product {product}")
            self.producer_lock.release()
            return False

        # Block Logic: Adds the product to the marketplace's inventory.
        if product not in self.products_dict:
            self.products_dict[product] = [producer_id]
        else:
            self.products_dict[product].append(producer_id)

        self.size_per_producer[producer_id] += 1
        logging.info(f"Producer with id {producer_id} published product {product}")
        self.producer_lock.release()
        return True

    def new_cart(self):
        """
        @brief Creates a new shopping cart for a consumer, assigning it a unique ID.
        Functional Utility: Atomically increments the cart ID and initializes an empty cart for the consumer.
        Synchronization: Uses `self.consumer_lock` to protect shared state during cart creation.
        @return: The ID of the newly created cart.
        """
        self.consumer_lock.acquire()
        logging.info("Consumer entered new_cart method")
        self.cart_id += 1
        self.carts[self.cart_id] = {}
        logging.info("Consumer registered new cart with id %d", self.cart_id)
        self.consumer_lock.release()
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Attempts to add a product to a consumer's cart.
        @param cart_id: The ID of the cart to which the product should be added.
        @param product: The product to add.
        Functional Utility: Transfers a product from the available products in the marketplace to the specified cart, ensuring atomicity.
        Synchronization: Uses `self.consumer_lock` to ensure thread-safe access to product inventory and carts.
        Pre-condition: The `product` must be available in the marketplace (`self.products_dict`).
        @return: `True` if the product was successfully added, `False` otherwise.
        """
        self.consumer_lock.acquire()
        logging.info("Consumer with card id %d entered add_to_cart method", cart_id)
        # Block Logic: Checks if the product is available in the marketplace.
        if product in self.products_dict:
            # Functional Utility: Retrieves a producer ID associated with the product and moves the product to the cart.
            producer_id = self.products_dict[product].pop(0)
            if product in self.carts[cart_id]:
                self.carts[cart_id][product].append(producer_id)
            else:
                self.carts[cart_id][product] = [producer_id]
            
            # Functional Utility: Removes the product from the marketplace's main product list if no more units are available.
            if len(self.products_dict[product]) == 0:
                del self.products_dict[product]

            logging.info(f"Consumer with card id {cart_id} added product {product} to cart")
            self.consumer_lock.release()
            return True
        logging.info(f"Consumer with card id {cart_id} failed to add product {product} to cart")
        self.consumer_lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a consumer's cart and returns it to the marketplace.
        @param cart_id: The ID of the cart from which to remove the product.
        @param product: The product to remove.
        Functional Utility: Moves a product back from a consumer's cart to the available products in the marketplace, correctly updating quantities.
        Synchronization: Uses `self.consumer_lock` for thread-safe access to carts and product inventory.
        """
        self.consumer_lock.acquire()
        logging.info("Consumer with card id %d entered remove_from_cart method", cart_id)

        # Functional Utility: Removes a specific instance of the product from the cart.
        given_id = self.carts[cart_id][product].pop(0)
        # Block Logic: If no more units of the product remain in the cart, removes the product entry from the cart.
        if len(self.carts[cart_id][product]) == 0:
            del self.carts[cart_id][product]

        # Block Logic: Returns the product to the general marketplace inventory.
        if product not in self.products_dict:
            self.products_dict[product] = [given_id]
        else:
            self.products_dict[product].append(given_id)
        logging.info(f"Consumer with card id {cart_id} removed product {product} from cart")
        self.consumer_lock.release()

    def place_order(self, cart_id):
        """
        @brief Finalizes a consumer's order for a given cart.
        @param cart_id: The ID of the cart for which to place the order.
        Functional Utility: Processes the items in the cart, decrements the producer's product count for each item, and returns the list of purchased products.
        Synchronization: Uses `self.consumer_lock` for thread-safe access to cart contents and producer inventory.
        @return: A list of products successfully ordered.
        """
        self.consumer_lock.acquire()
        logging.info("Consumer with card id %d entered place_order method", cart_id)
        
        # Functional Utility: Iterates through the products in the cart to finalize the purchase and update producer counts.
        products = []
        for product in self.carts[cart_id]:
            for given_id in self.carts[cart_id][product]:
                self.size_per_producer[given_id] -= 1
                products.append(product)
        logging.info("Consumer with card id %d placed order", cart_id)
        self.consumer_lock.release()
        return products

    def get_consumer_lock(self):
        """
        @brief Provides access to the consumer lock.
        Functional Utility: Allows consumers to synchronize access to shared resources, such as console output.
        @return: The `consumer_lock` object.
        """
        logging.info("A consumer entered get_consumer_lock method")
        return self.consumer_lock


class TestMarketplace(unittest.TestCase):
    """
    @brief Provides unit tests for the `Marketplace` class to ensure its functionality and thread safety.
    Functional Utility: Verifies the correct behavior of the Marketplace's core methods under various scenarios.
    """
    def setUp(self):
        """
        @brief Sets up a new `Marketplace` instance before each test.
        Functional Utility: Ensures a clean and consistent state for each test case by initializing a fresh marketplace.
        """
        self.marketplace = Marketplace(5)

    def test_register_producer(self):
        """
        @brief Tests the `register_producer` method.
        Functional Utility: Verifies that a producer can be successfully registered and assigned a unique ID.
        """
        self.assertEqual(self.marketplace.register_producer(), 0)

    def test_true_publish(self):
        """
        @brief Tests successful product publishing.
        Functional Utility: Confirms that a producer can publish a product and it is correctly added to the marketplace.
        """
        producer_id = self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(producer_id, "id1"))
        self.assertEqual(len(self.marketplace.products_dict), 1)
        self.assertEqual(len(self.marketplace.products_dict["id1"]), 1)

    def test_false_publish(self):
        """
        @brief Tests product publishing when the producer's queue is full.
        Functional Utility: Verifies that the marketplace correctly prevents a producer from publishing beyond its allowed capacity.
        """
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")
        self.marketplace.publish(producer_id, "id2")
        self.marketplace.publish(producer_id, "id2")
        self.marketplace.publish(producer_id, "id1")
        self.marketplace.publish(producer_id, "id2")
        self.assertFalse(self.marketplace.publish(producer_id, "id1"))

    def test_new_cart(self):
        """
        @brief Tests the `new_cart` method.
        Functional Utility: Verifies that a new shopping cart can be successfully created and assigned a unique ID.
        """
        self.assertEqual(self.marketplace.new_cart(), 0)

    def test_true_add_to_cart(self):
        """
        @brief Tests successful product addition to a cart.
        Functional Utility: Confirms that a product can be added to a cart and is correctly removed from the marketplace's inventory.
        """
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")
        cart_id = self.marketplace.new_cart()

        self.assertTrue(self.marketplace.add_to_cart(cart_id, "id1"))
        self.assertEqual(len(self.marketplace.products_dict), 0)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 1)
        self.assertEqual(len(self.marketplace.carts[cart_id]["id1"]), 1)

        self.marketplace.publish(producer_id, "id1")
        self.assertTrue(self.marketplace.add_to_cart(cart_id, "id1"))
        self.assertEqual(len(self.marketplace.carts[cart_id]["id1"]), 2)

    def test_false_add_to_cart(self):
        """
        @brief Tests product addition to a cart when the product is unavailable.
        Functional Utility: Verifies that attempting to add an unavailable product to a cart correctly returns False.
        """
        cart_id = self.marketplace.new_cart()
        self.assertFalse(self.marketplace.add_to_cart(cart_id, "id1"))

    def test_remove_from_cart(self):
        """
        @brief Tests removing a product from a cart and its return to the marketplace.
        Functional Utility: Ensures that products can be correctly removed from a cart and their availability is updated in the marketplace.
        """
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")


        self.marketplace.publish(producer_id, "id2")
        self.marketplace.publish(producer_id, "id2")

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "id1")
        self.marketplace.add_to_cart(cart_id, "id2")
        self.assertEqual(len(self.marketplace.products_dict), 1)

        self.marketplace.remove_from_cart(cart_id, "id1")
        self.assertEqual(len(self.marketplace.products_dict), 2)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 1)
        self.assertFalse("id1" in self.marketplace.carts[cart_id])

        self.marketplace.remove_from_cart(cart_id, "id2")
        self.assertEqual(len(self.marketplace.products_dict), 2)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 0)
        self.assertFalse("id2" in self.marketplace.carts[cart_id])

    def test_place_order(self):
        """
        @brief Tests the `place_order` method and its impact on producer inventory.
        Functional Utility: Verifies that placing an order correctly processes the cart, updates producer-specific product counts, and returns the purchased items.
        """
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, "id1")
        self.marketplace.publish(producer_id, "id2")
        self.assertEqual(self.marketplace.size_per_producer[producer_id], 2)

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "id1")

        expected_products = ["id1"]
        products = self.marketplace.place_order(cart_id)
        self.assertEqual(self.marketplace.size_per_producer[producer_id], 1)
        self.assertEqual(expected_products, products)

    def test_get_consumer_lock(self):
        """
        @brief Tests that the correct consumer lock object is returned.
        Functional Utility: Ensures that the marketplace provides the intended lock for consumer synchronization.
        """
        self.assertEqual(self.marketplace.consumer_lock, self.marketplace.get_consumer_lock())


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    @brief Represents a producer that continuously publishes products to the marketplace.
    Functional Utility: Manages the product supply side of the e-commerce simulation, ensuring products are made available in the marketplace according to a defined schedule.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.
        @param products: A list of products to publish, including product ID, quantity, and a sleep time after publishing.
        @param marketplace: Reference to the shared `Marketplace` instance.
        @param republish_wait_time: Time to wait before retrying to publish if the marketplace is full for this producer.
        @param **kwargs: Additional keyword arguments, including the thread name.
        Functional Utility: Sets up the producer with its inventory, access to the marketplace, and retry logic.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        @brief Executes the producer's publishing logic.
        Functional Utility: Continuously registers with the marketplace and attempts to publish its products, handling potential delays due to marketplace capacity.
        """
        # Functional Utility: Registers the producer with the marketplace and obtains a unique ID.
        producer_id = self.marketplace.register_producer()
        # Block Logic: Enters an infinite loop to continuously publish products.
        while True:
            # Block Logic: Iterates through the list of products this producer can supply.
            for product in self.products:
                # Block Logic: Publishes the specified quantity of each product.
                for _ in range(0, product[1]):
                    # Invariant: Product not successfully published.
                    # Functional Utility: Continuously attempts to publish the product until successful, pausing if the marketplace capacity for this producer is reached.
                    while self.marketplace.publish(producer_id, product[0]) is not True:
                        # Functional Utility: Pauses execution before retrying to publish.
                        sleep(self.republish_wait_time)
                    # Functional Utility: Pauses after publishing a product, simulating production time.
                    sleep(product[2])
