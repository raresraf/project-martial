"""
@bf7e78c0-af21-48be-9d51-edc8876d4b81/consumer.py
@brief Producer-Consumer simulation for a marketplace system with multi-threaded inventory management.
* Algorithm: Concurrent asynchronous task processing with exponential/fixed backoff for resource contention.
* Functional Utility: Facilitates a virtual market where producers publish goods and consumers manage carts and place orders.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    @brief Consumer entity that interacts with the marketplace to acquire products.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes the consumer with its target shopping carts and marketplace connection.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        @brief Main consumer execution loop processing a sequence of shopping carts.
        Algorithm: Iterative operation processing (add/remove) followed by a final order placement.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for oper in cart:
                type_of_operation = oper["type"]
                prod = oper["product"]
                quantity = oper["quantity"]
                
                # Logic: Dispatches marketplace interaction based on the requested operation type.
                if type_of_operation == "add":
                    self.add_cart(cart_id, prod, quantity)
                elif type_of_operation == "remove":
                    self.remove_cart(cart_id, prod, quantity)
            
            # Post-condition: Completes the transaction and displays the results.
            p_purchased = self.marketplace.place_order(cart_id)
            for prod in p_purchased:
                print(f"{self.getName()} bought {prod}")

    def add_cart(self, cart_id, product_id, quantity):
        """
        @brief Attempts to add a specific quantity of a product to the cart.
        Logic: Uses a busy-wait loop with retry backoff to handle temporary inventory depletion.
        """
        for _ in range(quantity):
            while True:
                added = self.marketplace.add_to_cart(cart_id, product_id)
                if added:
                    break
                # Functional Utility: Prevents excessive CPU usage during inventory unavailability.
                sleep(self.retry_wait_time)

    def remove_cart(self, cart_id, product_id, quantity):
        """
        @brief Attempts to remove a specific quantity of a product from the cart.
        Logic: Busy-wait loop with retry backoff to handle cart modification constraints.
        """
        for _ in range(quantity):
            while True:
                removed = self.marketplace.remove_from_cart(cart_id, product_id)
                if removed:
                    break
                sleep(self.retry_wait_time)


from threading import Lock
import unittest
import sys
# Logic: Dynamic path injection to import product-specific definitions.
sys.path.insert(1, './tema')
import product as produs

class Marketplace:
    """
    @brief Centralized hub for product inventory and transaction coordination.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the marketplace with capacity constraints and internal storage.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0
        self.cart_id = 0
        self.queues = [] # Domain: Inventory storage partitioned by producer.
        self.carts = []  # Domain: Active transactions.
        self.mutex = Lock()
        self.products_dict = {} # Intent: Maps products to their respective producer IDs for efficient removal.

    def register_producer(self):
        """
        @brief Onboards a new producer and allocates an inventory queue.
        Invariant: Uses mutex to ensure thread-safe increment of producer IDs.
        """
        self.mutex.acquire()
        producer_id = self.producer_id
        self.producer_id += 1
        self.queues.append([])
        self.mutex.release()
        return str(producer_id)

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to add a product to the marketplace inventory.
        Pre-condition: Producer's queue must not exceed queue_size_per_producer.
        """
        index_prod = int(producer_id)
        if len(self.queues[index_prod]) == self.queue_size_per_producer:
            return False
        self.queues[index_prod].append(product)
        self.products_dict[product] = index_prod
        return True

    def new_cart(self):
        """
        @brief Generates a new unique cart ID for a consumer transaction.
        """
        self.mutex.acquire()
        cart_id = self.cart_id
        self.cart_id += 1
        self.mutex.release()
        self.carts.append([])
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Transfers a product from producer inventory to a consumer cart.
        Logic: Linear search across all producer queues to find and consume the item.
        """
        prod_in_queue = False
        for queue in self.queues:
            if product in queue:
                prod_in_queue = True
                queue.remove(product)
                break
        if not prod_in_queue:
            return False
        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Returns a product from a consumer cart back to its producer's inventory.
        Pre-condition: Target producer's queue must have available capacity.
        """
        if product not in self.carts[cart_id]:
            return False
        index_producer = self.products_dict[product]
        if len(self.queues[index_producer]) == self.queue_size_per_producer:
            return False
        self.carts[cart_id].remove(product)
        self.queues[index_producer].append(product)
        return True

    def place_order(self, cart_id):
        """
        @brief Finalizes a transaction and returns the list of purchased products.
        """
        cart_product_list = self.carts[cart_id]
        self.carts[cart_id] = []
        return cart_product_list

class TestMarketplace(unittest.TestCase):
    """
    @brief Unit tests for validating Marketplace functionality.
    """
    
    def setUp(self):
        self.marketplace = Marketplace(4)

    def test_register_producer(self):
        self.assertEqual(self.marketplace.register_producer(), str(0))
        self.assertNotEqual(self.marketplace.register_producer(), str(3))
        self.assertEqual(self.marketplace.register_producer(), str(2))

    def test_publish(self):
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))
        self.assertFalse(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))

    def test_new_cart(self):
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.new_cart(), 1)

    def test_add_to_cart(self):
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal"))
        self.assertTrue(self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal")))
        self.assertFalse(self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal")))

    def test_remove_from_cart(self):
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal"))
        self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal"))
        self.assertTrue(self.marketplace.remove_from_cart(0, produs.Tea("Linden", 9, "Herbal")))
        self.assertFalse(self.marketplace.remove_from_cart(0, produs.Tea("Linden", 9, "Herbal")))

    def test_place_order(self):
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal"))
        self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal"))
        self.assertEqual([produs.Tea("Linden", 9, "Herbal")], self.marketplace.place_order(0))


from threading import Thread
from time import sleep

class Producer(Thread):
    """
    @brief Producer entity that generates and publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes the producer with its product line and marketplace affinity.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        @brief Continuous production loop.
        Algorithm: Iterative product generation and publication based on requested quantities and production times.
        """
        while True:
            for product in self.products:
                quantity = product[1]
                for _ in range(0, quantity):
                    self.publish_product(product[0], product[2])

    def publish_product(self, product, production_time):
        """
        @brief Handles the publication of a single item unit.
        Logic: Busy-wait with backoff to handle inventory capacity limits.
        """
        while True:
            published = self.marketplace.publish(self.producer_id, product)
            if published:
                # Domain: Production Latency - Simulates the time taken to create the item.
                sleep(production_time)
                break
            # Functional Utility: Throttles retry attempts during capacity saturation.
            sleep(self.republish_wait_time)
