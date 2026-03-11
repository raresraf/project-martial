"""
@file consumer.py
@brief Implements a multi-threaded producer-consumer marketplace simulation, including unit tests.

This module defines classes for:
- `Consumer`: Simulates entities that purchase products from the marketplace.
- `Marketplace`: Acts as a central exchange where producers publish products and consumers manage carts.
- `Producer`: Simulates entities that generate and publish products to the marketplace.
- `TestMarketplace`: A unit test suite for the `Marketplace` class.

The simulation uses Python's `threading` module for concurrent operations and
employs locks and semaphores to ensure thread safety when managing shared resources
like product queues and shopping carts.
"""

from threading import Thread
from time import sleep
from threading import Lock
import unittest
import sys
sys.path.insert(1, './tema')
import product as produs


class Consumer(Thread):
    """
    @brief Simulates a consumer entity that interacts with the marketplace
           to add and remove products from its shopping carts.

    Consumers operate as separate threads, processing a list of predefined
    cart operations (add/remove products). They handle retries for failed
    operations and ultimately place orders for their accumulated products.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.

        @param carts: A list of cart operations (dictionaries) to be performed by this consumer.
                      Each operation specifies quantity, type (add/remove), and product.
        @param marketplace: The Marketplace instance to interact with.
        @param retry_wait_time: Time in seconds to wait before retrying a failed cart operation.
        @param kwargs: Additional keyword arguments passed to the Thread constructor.
        """

        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        @brief The main execution method for the Consumer thread.

        Iterates through the assigned carts, creates a new cart in the marketplace,
        and then processes each operation (add/remove product) for that cart.
        Includes a retry mechanism for failed operations. Once all operations
        for a cart are attempted, the order is placed and purchased items are printed.
        """

        # Block Logic: Iterate through each cart assigned to this consumer.
        for cart in self.carts:
            cart_id = self.marketplace.new_cart() # Create a new cart for the current set of operations.
            # Block Logic: Process each operation (add or remove) within the current cart.
            for oper in cart:
                type_of_operation = oper["type"]
                prod = oper["product"]
                quantity = oper["quantity"]
                # Conditional Logic: Distinguish between add and remove operations.
                if type_of_operation == "add":
                    self.add_cart(cart_id, prod, quantity)
                elif type_of_operation == "remove":
                    self.remove_cart(cart_id, prod, quantity)
            # Functional Utility: Place the order for the products accumulated in the cart.
            p_purchased = self.marketplace.place_order(cart_id)
            # Block Logic: Print details of each product successfully purchased.
            for prod in p_purchased:
                print(f"{self.getName()} bought {prod}")

    def add_cart(self, cart_id, product_id, quantity):
        """
        @brief Attempts to add a specified quantity of a product to a consumer's cart.

        This method includes a retry mechanism: if adding the product fails (e.g., product
        not available), it waits `retry_wait_time` and tries again until successful.
        @param cart_id: The ID of the consumer's cart.
        @param product_id: The ID of the product to add.
        @param quantity: The number of units of the product to add.
        """

        # Block Logic: Attempt to add the specified quantity of the product.
        for _ in range(quantity):
            # Block Logic: Loop until the product is successfully added.
            while True:
                added = self.marketplace.add_to_cart(cart_id, product_id)
                if added:
                    break # Exit loop if product was added.
                sleep(self.retry_wait_time) # Wait before retrying.

    def remove_cart(self, cart_id, product_id, quantity):
        """
        @brief Attempts to remove a specified quantity of a product from a consumer's cart.

        This method includes a retry mechanism: if removing the product fails,
        it waits `retry_wait_time` and tries again until successful.
        @param cart_id: The ID of the consumer's cart.
        @param product_id: The ID of the product to remove.
        @param quantity: The number of units of the product to remove.
        """

        # Block Logic: Attempt to remove the specified quantity of the product.
        for _ in range(quantity):
            # Block Logic: Loop until the product is successfully removed.
            while True:
                removed = self.marketplace.remove_from_cart(cart_id, product_id)
                if removed:
                    break # Exit loop if product was removed.
                sleep(self.retry_wait_time) # Wait before retrying.


class Marketplace:
    """
    @brief Manages products, producers, and consumer carts in a thread-safe manner.

    The Marketplace acts as a central hub for the simulation. It provides functionalities
    for producers to publish products, consumers to create carts, add/remove products,
    and place orders. It uses threading locks to ensure atomicity and consistency
    of shared data structures during concurrent access.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace with specified queue size and sets up internal data structures.

        @param queue_size_per_producer: The maximum number of products a single producer can have
                                        in the marketplace's available products list at any given time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0 # Counter for assigning unique producer IDs.
        self.cart_id = 0     # Counter for assigning unique cart IDs.
        self.queues = []     # List of lists, each sublist represents a producer's queue of published products.
        self.carts = []      # List of lists, each sublist represents a consumer's cart.
        self.mutex = Lock()  # Global lock to protect critical sections for Marketplace operations.
        self.products_dict = {} # Maps product instances to the producer ID that published them.

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace, assigning it a unique ID.

        Each producer is allocated an empty queue to track its current stock in the marketplace.
        This operation is protected by a mutex to ensure unique ID assignment in a multi-threaded environment.
        @return: The unique string ID assigned to the new producer.
        """

        self.mutex.acquire() # Acquire lock for thread-safe ID generation and queue allocation.
        producer_id = self.producer_id
        self.producer_id += 1
        self.queues.append([]) # Assign a new empty queue for the registered producer.
        self.mutex.release() # Release lock.
        return str(producer_id) # Return producer ID as a string.

    def publish(self, producer_id, product):
        """
        @brief Publishes a product to the marketplace, making it available for consumers.

        The product is added only if the producer's queue for available products
        does not exceed the `queue_size_per_producer` limit. This operation is thread-safe.
        @param producer_id: The ID of the producer publishing the product.
        @param product: The product to be published.
        @return: True if the product was successfully published, False otherwise (e.g., queue full).
        """

        index_prod = int(producer_id) # Convert producer_id to integer index.
        # Conditional Logic: Check if the producer's queue is full.
        if len(self.queues[index_prod]) == self.queue_size_per_producer:
            return False # Cannot publish if queue is full.
        self.queues[index_prod].append(product) # Add product to the producer's queue.
        self.products_dict[product] = index_prod # Record which producer published this product.
        return True # Publication successful.

    def new_cart(self):
        """
        @brief Creates a new, empty shopping cart for a consumer and returns its unique ID.

        This operation is protected by a mutex to ensure thread-safe cart ID generation
        and initialization of the cart.
        @return: The unique integer ID of the newly created cart.
        """

        self.mutex.acquire() # Acquire lock for thread-safe ID generation and cart allocation.
        cart_id = self.cart_id
        self.cart_id += 1
        self.mutex.release() # Release lock.
        self.carts.append([]) # Assign a new empty cart for the consumer.
        return cart_id # Return the new cart's ID.

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a specified product to a consumer's cart.

        This operation is thread-safe using implicit locking through queue manipulation.
        It checks product availability by searching through all producer queues.
        If found, the product is moved from the producer's queue to the consumer's cart.
        @param cart_id: The ID of the consumer's cart.
        @param product: The product to add to the cart.
        @return: True if the product was successfully added, False if the product is not available.
        """

        prod_in_queue = False
        # Block Logic: Search through all producer queues to find the product.
        for queue in self.queues:
            if product in queue:
                prod_in_queue = True
                queue.remove(product) # Remove product from producer's queue.
                break # Product found and removed, no need to check other queues.
        if not prod_in_queue:
            return False # Product not found in any queue.
        self.carts[cart_id].append(product) # Add product to the consumer's cart.
        return True # Product added to cart successfully.

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a specified product from a consumer's cart and returns it to the marketplace.

        This operation checks if the product is in the cart and if the original producer's
        queue has capacity to receive it back. It is thread-safe due to the implicit
        atomic operations on lists and conditional checks.
        @param cart_id: The ID of the consumer's cart.
        @param product: The product to remove from the cart.
        @return: True if the product was successfully removed and returned, False otherwise.
        """

        # Conditional Logic: Check if the product is actually in the specified cart.
        if product not in self.carts[cart_id]:
            return False # Product not in cart.
        index_producer = self.products_dict[product] # Get the producer ID for this product.
        # Conditional Logic: Check if the producer's queue has space to take the product back.
        if len(self.queues[index_producer]) == self.queue_size_per_producer:
            return False # Producer's queue is full, cannot return product.


        self.carts[cart_id].remove(product) # Remove product from the consumer's cart.
        self.queues[index_producer].append(product) # Return product to the producer's queue.
        return True # Product removed and returned successfully.

    def place_order(self, cart_id):
        """
        @brief Finalizes the order for a given cart, clearing it and returning the purchased products.

        The products in the cart are considered "bought".
        @param cart_id: The ID of the cart to place the order for.
        @return: A list of products that were in the placed order.
        """

        cart_product_list = self.carts[cart_id] # Get the list of products in the cart.
        self.carts[cart_id] = [] # Clear the cart after placing the order.
        return cart_product_list # Return the list of purchased products.

class TestMarketplace(unittest.TestCase):
    """
    @brief Unit test suite for the `Marketplace` class.

    This class contains various test methods to ensure the correct functionality
    and thread safety of the `Marketplace`'s core operations.
    """

    def setUp(self):
        """
        @brief Sets up the test environment before each test method is executed.

        Initializes a new `Marketplace` instance with a specific queue size.
        """

        self.marketplace = Marketplace(4) # Initialize Marketplace with queue size 4.

    def test_register_producer(self):
        """
        @brief Tests the `register_producer` method to ensure unique IDs are assigned.
        """

        self.assertEqual(self.marketplace.register_producer(), str(0))
        self.assertNotEqual(self.marketplace.register_producer(), str(3))
        self.assertEqual(self.marketplace.register_producer(), str(2))
        self.assertNotEqual(self.marketplace.register_producer(), str(0))
        self.assertNotEqual(self.marketplace.register_producer(), str(3))
        self.assertNotEqual(self.marketplace.register_producer(), str(2))

    def test_publish(self):
        """
        @brief Tests the `publish` method, including queue size limits for producers.
        """

        self.marketplace.register_producer() # Register producer with ID 0.
        self.marketplace.register_producer() # Register producer with ID 1.
        # Publish products, checking if the queue limit is respected.
        self.assertTrue(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))
        self.assertFalse(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal"))) # Should fail, queue full.
        self.assertTrue(self.marketplace.publish(str(1), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(1), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(1), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(1), produs.Tea("Linden", 9, "Herbal")))
        self.assertFalse(self.marketplace.publish(str(1), produs.Tea("Linden", 9, "Herbal"))) # Should fail, queue full.
        self.assertFalse(self.marketplace.publish(str(1), produs.Tea("Linden", 9, "Herbal")))

    def test_new_cart(self):
        """
        @brief Tests the `new_cart` method to ensure unique cart IDs are generated.
        """

        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertNotEqual(self.marketplace.new_cart(), 3)
        self.assertEqual(self.marketplace.new_cart(), 2)
        self.assertNotEqual(self.marketplace.new_cart(), 0)
        self.assertNotEqual(self.marketplace.new_cart(), 3)
        self.assertNotEqual(self.marketplace.new_cart(), 2)

    def test_add_to_cart(self):
        """
        @brief Tests the `add_to_cart` method, verifying product availability and addition.
        """

        self.marketplace.register_producer() # Register a producer.
        self.marketplace.new_cart() # Create a new cart.
        self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")) # Publish a product.
        self.assertTrue(self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal"))) # Add product to cart.
        self.assertFalse(self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal"))) # Should fail, product moved from queue.

    def test_remove_from_cart(self):
        """
        @brief Tests the `remove_from_cart` method, ensuring products are correctly removed and returned.
        """

        self.marketplace.register_producer() # Register a producer.
        self.marketplace.new_cart() # Create a new cart.
        self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")) # Publish a product.
        self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal")) # Add product to cart.
        self.assertTrue(self.marketplace.remove_from_cart(0, produs.Tea("Linden", 9, "Herbal"))) # Remove product.
        self.assertFalse(self.marketplace.remove_from_cart(0, produs.Tea("Linden", 9, "Herbal"))) # Should fail, product no longer in cart.

    def test_place_order(self):
        """
        @brief Tests the `place_order` method, verifying that the cart is cleared and purchased products are returned.
        """

        self.marketplace.register_producer() # Register a producer.
        self.marketplace.new_cart() # Create a new cart.
        self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")) # Publish a product.
        self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal")) # Add product to cart.
        # Verify that placing the order returns the correct list of products.
        self.assertEqual([produs.Tea("Linden", 9, "Herbal")], self.marketplace.place_order(0))


class Producer(Thread):
    """
    @brief Simulates a producer entity that continuously generates and publishes products
           to the marketplace.

    Producers operate as separate threads, each responsible for a specific set of products.
    They attempt to publish products to the marketplace, respecting marketplace limits,
    and include a mechanism for retrying publication if the marketplace is full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.

        @param products: A list of products this producer will publish. Each item is a tuple
                         (product_name, number_of_products_to_publish, publish_wait_time_after_success).
        @param marketplace: The Marketplace instance to interact with.
        @param republish_wait_time: Time in seconds to wait before retrying to publish a product
                                    if the marketplace's queue is full.
        @param kwargs: Additional keyword arguments passed to the Thread constructor.
        """

        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        # Register the producer with the marketplace to get a unique ID.
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        @brief The main execution method for the Producer thread.

        It continuously attempts to publish its predefined list of products to the marketplace.
        It iterates through each product specification and tries to publish the required
        number of units, handling delays as specified.
        """

        # Block Logic: Main production loop, ensuring continuous publishing of products.
        while True:
            # Block Logic: Iterate through each type of product this producer is responsible for.
            for product_info in self.products:
                quantity = product_info[1] # Number of units of this product to publish.
                # Block Logic: Publish the specified quantity of the current product.
                for _ in range(0, quantity):
                    self.publish_product(product_info[0], product_info[2]) # Publish one unit.

    def publish_product(self, product, production_time):
        """
        @brief Attempts to publish a single unit of a product to the marketplace.

        If successful, it waits for a specified `production_time`. If unsuccessful
        (e.g., marketplace queue is full), it waits for `republish_wait_time` before
        the next retry.
        @param product: The product to publish.
        @param production_time: The time to wait (in seconds) after a successful publication.
        @return: This method implicitly returns after a successful publish due to the `break` statement.
        """

        # Block Logic: Loop until the product is successfully published.
        while True:
            published = self.marketplace.publish(self.producer_id, product)
            if published:
                sleep(production_time) # Wait for `production_time` after successful publication.
                break # Exit loop if product was published.
            sleep(self.republish_wait_time) # Wait before retrying if publication failed.