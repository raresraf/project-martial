


"""
@8e606774-89c1-4878-bee9-14c8897e27ba/consumer.py
@brief Implements consumer and producer agents interacting with a shared marketplace.

This module defines the core components for a multi-threaded simulation of
e-commerce interactions. It includes a `Consumer` thread that simulates
buying actions (adding/removing items from a cart and placing orders),
a `Producer` thread that simulates selling actions (publishing products),
and a `Marketplace` class that orchestrates these interactions with thread-safe
operations. Unit tests for the `Marketplace` are also included.

Classes:
- Consumer: A thread simulating a buyer's interactions with the marketplace.
- Marketplace: Manages product publishing, cart operations, and orders in a thread-safe manner.
- Producer: A thread simulating a seller's actions of publishing products.
- Product, Tea, Coffee: Data classes for defining product types.

Domain: Concurrent Programming, E-commerce Simulation, Multi-threading.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    @brief Simulates a consumer agent interacting with a marketplace.

    This thread creates a new shopping cart, adds and removes products
    based on a predefined list of operations, and eventually places an order.
    It handles retries if marketplace operations fail.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a new Consumer thread.

        @param carts: A list of cart operations for this consumer to perform.
                      Each cart is a list of dictionaries, where each dictionary
                      represents an operation (e.g., {'type': 'add', 'product': product_obj, 'quantity': 1}).
        @param marketplace: The Marketplace instance to interact with.
        @param retry_wait_time: The time in seconds to wait before retrying a failed operation.
        @param kwargs: Arbitrary keyword arguments passed to the Thread.__init__ method.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def get_name(self):
        """
        @brief Returns the name of the consumer thread.

        @return The name of the thread.
        """
        return self.name

    def run(self):
        """
        @brief The main execution loop for the Consumer thread.

        This method iterates through the assigned carts, performs add/remove operations
        on the marketplace, retries failed operations after a delay, and finally
        places the order.
        """
        for cart in self.carts:
            # Block Logic: Creates a new shopping cart in the marketplace for this consumer.
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                quantity = 0
                # Block Logic: Continuously attempts to perform the operation until the desired
                # quantity of the product is added or removed.
                while quantity < operation['quantity']:
                    # Pre-condition: Checks the type of operation (add or remove).
                    if operation['type'] == 'add':
                        result = self.marketplace.add_to_cart(cart_id, operation['product'])
                    if operation['type'] == 'remove':
                        result = self.marketplace.remove_from_cart(cart_id, operation['product'])

                    # Invariant: If the operation was successful (result is not None and not False),
                    # increment the quantity. Otherwise, wait and retry.
                    if result is None or result is True:
                        quantity += 1
                    else:
                        time.sleep(self.retry_wait_time) # Inline: Pauses for a short duration before retrying the operation.

            # Block Logic: Once all operations for a cart are complete, the order is placed.
            self.marketplace.place_order(cart_id)


from threading import currentThread, Lock
import unittest

class Marketplace:
    """
    @brief Simulates a central marketplace for producers and consumers.

    This class manages product listings, producer registrations, shopping carts,
    and order placement. It uses a Lock to ensure thread-safe access to shared
    data structures, preventing race conditions between concurrent producer and
    consumer operations.
    """
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes a new Marketplace instance.

        @param queue_size_per_producer: The maximum number of items a single producer
                                       can have listed in the marketplace at any time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        
        self.num_producers = 0 # Counter for assigning unique producer IDs.
        self.num_carts = 0     # Counter for assigning unique cart IDs.
        self.num_items_producer = [] # List to track the number of items published by each producer.

        self.producers = {} # Dictionary to store products and their associated producers.
                            # Key: product object, Value: producer ID.
        self.carts = {}     # Dictionary to store shopping carts.
                            # Key: cart ID, Value: list of products in the cart.

        self.lock = Lock() # Lock for protecting shared marketplace data during concurrent access.

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace and assigns a unique ID.

        @return The unique integer ID assigned to the newly registered producer.
        """
        # Block Logic: Ensures exclusive access to update producer count and assign ID.
        with self.lock:
            producer_id = self.num_producers
            self.num_producers += 1
            # Invariant: `num_items_producer` is extended to accommodate the new producer.
            self.num_items_producer.insert(producer_id, 0)

        return producer_id

    def publish(self, producer_id, product):
        """
        @brief Publishes a product to the marketplace by a specific producer.

        The product is published only if the producer has not exceeded its
        queue size limit.

        @param producer_id: The ID of the producer publishing the product.
        @param product: The product object to be published.
        @return True if the product was successfully published, False otherwise.
        """
        casted_producer_int = int(producer_id)

        # Pre-condition: Checks if the producer has reached their item limit.
        # If so, the product cannot be published.
        if self.num_items_producer[casted_producer_int] >= self.queue_size_per_producer:
            return False

        # Block Logic: Atomically updates the count of items for the producer and adds
        # the product to the marketplace's list of available products.
        self.num_items_producer[casted_producer_int] += 1
        self.producers[product] = casted_producer_int

        return True

    def new_cart(self):
        """
        @brief Creates a new shopping cart and assigns a unique cart ID.

        @return The unique integer ID of the newly created cart.
        """
        # Block Logic: Ensures exclusive access to update cart count and assign ID.
        with self.lock:
            self.num_carts = self.num_carts + 1
            cart_id = self.num_carts

        # Post-condition: Initializes an empty list for the new cart.
        self.carts[cart_id] = []
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a specified shopping cart.

        If the product is available in the marketplace, it's moved from
        the marketplace's inventory to the cart.

        @param cart_id: The ID of the cart to add the product to.
        @param product: The product object to add.
        @return True if the product was successfully added, False if not found.
        """
        # Block Logic: Ensures thread-safe access to marketplace products and cart updates.
        with self.lock:
            # Pre-condition: Checks if the product is currently listed in the marketplace.
            if self.producers.get(product) is None:
                return False

            # Functional Utility: Decrements the producer's item count and removes the product
            # from the marketplace's general listing.
            self.num_items_producer[self.producers[product]] -= 1
            producers_id = self.producers.pop(product)

        # Post-condition: Adds the product and its original producer's ID to the cart.
        self.carts[cart_id].append(product)
        self.carts[cart_id].append(producers_id)

        return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specified shopping cart and returns it to the marketplace.

        @param cart_id: The ID of the cart to remove the product from.
        @param product: The product object to remove.
        """
        # Pre-condition: Checks if the product exists in the specified cart.
        if product in self.carts[cart_id]:
            index = self.carts[cart_id].index(product)
            self.carts[cart_id].remove(product)
            # Functional Utility: Retrieves the producer ID associated with the removed product.
            producers_id = self.carts[cart_id].pop(index)
            # Post-condition: Returns the product to the marketplace listing and increments
            # the producer's item count.
            self.producers[product] = producers_id

            # Block Logic: Ensures thread-safe update of the producer's item count.
            with self.lock:
                self.num_items_producer[int(producers_id)] += 1

    def place_order(self, cart_id):
        """
        @brief Places an order for all items currently in the specified cart.

        Items are removed from the cart and the producer's item counts are adjusted.
        A message indicating the purchase is printed to the console.

        @param cart_id: The ID of the cart for which to place the order.
        @return A list of products that were in the placed order.
        """
        # Post-condition: Clears the cart after the order is placed.
        product_list = self.carts.pop(cart_id)

        # Block Logic: Iterates through the ordered products, prints a purchase message,
        # and decrements the corresponding producer's item count.
        for i in range(0, len(product_list), 2): # Invariant: Product list contains (product, producer_id) pairs.
            with self.lock:
                print(currentThread().get_name() + " bought " + str(product_list[i]))
                # Functional Utility: Adjusts the producer's inventory count after a purchase.
                self.num_items_producer[product_list[i + 1]] -= 1

        return product_list

class TestMarketplace(unittest.TestCase):
    """
    @brief Unit tests for the Marketplace class.

    These tests verify the core functionalities of the Marketplace,
    including product publishing limits, adding products to carts,
    creating new carts, and removing products from carts.
    """
    product = "Tea(name='Linden', price=9, type='Herbal')"
    product2 = "Tea(name='Linden', price=10, type='Herbal')"
    def setUp(self):
        """
        @brief Sets up the test environment before each test method.

        Initializes a new Marketplace instance with a queue size of 13.
        """
        self.marketplace = Marketplace(13)

    def test_publish_limit_fail(self):
        """
        @brief Tests that a producer cannot publish a product if its queue limit is reached.
        """
        producer_id = self.marketplace.queue_size_per_producer - 1
        self.marketplace.num_items_producer = [0] * self.marketplace.queue_size_per_producer
        self.marketplace.num_items_producer[producer_id] = self.marketplace.queue_size_per_producer

        self.assertFalse(self.marketplace.publish(str(producer_id), self.product),
            "Queue size per producer limit!")

    def test_publish_limit_success(self):
        """
        @brief Tests that a producer can publish a product if its queue limit is not yet reached.
        """
        producer_id = self.marketplace.queue_size_per_producer - 1
        self.marketplace.num_items_producer = [0] * self.marketplace.queue_size_per_producer
        self.marketplace.num_items_producer[producer_id] \
            = self.marketplace.queue_size_per_producer - 1

        self.assertTrue(self.marketplace.publish(str(producer_id), self.product), "Cannot publish!")

    def test_add_to_cart_fail(self):
        """
        @brief Tests that adding a non-existent product to a cart fails.
        """
        cart_id = self.marketplace.new_cart()

        self.assertFalse(self.marketplace.add_to_cart(cart_id, self.product),
            "Product shouldn't be found!")

    def test_add_to_cart_success(self):
        """
        @brief Tests that adding an existing product to a cart succeeds.
        """
        product = "Tea(name='Linden', price=9, type='Herbal')"
        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        self.marketplace.publish(str(producer_id), product)

        self.assertTrue(self.marketplace.add_to_cart(cart_id, product),
            "Product should be found!")

    def test_new_cart(self):
        """
        @brief Tests the creation of a new shopping cart.

        Verifies that the cart count increases and a new empty list is initialized for the cart.
        """
        cart_id = self.marketplace.new_cart()
        self.assertEqual(cart_id, self.marketplace.num_carts, "Number of carts not increased!")
        self.assertIsInstance(self.marketplace.carts[cart_id], type([]), "List not initialized!")

    def test_remove_from_cart(self):
        """
        @brief Tests the removal of a product from a shopping cart.

        Verifies that the product is no longer in the cart after removal.
        """
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(str(producer_id), self.product)
        self.marketplace.publish(str(producer_id), self.product2)
        self.marketplace.publish(str(producer_id), self.product)

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, self.product)
        self.marketplace.add_to_cart(cart_id, self.product)
        self.marketplace.add_to_cart(cart_id, self.product2)

        self.marketplace.remove_from_cart(cart_id, self.product2)

        self.assertNotIn(self.product2, self.marketplace.carts[cart_id], "Product not removed!")>>>> file: producer.py


class Producer(Thread):
    """
    @brief Simulates a producer agent that publishes products to a marketplace.

    This thread registers itself with the marketplace, and then continuously
    attempts to publish a predefined list of products, waiting and retrying
    if the marketplace is temporarily unable to accept more products from it.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a new Producer thread.

        @param products: A list of products (product object, quantity, wait time)
                         for this producer to publish.
        @param marketplace: The Marketplace instance to interact with.
        @param republish_wait_time: The time in seconds to wait before retrying to publish a product.
        @param kwargs: Arbitrary keyword arguments passed to the Thread.__init__ method.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = 0 # Unique ID assigned by the marketplace after registration.

    def run(self):
        """
        @brief The main execution loop for the Producer thread.

        This method registers the producer with the marketplace, then
        iterates through its list of products, attempting to publish them
        to the marketplace. It handles retries if publishing fails.
        """
        # Pre-condition: Registers the producer with the marketplace to obtain a unique ID.
        self.producer_id = self.marketplace.register_producer()

        # Block Logic: Main loop for continuously publishing products.
        while True:
            for product_info in self.products:
                produced = 0
                # Block Logic: Attempts to publish the product until the desired quantity is reached.
                while produced < product_info[1]:
                    # Functional Utility: Attempts to publish the product to the marketplace.
                    # The `product_info[0]` is the product object, and `str(self.producer_id)`
                    # is used to identify the publishing producer.
                    result = self.marketplace.publish(str(self.producer_id), product_info[0])

                    # Invariant: If publishing was successful, increment the count of produced items
                    # and wait for a specified time before attempting to publish the next item.
                    # Otherwise, wait and retry.
                    if result:
                        time.sleep(product_info[2]) # Inline: Pauses for a specified time after successfully publishing a product.
                        produced += 1
                    else:
                        time.sleep(self.republish_wait_time) # Inline: Pauses for a specified time before retrying to publish a product.


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base data class representing a generic product.

    Attributes:
    - name (str): The name of the product.
    - price (int): The price of the product.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Data class representing a type of tea, inheriting from Product.

    Attributes:
    - type (str): The type or variety of the tea (e.g., 'Herbal', 'Black', 'Green').
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Data class representing a type of coffee, inheriting from Product.

    Attributes:
    - acidity (str): The acidity level of the coffee.
    - roast_level (str): The roast level of the coffee.
    """
    acidity: str
    roast_level: str
