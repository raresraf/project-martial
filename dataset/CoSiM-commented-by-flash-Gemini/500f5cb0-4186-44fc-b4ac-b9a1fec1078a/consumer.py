"""
@500f5cb0-4186-44fc-b4ac-b9a1fec1078a/consumer.py
@brief Implements a multithreaded marketplace simulation with Consumers, Producers, and a Marketplace.

This module simulates an e-commerce system where producers supply products
to a marketplace, and consumers interact with the marketplace to add,
remove, and purchase products. It includes threading for concurrent
operations and unit tests for the marketplace functionality.
Synchronization mechanisms (e.g., threading.Lock) are used to ensure
thread-safe access to shared resources.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    @brief Represents a consumer in the marketplace simulation.

    Each Consumer operates as an independent thread, performing a series
    of add and remove operations on a shopping cart within the marketplace,
    and finally placing an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.

        @param carts: A list of shopping carts for this consumer. Each cart
                      is a list of product dictionaries, specifying "product",
                      "quantity", and "type" ("add" or "remove").
        @param marketplace: The shared Marketplace instance this consumer
                            will interact with.
        @param retry_wait_time: The time in seconds to wait before retrying
                                a failed cart operation (e.g., product
                                unavailable).
        @param kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """

        Thread.__init__(self, **kwargs)

        self.retry_wait_time = retry_wait_time
        self.carts = carts
        self.marketplace = marketplace


    def add(self, cart_id, product_id, qty):
        """
        @brief Adds a specified quantity of a product to the consumer's cart.

        This method repeatedly attempts to add the product until the desired
        quantity is met. If an attempt fails, it waits for `retry_wait_time`
        before trying again.

        @param cart_id: The ID of the consumer's cart.
        @param product_id: The ID of the product to add.
        @param qty: The quantity of the product to add.
        """

        # Initialize a counter for successfully added products.
        count = 0
        # Block Logic: Continuously attempt to add the product until the target quantity is reached.
        # Invariant: 'count' tracks the number of products successfully added.
        while count < qty:
            # Attempt to add the product to the cart.
            must_wait = not (self.marketplace.add_to_cart(cart_id, product_id))
            if must_wait:
                # If adding failed, wait before retrying.
                time.sleep(self.retry_wait_time)
            else:
                # If successful, increment the count.
                count += 1

    def rm(self, cart_id, product_id, qty):
        """
        @brief Removes a specified quantity of a product from the consumer's cart.

        This method repeatedly attempts to remove the product until the desired
        quantity is met.

        @param cart_id: The ID of the consumer's cart.
        @param product_id: The ID of the product to remove.
        @param qty: The quantity of the product to remove.
        """

        # Initialize a counter for successfully removed products.
        count = 0
        # Block Logic: Continuously attempt to remove the product until the target quantity is met.
        # Invariant: 'count' tracks the number of products successfully removed.
        while count < qty:
            # Remove the product from the cart.
            self.marketplace.remove_from_cart(cart_id, product_id)
            count += 1

    def run(self):
        """
        @brief The main execution logic for the Consumer thread.

        This method registers a new cart, iterates through its assigned shopping
        carts, performs add/remove operations for each product, and then places
        the final order.
        """
        # Register a new cart for this consumer in the marketplace.
        cart_id = self.marketplace.new_cart()

        # Block Logic: Iterate through each shopping cart in the consumer's list.
        for cart in self.carts:
            # Block Logic: Iterate through each product within the current cart.
            for product in cart:
                # Extract product details from the dictionary.
                product_id = product.get("product")
                qty = product.get("quantity")
                op = product.get("type")

                # Block Logic: Perform 'add' or 'remove' operation based on product type.
                if op == "add":
                    self.add(cart_id, product_id, qty)
                elif op == "remove":
                    self.rm(cart_id, product_id, qty)

        # Place the final order for the cart.
        order = self.marketplace.place_order(cart_id)
        # Block Logic: Print the details of the products bought in the order.
        for product in order:
            # Print the consumer's name and the product bought.
            print(self.name, "bought", product)


from threading import Lock
import unittest


class TestMarketplace(unittest.TestCase):
    """
    @brief Unit tests for the Marketplace class.

    This class provides a suite of tests to verify the core functionalities
    of the Marketplace, including producer registration, product publishing,
    cart creation, and adding/removing items from carts.
    """
    def setUp(self):
        """
        @brief Set up method called before each test.

        Initializes a Marketplace instance, registers a producer, defines a
        sample product, and creates a new cart for testing.
        """
        # Initialize a Marketplace with a queue size of 10 per producer.
        self.marketplace = Marketplace(10)

        # Register a producer and store its ID.
        self.producer_id = self.marketplace.register_producer()

        # Define a sample product dictionary.
        self.product = {"product_type": "Coffee"}
        self.product["name"] = "Indonezia"
        self.product["acidity"] = 5.05
        self.product["roast_level"] = "MEDIUM"
        self.product["price"] = 1

        # Create a new cart and store its ID.
        self.cart_id = self.marketplace.new_cart()

    def tearDown(self):
        """
        @brief Tear down method called after each test.

        Resets test variables and marketplace state to ensure clean execution
        of subsequent tests.
        """
        self.product = None
        self.producer_id = -1
        self.cart_id = -1
        # Reset internal marketplace state, crucial for isolated test runs.
        self.marketplace.producers = {}
        self.marketplace.carts = {}
        self.marketplace = None

    def test_register_producer(self):
        """
        @brief Tests the `register_producer` method.

        Verifies that a valid producer ID is returned and that the producer's
        entry is correctly initialized in the marketplace.
        """
        # Assert that the producer ID is greater than 0.
        self.assertGreater(self.producer_id, 0)
        # Assert that the newly registered producer has an empty list of products.
        self.assertListEqual(self.marketplace.producers.get(self.producer_id), [])

    def test_publish(self):
        """
        @brief Tests the `publish` method.

        Verifies that a product can be successfully published by a producer.
        """
        # Attempt to publish a product.
        ret = self.marketplace.publish(self.producer_id, self.product)

        # Assert that the publish operation was successful.
        self.assertTrue(ret)

    def test_new_cart(self):
        """
        @brief Tests the `new_cart` method.

        Verifies that a valid cart ID is returned and that the cart's entry
        is correctly initialized in the marketplace.
        """
        # Assert that the cart ID is greater than 0.
        self.assertGreater(self.cart_id, 0)
        # Assert that the newly created cart has an empty list of products.
        self.assertListEqual(self.marketplace.carts.get(self.cart_id), [])

    def test_add_to_cart(self):
        """
        @brief Tests the `add_to_cart` method.

        Verifies that a product can be successfully added to a cart after
        being published.
        """
        # Publish a product.
        ret = self.marketplace.publish(self.producer_id, self.product)
        # Assert successful publish.
        self.assertTrue(ret)
        # Add the published product to the cart.
        ret = self.marketplace.add_to_cart(self.cart_id, self.product)
        # Assert successful add to cart.
        self.assertTrue(ret)

    def test_remove_from_cart(self):
        """
        @brief Tests the `remove_from_cart` method.

        Verifies that a product can be successfully removed from a cart
        and that the cart becomes empty afterward.
        """
        # Publish a product.
        ret = self.marketplace.publish(self.producer_id, self.product)
        # Assert successful publish.
        self.assertTrue(ret)
        # Add the published product to the cart.
        ret = self.marketplace.add_to_cart(self.cart_id, self.product)
        # Assert successful add to cart.
        self.assertTrue(ret)
        # Remove the product from the cart.
        self.marketplace.remove_from_cart(self.cart_id, self.product)
        # Assert that the cart is now empty.
        self.assertListEqual(self.marketplace.carts.get(self.cart_id), [])

    def test_place_order(self):
        """
        @brief Tests the `place_order` method.

        Verifies that placing an order returns the correct list of products
        that were in the cart.
        """
        order = self.marketplace.place_order(self.cart_id)
        # Assert that the order returned is not None.
        self.assertIsNotNone(order)
        # Assert that the order list matches the products in the cart.
        self.assertListEqual(order, self.marketplace.carts.get(self.cart_id))


class Marketplace:
    """
    @brief Manages products, producers, and consumer carts in a simulated
           e-commerce environment.

    This class provides thread-safe operations for producers to publish
    products, and for consumers to create carts, add/remove products,
    and place orders. It uses locks to ensure data consistency in a
    concurrent setting.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.

        @param queue_size_per_producer: The maximum number of products
                                        a single producer can have published
                                        in the marketplace at any given time.
        """

        self.queue_size_per_producer = queue_size_per_producer

        # Counter for assigning unique producer IDs.
        self.producer_count = 0

        # Dictionary to store products published by each producer.
        # Key: producer_id, Value: List of products published by that producer.
        self.producers = {}

        # Lock to protect concurrent access to producer-related data
        # during publishing operations.
        self.publishing_lock = Lock()

        # Counter for assigning unique consumer (cart) IDs.
        self.consumer_count = 0

        # Dictionary to store consumer carts.
        # Key: cart_id, Value: List of products in that cart.
        self.carts = {}

        # Lock to protect concurrent access to cart-related data
        # during add/remove operations.
        self.cart_lock = Lock()


    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.

        Assigns a unique ID to the producer and initializes an empty
        product list for it.

        @return: The unique integer ID assigned to the new producer.
        """

        # Block Logic: Increment producer count and initialize a new producer entry.
        # Critical Section: Protected implicitly by assumption of single-threaded
        #                   producer registration or external synchronization.
        self.producer_count += 1
        self.producers[self.producer_count] = []

        return self.producer_count

    def publish(self, producer_id, product):
        """
        @brief Publishes a product from a producer to the marketplace.

        The product is added to the marketplace's inventory if the
        producer has not exceeded its `queue_size_per_producer` limit.
        This operation is thread-safe.

        @param producer_id: The ID of the producer publishing the product.
        @param product: The product (e.g., dictionary representing product details)
                        to be published.
        @return: True if the product was successfully published, False otherwise.
        """

        successful_publish = False

        # Block Logic: Acquire lock to ensure atomic access to producer's product list.
        self.publishing_lock.acquire()

        # Get the list of products currently published by this producer.
        products = self.producers.get(producer_id)
        queue_size = len(products)

        # Check if the producer has space in its publishing queue.
        if queue_size < self.queue_size_per_producer:
            successful_publish = True
            # Add the product to the producer's list of published products.
            self.producers.get(producer_id).append(product)

        # Release the lock.
        self.publishing_lock.release()

        return successful_publish

    def new_cart(self):
        """
        @brief Creates a new shopping cart for a consumer.

        Assigns a unique ID to the cart and initializes an empty list
        of products for it.

        @return: The unique integer ID assigned to the new cart.
        """

        # Block Logic: Increment consumer count and initialize a new cart entry.
        # Critical Section: Protected implicitly by assumption of single-threaded
        #                   cart creation or external synchronization.
        self.consumer_count += 1
        self.carts[self.consumer_count] = []

        return self.consumer_count

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a specified consumer's cart.

        This method attempts to find the product among all published products.
        If found, it is moved from the producer's inventory to the consumer's cart.
        This operation is thread-safe.

        @param cart_id: The ID of the cart to which the product should be added.
        @param product: The product to be added.
        @return: True if the product was successfully added, False otherwise.
        """

        # Initialize product_owner to None; it will store the ID of the producer
        # who owns the product if found.
        product_owner = None

        # Block Logic: Acquire lock to ensure atomic access to cart data during modification.
        self.cart_lock.acquire()

        # Block Logic: Iterate through all producers to find the product.
        # Invariant: Iterates through each producer and their products to locate the target product.
        # Note: Iterating `list(self.producers.keys())` creates a snapshot to prevent
        #       "dictionary changed size during iteration" errors in a multi-threaded context.
        for curr_producer in list(self.producers.keys()):
            # Block Logic: Iterate through products of the current producer.
            for curr_product in self.producers.get(curr_producer):
                if product == curr_product:
                    product_owner = curr_producer
                    break # Product found, exit inner loop.
            if product_owner is not None:
                break # Product found, exit outer loop.

        # If the product was found and its owner identified.
        if product_owner is not None:
            # Remove the product from the producer's inventory.
            self.producers.get(product_owner).remove(product)
            # Add the product and its owner's ID to the consumer's cart.
            self.carts.get(cart_id).append([product, product_owner])

        # Release the lock.
        self.cart_lock.release()

        # Return True if the product was found and added, False otherwise.
        ret = product_owner is not None
        return ret

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specified consumer's cart.

        If the product is found in the cart, it is moved back to the
        original producer's inventory. This operation is thread-safe.

        @param cart_id: The ID of the cart from which the product should be removed.
        @param product: The product to be removed.
        @return: void. (Implicitly returns True/False by logic, but no explicit return in original)
        """

        # Initialize product_owner to None.
        product_owner = None

        # Get the list of products in the current cart.
        cart_products = self.carts.get(cart_id)

        # Block Logic: Acquire lock to ensure atomic access to cart data during modification.
        self.cart_lock.acquire()

        # Block Logic: Iterate through products in the cart to find the target product.
        for curr_product in cart_products:
            # Check if the product matches and extract its original owner (producer).
            if product == curr_product[0]:
                product_owner = curr_product[1]
                break

        # If the product was found in the cart.
        if product_owner is not None:
            # Remove the product from the consumer's cart.
            self.carts[cart_id].remove([product, product_owner])
            # Return the product to the original producer's inventory.
            self.producers.get(product_owner).append(product)

        # Release the lock.
        self.cart_lock.release()

    def place_order(self, cart_id):
        """
        @brief Places an order for a specified consumer's cart.

        Retrieves the list of products from the cart to represent the order.
        Note: The original code does not clear the cart or move products
        to a "sold" state in the marketplace after placing an order.

        @param cart_id: The ID of the cart for which the order is being placed.
        @return: A list of products (dictionaries) that constitute the order.
        """

        # Initialize an empty list to store the ordered products.
        order = []

        # Get the list of products in the specified cart.
        cart_products = self.carts.get(cart_id)
        # Block Logic: Iterate through the products in the cart and add them to the order list.
        for product_owner_pair in cart_products:
            # Append only the product (first element of the pair) to the order.
            order.append(product_owner_pair[0])

        return order

import time
from threading import Thread


class Producer(Thread):
    """
    @brief Represents a producer in the marketplace simulation.

    Each Producer operates as an independent thread, continuously
    registering itself and publishing its predefined set of products
    to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.

        @param products: A list of products to be published. Each product
                         is a tuple: (product_id, quantity_to_publish,
                         time_to_wait_after_each_publish).
        @param marketplace: The shared Marketplace instance this producer
                            will interact with.
        @param republish_wait_time: The time in seconds to wait before
                                    attempting to republish the entire
                                    batch of products.
        @param kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """

        Thread.__init__(self, **kwargs)

        self.republish_wait_time = republish_wait_time
        self.products = products
        self.marketplace = marketplace

    def run(self):
        """
        @brief The main execution logic for the Producer thread.

        This method continuously registers the producer with the marketplace,
        then iterates through its product list, publishing each product
        a specified number of times with a delay between each publication.
        It then waits before restarting the entire publishing cycle.
        """
        # Block Logic: The producer's main loop for continuous operation.
        # Invariant: The producer attempts to publish its products indefinitely.
        while True:
            # Register the producer with the marketplace and get its unique ID.
            producer_id = self.marketplace.register_producer()

            # Block Logic: Iterate through each product definition to publish items.
            for product_data in self.products:
                # Initialize a counter for successfully published products of this type.
                count = 0

                # Extract product details from the tuple.
                product_id = product_data[0]
                qty = product_data[1]
                waiting_time = product_data[2]

                # Block Logic: Continuously attempt to publish the product until the target quantity is met.
                # Invariant: 'count' tracks the number of products successfully published.
                while count < qty:
                    # Attempt to publish the product.
                    must_wait = not (self.marketplace.publish(producer_id, product_id))

                    if must_wait:
                        # If publishing failed (e.g., queue full), wait before retrying.
                        time.sleep(self.republish_wait_time)
                    else:
                        # If successful, increment the count and wait for the specified production time.
                        count += 1
                        time.sleep(waiting_time)
