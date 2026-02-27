import time
from threading import Thread


class Consumer(Thread):
    """
    Represents a consumer thread in a marketplace simulation.

    This thread simulates a consumer that interacts with a marketplace by adding
    products to a cart, removing them, and eventually placing an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of cart operations to be performed. Each cart
                          is a list of products with operations ('add' or 'remove').
            marketplace (Marketplace): The central marketplace object.
            retry_wait_time (float): The time in seconds to wait before retrying
                                     to add a product if the marketplace is full.
            **kwargs: Additional arguments for the Thread constructor.
        """

        Thread.__init__(self, **kwargs)

        self.retry_wait_time = retry_wait_time
        self.carts = carts
        self.marketplace = marketplace

    def add(self, cart_id, product_id, qty):
        """
        Adds a specified quantity of a product to the cart.

        If the marketplace cannot add the product (e.g., it's out of stock),
        the consumer waits for `retry_wait_time` and tries again.

        Args:
            cart_id (int): The ID of the cart to add to.
            product_id (any): The identifier of the product to add.
            qty (int): The number of items of the product to add.
        """
        count = 0
        while count < qty:
            # Attempt to add the product to the cart via the marketplace.
            must_wait = not (self.marketplace.add_to_cart(cart_id, product_id))
            if must_wait:
                # If adding fails, wait before retrying.
                time.sleep(self.retry_wait_time)
            else:
                # Increment count on successful addition.
                count += 1

    def rm(self, cart_id, product_id, qty):
        """
        Removes a specified quantity of a product from the cart.

        Args:
            cart_id (int): The ID of the cart to remove from.
            product_id (any): The identifier of the product to remove.
            qty (int): The number of items to remove.
        """
        count = 0
        while count < qty:
            # Remove the product from the cart.
            self.marketplace.remove_from_cart(cart_id, product_id)
            count += 1

    def run(self):
        """
        The main execution loop for the consumer thread.

        It gets a new cart, processes all assigned cart operations (add/remove),
        and finally places the order.
        """
        # Obtain a new cart from the marketplace.
        cart_id = self.marketplace.new_cart()

        # Process the list of operations for this consumer.
        for cart in self.carts:
            for product in cart:
                # Extract product details and operation type.
                product_id = product.get("product")
                qty = product.get("quantity")
                op = product.get("type")

                # Perform the specified operation.
                if op == "add":
                    self.add(cart_id, product_id, qty)
                elif op == "remove":
                    self.rm(cart_id, product_id, qty)

        # Finalize the transaction by placing the order.
        order = self.marketplace.place_order(cart_id)
        for product in order:
            # Print the items bought by this consumer.
            print(self.name, "bought", product)


from threading import Lock
import unittest


class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace class.

    Note: This test class is defined before the Marketplace class it tests,
    which will cause a NameError when the script is run as is.
    """
    def setUp(self):
        """
        Sets up the test environment before each test case.
        """
        # A NameError will occur here because Marketplace is not yet defined.
        self.marketplace = Marketplace(10)

        # Register a producer for testing purposes.
        self.producer_id = self.marketplace.register_producer()

        # Define a sample product.
        self.product = {"product_type": "Coffee"}
        self.product["name"] = "Indonezia"
        self.product["acidity"] = 5.05
        self.product["roast_level"] = "MEDIUM"
        self.product["price"] = 1

        # Create a new cart for testing.
        self.cart_id = self.marketplace.new_cart()

    def tearDown(self):
        """
        Cleans up the test environment after each test case.
        """
        self.product = None
        self.producer_id = -1
        self.cart_id = -1
        self.marketplace.producers = {}
        self.marketplace.carts = {}
        self.marketplace = None

    def test_register_producer(self):
        """
        Tests if a producer is registered correctly.
        """
        self.assertGreater(self.producer_id, 0)
        self.assertListEqual(self.marketplace.producers.get(self.producer_id), [])

    def test_publish(self):
        """
        Tests if a product can be published successfully.
        """
        ret = self.marketplace.publish(self.producer_id, self.product)
        self.assertTrue(ret)

    def test_new_cart(self):
        """
        Tests if a new cart is created successfully.
        """
        self.assertGreater(self.cart_id, 0)
        self.assertListEqual(self.marketplace.carts.get(self.cart_id), [])

    def test_add_to_cart(self):
        """
        Tests adding a product to the cart.
        """
        self.marketplace.publish(self.producer_id, self.product)
        ret = self.marketplace.add_to_cart(self.cart_id, self.product)
        self.assertTrue(ret)

    def test_remove_from_cart(self):
        """
        Tests removing a product from the cart.
        """
        self.marketplace.publish(self.producer_id, self.product)
        self.marketplace.add_to_cart(self.cart_id, self.product)
        self.marketplace.remove_from_cart(self.cart_id, self.product)
        self.assertListEqual(self.marketplace.carts.get(self.cart_id), [])

    def test_place_order(self):
        """
        Tests placing an order.
        """
        order = self.marketplace.place_order(self.cart_id)
        self.assertIsNotNone(order)
        self.assertListEqual(order, self.marketplace.carts.get(self.cart_id))


class Marketplace:
    """
    A thread-safe marketplace for producers to sell products and consumers to buy them.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have listed at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_count = 0
        self.producers = {}
        self.publishing_lock = Lock()  # Lock for producer-related operations.
        self.consumer_count = 0
        self.carts = {}
        self.cart_lock = Lock()  # Lock for cart/consumer-related operations.

    def register_producer(self):
        """
        Registers a new producer, providing them with a unique ID.

        Returns:
            int: The new producer's ID.
        """
        self.producer_count += 1
        self.producers[self.producer_count] = []
        return self.producer_count

    def publish(self, producer_id, product):
        """
        Allows a producer to list a product for sale.

        This operation is thread-safe.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (any): The product to be published.

        Returns:
            bool: True if publishing was successful, False if the producer's queue is full.
        """
        successful_publish = False
        self.publishing_lock.acquire()
        products = self.producers.get(producer_id)
        queue_size = len(products)

        if queue_size < self.queue_size_per_producer:
            successful_publish = True
            self.producers.get(producer_id).append(product)

        self.publishing_lock.release()
        return successful_publish

    def new_cart(self):
        """
        Creates a new, empty cart for a consumer.

        Returns:
            int: The new cart's ID.
        """
        self.consumer_count += 1
        self.carts[self.consumer_count] = []
        return self.consumer_count

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's cart.

        This involves finding the product among all producers and moving it
        from the producer's inventory to the consumer's cart. This operation
        is thread-safe.

        Args:
            cart_id (int): The consumer's cart ID.
            product (any): The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        product_owner = None
        self.cart_lock.acquire()

        # Find which producer owns the product.
        for curr_producer in list(self.producers.keys()):
            for curr_product in self.producers.get(curr_producer):
                if product == curr_product:
                    product_owner = curr_producer
                    break
        
        if product_owner is not None:
            # If found, move product from producer to cart.
            self.producers.get(product_owner).remove(product)
            self.carts.get(cart_id).append([product, product_owner])

        self.cart_lock.release()
        return product_owner is not None

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a consumer's cart.

        This moves the product from the cart back to the original producer's
        inventory. This operation is thread-safe.

        Args:
            cart_id (int): The consumer's cart ID.
            product (any): The product to remove.
        """
        product_owner = None
        cart_products = self.carts.get(cart_id)
        self.cart_lock.acquire()

        # Find the product in the cart to identify its owner.
        for curr_product in cart_products:
            if product == curr_product[0]:
                product_owner = curr_product[1]
                break

        if product_owner is not None:
            # If found, move product from cart back to producer.
            self.carts[cart_id].remove([product, product_owner])
            self.producers.get(product_owner).append(product)

        self.cart_lock.release()

    def place_order(self, cart_id):
        """
        Finalizes the purchase, returning the list of products in the cart.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: A list of products that were in the cart.
        """
        order = []
        cart_products = self.carts.get(cart_id)
        for product in cart_products:
            order.append(product[0])
        return order

import time
from threading import Thread


class Producer(Thread):
    """
    Represents a producer thread in a marketplace simulation.

    This thread simulates a producer that publishes a list of products to the
    marketplace, respecting wait times and production quantities.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the Producer thread.

        Args:
            products (list): A list of products to produce. Each item is a tuple:
                             (product_id, quantity, waiting_time_after_publish).
            marketplace (Marketplace): The central marketplace object.
            republish_wait_time (float): Time to wait before retrying to publish
                                         if the marketplace queue is full.
            **kwargs: Additional arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.republish_wait_time = republish_wait_time
        self.products = products
        self.marketplace = marketplace

    def run(self):
        """
        The main execution loop for the producer thread.

        It registers with the marketplace and then enters an infinite loop
        to continuously produce and publish its assigned products.
        """
        while True:
            # Register as a new producer for each production cycle.
            producer_id = self.marketplace.register_producer()

            for product in self.products:
                count = 0
                product_id = product[0]
                qty = product[1]
                waiting_time = product[2]

                # Produce and publish the specified quantity of the product.
                while count < qty:
                    # Attempt to publish the product.
                    must_wait = not (self.marketplace.publish(producer_id, product_id))

                    if must_wait:
                        # If publishing fails (queue full), wait.
                        time.sleep(self.republish_wait_time)
                    else:
                        # On success, increment count and wait for the production time.
                        count += 1
                        time.sleep(waiting_time)
