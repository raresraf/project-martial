

"""
@file consumer.py
@brief Implements a simulated e-commerce system with producers, a marketplace, and consumers, demonstrating concurrent operations.

This module integrates several components: `Consumer` threads for purchasing products from the marketplace,
the `Marketplace` itself for managing product inventory and carts, `Producer` threads for supplying products,
and `Product` dataclasses (Coffee, Tea) for defining product types.
It utilizes threading and synchronization primitives to simulate a concurrent environment.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    @brief Represents a consumer thread in the e-commerce simulation.
    Each consumer attempts to purchase products defined in its shopping carts
    from the marketplace, handling retries if products are not immediately available.
    """
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.
        @param carts A list of shopping cart definitions, where each cart is a list of operations (add/remove product).
        @param marketplace The `Marketplace` instance from which to buy products.
        @param retry_wait_time The time in seconds to wait before retrying an add-to-cart operation if it fails.
        @param kwargs Arbitrary keyword arguments, including a "name" for the consumer.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.id = kwargs["name"] # @brief Unique identifier for the consumer, typically its thread name.

    def run(self):
        """
        @brief The main execution loop for the consumer thread.
        Functional Utility: Iterates through its assigned carts, attempts to add or remove
                            products from the marketplace based on the cart's operations,
                            handling retries for unavailable products. Finally, it places
                            the order and prints the purchased items.
        """
        for cart in self.carts:
            # Block Logic: Acquire marketplace lock to safely create a new cart.
            # Pre-condition: The marketplace's lock is acquired to prevent race conditions
            #                during cart ID generation and initialization.
            # Invariant: A unique `cart_id` is obtained for the current shopping cart.
            self.marketplace.lock.acquire()
            cart_id = self.marketplace.new_cart()
            self.marketplace.lock.release() // Functional Utility: Release the lock after cart creation.


            for operation in cart:
                type = operation['type']
                product = operation['product']
                quantity = operation['quantity']
                for i in range(quantity):
                    self.marketplace.lock.acquire() # Block Logic: Acquire marketplace lock for cart modification.
                    if type == "add":
                        # Block Logic: Continuously try to add product to cart until successful.
                        # This loop handles cases where the product might not be immediately
                        # available in the marketplace, retrying after a `retry_wait_time`.
                        # Releases lock, sleeps, and re-acquires lock if product is unavailable.
                        # Invariant: The product is successfully added to the cart before exiting the loop.
                        while not self.marketplace.add_to_cart(cart_id, product):
                            self.marketplace.lock.release() // Functional Utility: Temporarily release lock to allow other threads to operate.
                            sleep(self.retry_wait_time) // Functional Utility: Wait before retrying.
                            self.marketplace.lock.acquire() // Functional Utility: Re-acquire lock before next attempt.
                    else:
                        # Block Logic: Remove product from cart.
                        # Pre-condition: `type` is "remove".
                        # Invariant: The specified `product` is removed from the `cart_id`.
                        self.marketplace.remove_from_cart(cart_id, product)
                    self.marketplace.lock.release() // Functional Utility: Release marketplace lock after cart modification.

            # Block Logic: Place the final order for the current cart.
            # Invariant: The `place_order` method processes the cart and returns a list of purchased products.
            products = self.marketplace.place_order(cart_id)

            # Block Logic: Print purchased products.
            # Functional Utility: Outputs a message indicating which products were bought by this consumer.
            for product in products:
                print(self.id + " bought " + str(product))

import logging
import time
import unittest
from threading import Lock

from tema.product import Coffee, Product, Tea


class Marketplace:
    """
    @brief Manages product inventory, producer registration, shopping carts, and order placement.
    Functional Utility: Acts as the central hub for the e-commerce simulation,
                        coordinating interactions between producers and consumers.
                        It maintains the state of available products, active shopping carts,
                        and ensures thread-safe operations through a locking mechanism.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.
        @param queue_size_per_producer The maximum number of products a single producer can have in the marketplace at any time.
        """
        self.max_queue_size = queue_size_per_producer # @brief Maximum number of products a single producer can have in the marketplace.
        self.available_products = {} # @brief Dictionary storing available products per producer: {producer_id: {product: quantity}}.
        self.no_available_products = {} # @brief Dictionary storing the total count of available products for each producer.

        self.carts_in_use = {} # @brief Dictionary storing products in active carts: {cart_id: [(product, producer_id), ...]}.
        self.last_cart_id = -1 # @brief Tracks the last assigned cart ID to ensure unique IDs.

        self.last_producer_id = 0 # @brief Tracks the last assigned producer ID to ensure unique IDs.

        self.lock = Lock() # @brief A threading.Lock to ensure thread-safe access to marketplace data structures.
        logging.Formatter.converter = time.gmtime
        logging.basicConfig(filename="marketplace.log",
                            filemode="a",
                            format='%(asctime)s,%(msecs)d %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger() # @brief Logger instance for recording marketplace events.

    def add_available_product(self, producer_id, product):
        """
        @brief Adds a product to the list of available products for a specific producer.
        Functional Utility: This method updates the marketplace's inventory by increasing
                            the count of a given `product` for a `producer_id`. It
                            ensures that the product is tracked correctly, either by
                            initializing its count or incrementing an existing one.
                            Logging is used to trace the operation.
        @param producer_id The ID of the producer supplying the product.
        @param product The `Product` instance to add.
        """
        self.logger.info(
            'Entered function "add_available_product" with parameter: ' + str(producer_id) + ' and ' + str(product))
        if product not in self.available_products[producer_id]:
            self.available_products[producer_id][product] = 1
        else:
            self.available_products[producer_id][product] += 1

        self.no_available_products[producer_id] += 1
        self.logger.info('Exit function "add_available_product"')

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.
        Functional Utility: Assigns a unique ID to a new producer, initializes its
                            product queues (available products and total count) in the
                            marketplace's internal state, and logs the registration event.
        @return The newly assigned unique `producer_id`.
        """
        self.logger.info('Entered function "register_producer"')

        self.last_producer_id += 1

        self.no_available_products[self.last_producer_id] = 0
        self.available_products[self.last_producer_id] = {}

        self.logger.info('Exit function "register_producer" with value: ' + str(self.last_producer_id))

        return self.last_producer_id

    def publish(self, producer_id, product):
        """
        @brief Publishes a product from a producer to the marketplace.
        Functional Utility: Attempts to add a `product` from a `producer_id` to the
                            marketplace's available inventory. This operation is
                            constrained by `max_queue_size_per_producer`. If the
                            producer's queue is full, the product is not published.
                            Logging is used to trace the operation.
        @param producer_id The ID of the producer publishing the product.
        @param product The `Product` instance to publish.
        @return `True` if the product was successfully published, `False` otherwise.
        """
        self.logger.info('Entered function "publish" with parameters ' + str(producer_id) + ' and  ' + str(product))

        if self.no_available_products[producer_id] >= self.max_queue_size:
            self.logger.info('Exit function "publish" with value: False')
            return False

        self.add_available_product(producer_id, product)

        self.logger.info('Exit function "publish" with value: True')

        return True

    def new_cart(self):
        """
        @brief Creates a new, empty shopping cart in the marketplace.
        Functional Utility: Generates a unique `cart_id`, associates it with an
                            empty list of products in `carts_in_use`, and logs
                            the cart creation event. This ID is then used by
                            consumers to add or remove products.
        @return The newly assigned unique `cart_id`.
        """
        self.logger.info('Entered function "new_cart"')

        self.last_cart_id += 1
        self.carts_in_use[self.last_cart_id] = []

        self.logger.info('Exit function "new_cart" with value: ' + str(self.last_cart_id))

        return self.last_cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a specified product to a given cart.
        Functional Utility: Searches through available products from all producers. If the
                            `product` is found, its quantity is decremented in the
                            marketplace's inventory, and the product (along with its
                            producer ID) is added to the `cart_id`. This method
                            ensures that products are acquired from the marketplace's
                            stock. Logging is used for tracing.
        @param cart_id The ID of the cart to which the product should be added.
        @param product The `Product` instance to add to the cart.
        @return `True` if the product was successfully added, `False` if the product
                was not available from any producer.
        """
        self.logger.info('Entered function "add_to_cart" with parameters: ' + str(cart_id) + ' and  ' + str(product))

        # Block Logic: Iterate through producers to find an available product.
        # Invariant: If a matching product is found, its quantity is adjusted,
        #            and it's added to the cart before returning True.
        for producer in self.available_products.keys():
            if product in self.available_products[producer]:
                number_of_products = self.available_products[producer][product]
                number_of_products -= 1
                if number_of_products <= 0:
                    del self.available_products[producer][product] # Functional Utility: Remove product entry if quantity drops to zero or less.
                else:
                    self.available_products[producer][product] = number_of_products # Functional Utility: Decrement product quantity.

                self.carts_in_use[cart_id].append((product, producer)) # Functional Utility: Add product and producer ID to the cart.

                self.no_available_products[producer] -= 1 # Functional Utility: Decrement the total count of available products for the producer.

                self.logger.info('Exit function "add_to_cart" with value: True')

                return True

        self.logger.info('Exit function "add_to_cart" with value: False')

        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a specified product from a given cart and returns it to the marketplace.
        Functional Utility: Iterates through the items in the `cart_id`. If the
                            `product` is found, it is removed from the cart, and
                            its availability is restored to the original producer
                            in the marketplace's inventory.
        @param cart_id The ID of the cart from which the product should be removed.
        @param product The `Product` instance to remove from the cart.
        """
        self.logger.info(
            'Entered function "remove_from_cart" with parameters: ' + str(cart_id) + ' and ' + str(product))

        if cart_id not in self.carts_in_use: # Block Logic: Check if the cart ID is valid.
            return

        # Block Logic: Iterate through the products in the specified cart.
        # Invariant: If the `product` is found, it's returned to the producer
        #            and removed from the cart.
        for product_producer in self.carts_in_use[cart_id]:
            current_product = product_producer[0]
            current_producer = product_producer[1]
            if current_product == product: # Inline: Check if the current product in the cart matches the product to be removed.
                self.add_available_product(current_producer, product) # Functional Utility: Return the product to the producer's available stock.
                self.carts_in_use[cart_id].remove(product_producer) # Functional Utility: Remove the product from the cart.
                break

        self.logger.info('Exit function "remove_from_cart"')

    def place_order(self, cart_id):
        """
        @brief Finalizes an order for a given cart.
        Functional Utility: Retrieves the list of products from the specified `cart_id`,
                            logs the order placement event, and returns the list of
                            products that were in the cart. The cart is not removed
                            from `carts_in_use` by this method.
        @param cart_id The ID of the cart for which to place the order.
        @return A list of `Product` instances that were in the ordered cart, or
                an empty list if the `cart_id` is invalid.
        """
        self.logger.info('Entered function "place_order" + with parameter: ' + str(cart_id))

        if cart_id not in self.carts_in_use: # Block Logic: Validate cart_id.
            self.logger.info('Exit function "place_order" with value: []')
            return []

        self.logger.info('Exit function "place_order" with value: ' + str(self.carts_in_use[cart_id]))

        produce_list = [] # @brief List to store the products from the cart.
        # Block Logic: Extract products from the cart for the final order list.
        for product_producer in self.carts_in_use[cart_id]:
            produce_list.append(product_producer[0])

        return produce_list


class TestMarketplace(unittest.TestCase):
    """
    @brief Unit tests for the Marketplace class.
    Functional Utility: This class contains a suite of tests to verify the
                        correct functionality of the `Marketplace` class,
                        including producer registration, product publishing,
                        cart management, and order placement.
    """
    def setUp(self):
        """
        @brief Sets up the test environment before each test method.
        Functional Utility: Initializes a fresh `Marketplace` instance and
                            defines various `Product` objects (Coffee, Tea)
                            for use in the tests.
        """
        self.marketplace = Marketplace(1)
        self.coffee1 = Coffee(name="Ethiopia", price=10, acidity="6", roast_level='MEDIUM')
        self.coffee2 = Coffee(name="China", price=10, acidity="1", roast_level='HIGH')
        self.tea = Tea(name="Ethiopia", price=10, type="Black")

    def test_register_producer(self):
        """
        @brief Tests the registration of a new producer.
        Functional Utility: Verifies that `register_producer` correctly assigns
                            a unique ID to the producer.
        """
        id = self.marketplace.register_producer()
        self.assertEqual(1, id)

    def test_publish(self):
        """
        @brief Tests publishing products by a producer.
        Functional Utility: Verifies that a producer can publish products
                            and that these products are correctly recorded
                            in the marketplace's available inventory.
        """
        id = self.marketplace.register_producer()
        self.marketplace.publish(id, self.coffee1)
        self.marketplace.publish(id, self.coffee2)
        self.assertEqual(1, self.marketplace.available_products[id][self.coffee1])
        self.assertEqual(False, self.coffee2 in self.marketplace.available_products[id])

    def test_new_cart(self):
        """
        @brief Tests the creation of a new cart.
        Functional Utility: Verifies that `new_cart` assigns a unique ID
                            to the newly created cart.
        """
        id = self.marketplace.new_cart()
        self.assertEqual(id, 0)

    def test_add_to_cart(self):
        """
        @brief Tests adding products to a cart.
        Functional Utility: Verifies that products can be successfully added
                            to a cart after being published by a producer.
        """
        id_cart = self.marketplace.new_cart()


        id_producer = self.marketplace.register_producer()
        self.marketplace.publish(id_producer, self.tea)
        self.marketplace.add_to_cart(id_cart, self.tea)
        self.assertEqual(True, (self.tea, id_producer) in self.marketplace.carts_in_use[id_cart])

    def test_remove_from_cart(self):
        """
        @brief Tests removing products from a cart.
        Functional Utility: Verifies that products can be removed from a cart
                            and that their availability is correctly restored
                            in the marketplace.
        """
        id_cart = self.marketplace.new_cart()
        id_producer = self.marketplace.register_producer()


        self.marketplace.publish(id_producer, self.tea)
        self.marketplace.add_to_cart(id_cart, self.tea)
        self.marketplace.remove_from_cart(id_cart, self.tea)
        self.assertEqual(False, (self.tea, id_producer) in self.marketplace.carts_in_use[id_cart])

    def test_place_order(self):
        """
        @brief Tests placing an order from a cart.
        Functional Utility: Verifies that `place_order` correctly returns
                            the list of products in the specified cart.
        """

        id_cart = self.marketplace.new_cart()

        id_producer1 = self.marketplace.register_producer()
        id_producer2 = self.marketplace.register_producer()

        self.marketplace.publish(id_producer1, self.coffee1)
        self.marketplace.publish(id_producer2, self.coffee2)



        self.marketplace.add_to_cart(id_cart, self.coffee1)
        self.marketplace.add_to_cart(id_cart, self.coffee2)

        order_list = self.marketplace.place_order(id_cart)

        self.assertEqual([self.coffee1, self.coffee2], order_list)

    def test_add_available_product(self):
        """
        @brief Tests internal method for adding available products.
        Functional Utility: Verifies that `add_available_product` correctly
                            updates the internal count of available products
                            for a producer.
        """
        id = self.marketplace.register_producer()
        self.marketplace.add_available_product(id, self.coffee1)
        self.assertEqual(1, self.marketplace.available_products[id][self.coffee1])


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    @brief Represents a producer thread in the e-commerce simulation.
    Functional Utility: Each producer is responsible for continuously supplying a defined
                        set of `products` to the marketplace. It simulates production
                        time and handles re-publishing products if the marketplace queue
                        is temporarily full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.
        @param products A list of tuples, where each tuple contains (Product, quantity, time_to_produce).
        @param marketplace The `Marketplace` instance to which products will be published.
        @param republish_wait_time The time in seconds to wait before retrying to publish if the marketplace queue is full.
        @param kwargs Arbitrary keyword arguments, including a "name" for the producer.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.id = -1 # @brief Unique identifier for the producer, assigned by the marketplace.

    def run(self):
        """
        @brief The main execution loop for the producer thread.
        Functional Utility: Registers the producer with the marketplace, then enters
                            an infinite loop to produce and publish products. It
                            simulates production time (`time_to_produce`) and
                            includes retry logic if the marketplace's queue for
                            this producer is full.
        """
        # Block Logic: Register the producer with the marketplace.
        # Pre-condition: Marketplace lock is acquired to ensure thread-safe registration.
        # Invariant: The producer receives a unique `id` from the marketplace.
        self.marketplace.lock.acquire()
        self.id = self.marketplace.register_producer()
        self.marketplace.lock.release()

        # Block Logic: Infinite loop for continuous product production and publishing.
        # Invariant: Products are continually produced and attempts are made to publish them.
        while True:
            for product in self.products:
                real_product = product[0]
                quantity = product[1]
                time_to_produce = product[2]
                for i in range(quantity):
                    sleep(time_to_produce) # Functional Utility: Simulate time taken to produce the product.
                    self.marketplace.lock.acquire() # Block Logic: Acquire marketplace lock before publishing.
                    # Block Logic: Continuously try to publish product until successful.
                    # Releases lock, sleeps, and re-acquires lock if marketplace queue is full.
                    while not self.marketplace.publish(self.id, real_product):
                        self.marketplace.lock.release() # Functional Utility: Temporarily release lock to allow other threads to operate.
                        sleep(self.republish_wait_time) # Functional Utility: Wait before retrying publish.
                        self.marketplace.lock.acquire() # Functional Utility: Re-acquire lock before next attempt.
                    self.marketplace.lock.release() # Functional Utility: Release marketplace lock after publishing attempt.


from dataclasses import dataclass


class Product:
    """
    @dataclass
    @brief Base class for all products in the e-commerce simulation.
    Functional Utility: Defines common attributes for any product, such as
                        its `name` and `price`. It serves as a foundation
                        for more specific product types like `Tea` and `Coffee`.
    @attribute name: The name of the product (string).
    @attribute price: The price of the product (integer).
    """
    name: str
    price: int


class Tea(Product):
    """
    @dataclass
    @brief Represents a specific type of product: Tea.
    Functional Utility: Inherits from `Product` and adds a `type` attribute
                        to specify the kind of tea (e.g., "Black", "Green", "Herbal").
    @attribute type: The type of tea (string).
    """
    type: str


class Coffee(Product):
    """
    @dataclass
    @brief Represents a specific type of product: Coffee.
    Functional Utility: Inherits from `Product` and adds `acidity` and
                        `roast_level` attributes to describe specific characteristics
                        of the coffee.
    @attribute acidity: The acidity level of the coffee (string).
    @attribute roast_level: The roast level of the coffee (string).
    """
    acidity: str
    roast_level: str
