
"""
@7b006929-595e-4e40-8792-2cd46d94c01f/consumer.py
@brief Implements a multithreaded marketplace simulation with Consumers and Producers.

This module defines the core logic for a marketplace where multiple consumers
and producers interact. Consumers attempt to add and remove products from
a shared marketplace, and producers continuously publish products.
Synchronization primitives (e.g., Lock) are used to manage concurrent access
to shared resources. Logging is integrated to track marketplace activities.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    @brief Represents a consumer in the marketplace.

    Each consumer operates as a separate thread, attempting to add or remove
    products from the marketplace's carts and ultimately placing an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.

        @param carts: A list of carts, where each cart contains a list of product data.
                      Product data is a dictionary with "quantity", "type" ("add" or "remove"), and "product".
        @param marketplace: The shared Marketplace instance to interact with.
        @param retry_wait_time: The time in seconds to wait before retrying an operation
                                if it fails (e.g., product not available).
        @param kwargs: Additional keyword arguments for the Thread constructor.
        """
        super().__init__(**kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # Registers a new cart with the marketplace for this consumer.
        self.cart_id = self.marketplace.new_cart()

    def run(self):
        """
        @brief The main execution logic for the Consumer thread.

        It iterates through its assigned carts and products, attempting to add or
        remove them from the marketplace. If an operation fails, it retries after
        a specified wait time. Finally, it places the order for its cart.
        """
        # Block Logic: Iterates through each cart assigned to this consumer.
        # Invariant: Each cart contains a sequence of product operations (add/remove).
        for cart in self.carts:
            # Block Logic: Processes each product operation within the current cart.
            # Invariant: 'data' specifies the product, quantity, and operation type.
            for data in cart:
                # Block Logic: Executes the specified product operation for the given quantity.
                # Invariant: The loop continues until the required quantity is processed.
                for i in range(data["quantity"]):
                    ret = False
                    # Block Logic: Retries the add/remove operation until successful.
                    # Pre-condition: 'ret' is False, indicating a failed or untried operation.
                    # Invariant: 'ret' becomes True upon successful operation.
                    while not ret:
                        if data["type"] == "add":
                            # Attempt to add the product to the cart.
                            ret = self.marketplace.add_to_cart(self.cart_id, data["product"])
                        else:
                            # Attempt to remove the product from the cart.
                            ret = self.marketplace.remove_from_cart(self.cart_id, data["product"])
                        # If the operation failed, wait before retrying.
                        if not ret:
                            time.sleep(self.retry_wait_time)
        # Once all product operations are attempted, place the order for this cart.
        self.marketplace.place_order(self.cart_id)

import time
from threading import Lock
import logging
import logging.handlers


class Marketplace:
    """
    @brief Manages products, carts, producers, and consumers in a simulated e-commerce environment.

    This class handles the core logic of product publication, adding/removing
    items from carts, and processing orders, ensuring thread-safe operations
    using locks and logging all significant events.
    """
    
    # Logger setup for recording marketplace activities.
    formatter = logging.Formatter('(%(asctime)s %(threadName)-9s) %(funcName)s %(message)s')
    formatter.converter = time.gmtime
    handler = logging.handlers.RotatingFileHandler('marketplace.log')
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # MARK (Market): A global list representing available products in the marketplace.
    MARK = []

    # GET_PROD: Maps products to the producer_id that published them.
    GET_PROD = {}

    # PROD: Stores the current queue size per producer (number of products a producer can still publish).
    PROD = {}

    # CONS: Stores consumer carts, keyed by cart_id. Each cart contains products added by the consumer.
    CONS = {}

    # Global lock to ensure thread-safe access to shared marketplace data structures.
    lock = Lock()

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.

        @param queue_size_per_producer: The maximum number of products each producer
                                        can have in the marketplace's internal queue.
        """
        self.logger.info(f" <- queue_zie_per_producer = {queue_size_per_producer}")
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.

        Assigns a unique ID to the producer and initializes its allowed
        publication queue size.

        @return: The unique integer ID assigned to the new producer.
        """
        # Block Logic: Acquires a lock to ensure atomic registration of a new producer.
        self.lock.acquire()
        producer_id = len(self.PROD)

        # Initialize the producer's available queue size.
        self.PROD[producer_id] = self.queue_size_per_producer
        self.lock.release()
        self.logger.info(f" -> producer_id = {producer_id}")
        return producer_id

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to publish a product to the marketplace.

        The product is added to the market's inventory if the producer's
        queue limit has not been reached.

        @param producer_id: The ID of the producer publishing the product.
        @param product: The product (e.g., product name or object) to be published.
        @return: True if the product was successfully published, False otherwise.
        """
        self.logger.info(f" <- producer_id = {producer_id}, product = {product}")
        # Block Logic: Acquires a lock to ensure atomic publication.
        # Invariant: Checks if the producer has space in its allowed queue.
        self.lock.acquire()
        if self.PROD[producer_id] > 0:
            self.PROD[producer_id] -= 1
            self.MARK.append(product) # Add product to global market list.
            self.GET_PROD[product] = producer_id # Map product to its producer.
            self.lock.release()
            self.logger.info(f" -> True")
            return True
        self.lock.release()
        self.logger.info(f" -> False")
        return False

    def new_cart(self):
        """
        @brief Creates a new shopping cart for a consumer.

        Assigns a unique cart ID and initializes an empty cart for it.

        @return: The unique integer ID assigned to the new cart.
        """
        # Block Logic: Acquires a lock to ensure atomic cart creation.
        self.lock.acquire()
        cart_id = len(self.CONS)
        self.CONS[cart_id] = {} # Initialize an empty dictionary for the new cart.

        self.lock.release()
        self.logger.info(f" -> cart_id = {cart_id}")
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a specified consumer's cart.

        If the product is available in the marketplace, it's moved from
        the market's inventory to the consumer's cart. The producer's
        queue size is also adjusted.

        @param cart_id: The ID of the cart to which the product should be added.
        @param product: The product to be added.
        @return: True if the product was successfully added, False otherwise.
        """
        self.logger.info(f" <- cart_id = {cart_id}, product = {product}")
        # Block Logic: Acquires a lock to ensure atomic cart modification.
        self.lock.acquire()
        try:
            self.MARK.remove(product) # Attempt to remove product from global market.
        except ValueError:
            # If product is not in MARK, it means it's not available.
            self.lock.release()
            self.logger.info(f" -> False")
            return False
        
        # Get the producer who originally published this product.
        producer_id = self.GET_PROD[product]
        try:
            # Add the product to the specific producer's list within the cart.
            self.CONS[cart_id][producer_id].append(product)
        except KeyError:
            # If this producer is not yet in the cart, create its entry.
            self.CONS[cart_id][producer_id] = []
            self.CONS[cart_id][producer_id].append(product)
        # Increment the producer's available queue size as the product is now in a cart.
        self.PROD[producer_id] += 1

        self.lock.release()
        self.logger.info(f" -> True")
        return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specified consumer's cart.

        If the product is found in the cart, it's moved back to the market's
        inventory. The producer's queue size is also adjusted.

        @param cart_id: The ID of the cart from which the product should be removed.
        @param product: The product to be removed.
        @return: True if the product was successfully removed, False otherwise.
        """
        self.logger.info(f" <- cart_id = {cart_id}, product = {product}")
        # Block Logic: Acquires a lock to ensure atomic cart modification.
        self.lock.acquire()
        # Block Logic: Iterates through the producers associated with the cart to find the product.
        # Invariant: Checks if the product exists within any producer's list in the consumer's cart.
        for entry in self.CONS[cart_id]: # 'entry' here refers to producer_id
            # Block Logic: Iterates through products from a specific producer in the cart.
            for search_product in self.CONS[cart_id][entry]:
                if product == search_product:
                    self.CONS[cart_id][entry].remove(search_product) # Remove product from cart.
                    self.MARK.append(product) # Return product to global market.
                    self.PROD[entry] -= 1 # Decrement producer's available queue size.
                    self.lock.release()
                    self.logger.info(f" -> True")

                    return True
        self.lock.release()
        self.logger.info(f" -> False")
        return False

    def place_order(self, cart_id):
        """
        @brief Places an order for a specified consumer's cart.

        This simulates the finalization of a purchase, printing out the items
        bought by the consumer.

        @param cart_id: The ID of the cart for which the order is being placed.
        """
        self.logger.info(f" <- cart_id = {cart_id}")
        # Block Logic: Iterates through all products across all producers in the given cart.
        # Invariant: Each product in the cart is announced as "bought".
        for prod_list in self.CONS[cart_id].values():
            for prod in prod_list:
                # Acquire and release lock to ensure synchronized printing.
                self.lock.acquire()
                print(f'cons{cart_id + 1} bought {prod}')
                self.lock.release()

import time
from threading import Thread


class Producer(Thread):
    """
    @brief Represents a producer in the marketplace.

    Each producer operates as a separate thread, continuously publishing
    products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.

        @param products: A list of products, where each product is a tuple
                         (product_name, quantity, time_to_produce).
        @param marketplace: The shared Marketplace instance to interact with.
        @param republish_wait_time: The time in seconds to wait before attempting
                                    to republish a batch of products.
        @param kwargs: Additional keyword arguments for the Thread constructor.
        """
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Registers this producer with the marketplace.
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        @brief The main execution logic for the Producer thread.

        It continuously publishes its defined products to the marketplace.
        Products are published one by one, with a delay after each, and
        the entire batch is republished after a `republish_wait_time`.
        """
        # Block Logic: The producer's main loop for continuous operation.
        # Invariant: The producer attempts to publish its products indefinitely.
        while True:
            # Block Logic: Iterates through each product definition to publish items.
            # Invariant: 'product' contains the name, quantity, and production time.
            for product in self.products:
                # Block Logic: Publishes the specified quantity of the current product.
                # Invariant: Each individual product item is published with retries.
                for i in range(product[1]):
                    ret = False
                    # Block Logic: Retries publishing the product until successful.
                    # Pre-condition: 'ret' is False, indicating a failed or untried publication.
                    # Invariant: 'ret' becomes True upon successful publication.
                    while not ret:
                        ret = self.marketplace.publish(self.producer_id, product[0])
                        # Simulate time taken to produce each item.
                        time.sleep(product[2])
            # Wait before starting the next cycle of republishing all products.
            time.sleep(self.republish_wait_time)
