

"""
@file consumer.py
@brief Implements a multi-threaded producer-consumer simulation with a marketplace.

This module models a system where `Producer` threads supply products to a `Marketplace`,
and `Consumer` threads interact with the marketplace to acquire and order these products.
It demonstrates fundamental concepts of concurrent programming, including thread
management and synchronization using semaphores to protect shared resources.
"""

from threading import Thread
import time
import random # Imported in Marketplace, but useful to mention here for module-level context.
from threading import Semaphore # Imported in Marketplace, but useful to mention here for module-level context.


class Consumer(Thread):
    """
    @class Consumer
    @brief Represents a customer thread that interacts with the Marketplace to buy products.

    Each consumer thread simulates a shopping experience, adding and removing products
    from a cart, and finally placing an order. It handles scenarios where products
    might not be immediately available in the marketplace by retrying after a delay.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.
        @param carts A list of predefined shopping cart actions (add/remove product, quantity).
        @param marketplace The shared Marketplace instance to interact with.
        @param retry_wait_time The time (in seconds) to wait before retrying an action if a product is unavailable.
        @param kwargs Additional keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        @brief The main execution logic for the Consumer thread.

        Block Logic:
        1. Obtains a new cart from the marketplace.
        2. Iterates through a predefined list of shopping actions.
        3. For "add" actions, it repeatedly tries to add the product to the cart,
           waiting if the product is not immediately available.
        4. For "remove" actions, it removes products from the cart.
        5. Finally, it places the order and prints the purchased items.
        """
        customer_id = self.marketplace.new_cart()
        # Block Logic: Processes each shopping cart (a list of actions) defined for this consumer.
        for cart in self.carts:
            # Block Logic: Processes each action (add or remove) within the current shopping cart.
            for action in cart:
                if action["type"] == "add":
                    i = 0
                    # Block Logic: Attempts to add the product to the cart multiple times.
                    # Invariant: `i` counts successfully added items.
                    while i < action["quantity"]:
                        # Pre-condition: Checks if the product can be added to the cart.
                        if self.marketplace.add_to_cart(customer_id, action["product"]):
                            i += 1 # Inline: Increments count on successful addition.
                        else:
                            # Block Logic: If the product is unavailable, waits and retries.
                            time.sleep(self.retry_wait_time)
                else: # Action type is "remove".
                    # Block Logic: Removes the specified quantity of the product from the cart.
                    for i in range(action["quantity"]):
                        self.marketplace.remove_from_cart(customer_id, action["product"])
        # Block Logic: Places the final order with the marketplace.
        order = self.marketplace.place_order(customer_id)
        # Block Logic: Prints the items bought by this consumer.
        for product in order:
            print(self.name, "bought", product)


class Marketplace:
    """
    @class Marketplace
    @brief Manages products, producers, customer carts, and order placement in a multi-threaded environment.

    This class acts as a central hub for all product and transaction logic. It ensures
    thread-safe operations using semaphores to protect shared data structures such as
    product inventories and customer carts.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.
        @param queue_size_per_producer The maximum number of products each producer can have in its inventory at any time.
        """
        self.queue_size_per_producer = queue_size_per_producer

        # Invariant: List of registered producer IDs.
        self.producers = []
        # Invariant: List of registered customer IDs (not directly used, but implies customer management).
        self.customers = []

        # Invariant: Dictionary mapping customer IDs to their shopping carts (list of (product, producer_id) tuples).
        self.carts = {}
        # Invariant: Dictionary mapping producer IDs to their list of products currently in stock.
        self.products = {}
        # Invariant: Semaphores used to protect critical sections related to various marketplace operations.
        # Each semaphore protects a specific shared resource or set of resources.
        self.sem = Semaphore(1) # Protects `self.products` during `publish`.
        self.sem1 = Semaphore(1) # Protects `self.products` and `self.carts` during `add_to_cart`.
        self.sem2 = Semaphore(1) # (Unused in provided code, but could be for future expansion).
        self.sem3 = Semaphore(1) # Protects `self.carts` and `self.products` during `remove_from_cart`.
        self.sem4 = Semaphore(1) # Protects `self.carts` during `place_order`.

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace and assigns a unique ID.
        @return The unique ID assigned to the new producer.
        """
        # Block Logic: Generates a random ID until a unique one is found.
        while True:
            rand = random.randint(0, 5000)
            # Pre-condition: Checks if the generated ID is already in use.
            if rand not in self.producers:
                self.producers.append(rand) # Inline: Adds the new producer ID.
                self.products[rand] = [] # Inline: Initializes an empty product list for the new producer.
                return rand

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to publish a product to its inventory in the marketplace.
        The product is added only if the producer's queue has not reached its maximum size.
        @param producer_id The ID of the producer publishing the product.
        @param product The product to be published.
        @return `True` if the product was successfully published, `False` otherwise.
        """
        self.sem.acquire() # Block Logic: Acquires semaphore to protect `self.products`.
        length = len(self.products[producer_id])
        # Pre-condition: Checks if the producer's product queue is full.
        if length < self.queue_size_per_producer:
            self.products[producer_id].append(product) # Inline: Adds product to producer's inventory.
            self.sem.release() # Inline: Releases semaphore.
            return True
        self.sem.release() # Inline: Releases semaphore.
        return False

    def new_cart(self):
        """
        @brief Creates a new shopping cart for a customer and assigns a unique cart ID.
        @return The unique ID assigned to the new cart.
        """
        # Block Logic: Generates a random cart ID until a unique one is found.
        while True:
            rand = random.randint(0, 5000)
            # Pre-condition: Ensures the generated ID is not already used by a producer (to prevent ID conflicts).
            if rand not in self.producers:
                self.carts[rand] = [] # Inline: Initializes an empty cart for the new customer ID.
                return rand

    def add_to_cart(self, cart_id, product):
        """
        @brief Attempts to add a specific product to a customer's cart.
        It searches through all producers' inventories for the product. If found,
        the product is moved from the producer's inventory to the customer's cart.
        @param cart_id The ID of the customer's cart.
        @param product The product to add.
        @return `True` if the product was successfully added, `False` otherwise (e.g., product not found).
        """
        self.sem1.acquire() # Block Logic: Acquires semaphore to protect `self.products` and `self.carts`.
        # Block Logic: Iterates through all registered producers to find the product.
        for producer in self.producers:
            products = self.products[producer]
            # Pre-condition: Checks if the product is in the current producer's inventory.
            if product in products:
                self.carts[cart_id].append((product, producer)) # Inline: Adds product to cart.
                self.products[producer].remove(product) # Inline: Removes product from producer's inventory.
                self.sem1.release() # Inline: Releases semaphore.
                return True
        self.sem1.release() # Inline: Releases semaphore.
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a customer's cart and returns it to the original producer's inventory.
        @param cart_id The ID of the customer's cart.
        @param product The product to remove.
        """
        self.sem3.acquire() # Block Logic: Acquires semaphore to protect `self.carts` and `self.products`.
        producer_id = -1 # Inline: Placeholder for the producer ID from which the product was originally taken.
        # Block Logic: Iterates through the items in the customer's cart.
        for aux in self.carts[cart_id]:
            # Pre-condition: Checks if the current item matches the product to be removed.
            if aux[0] == product:
                producer_id = aux[1] # Inline: Retrieves the original producer ID.
                self.carts[cart_id].remove((product, producer_id)) # Inline: Removes product from cart.
                self.products[producer_id].append(product) # Inline: Returns product to producer's inventory.
                self.sem3.release() # Inline: Releases semaphore.
                return
        self.sem3.release() # Inline: Releases semaphore.

    def place_order(self, cart_id):
        """
        @brief Finalizes a customer's cart into an order list.
        @param cart_id The ID of the customer's cart.
        @return A list of products in the placed order.
        """
        self.sem4.acquire() # Block Logic: Acquires semaphore to protect `self.carts`.
        order = []
        # Block Logic: Populates the order list from the customer's cart.
        for aux in self.carts[cart_id]:
            order.append(aux[0])
        self.sem4.release() # Inline: Releases semaphore.
        return order


class Producer(Thread):
    """
    @class Producer
    @brief Represents a thread that continuously registers with the Marketplace and publishes products.

    Each producer thread generates a stream of products and attempts to add them
    to its allocated inventory space within the Marketplace. If the inventory is full,
    the producer waits before retrying.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.
        @param products A list of product definitions, where each is `[product_name, quantity, time_to_sleep_after_publish]`.
        @param marketplace The shared Marketplace instance to interact with.
        @param republish_wait_time The time (in seconds) to wait if a product cannot be published due to full queue.
        @param kwargs Additional keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        @brief The main execution logic for the Producer thread.

        Block Logic:
        1. Registers itself with the marketplace to get a unique producer ID.
        2. Continuously attempts to publish its predefined products.
        3. If a product is successfully published, it waits for a specified time.
        4. If the product queue is full, it waits for a `republish_wait_time`
           before retrying to publish the same product.
        """
        # Block Logic: The producer continuously attempts to register itself.
        while True:
            producer_id = self.marketplace.register_producer()
            # Block Logic: Iterates through each product definition to publish products.
            for [prod, quantity, timee] in self.products:
                # Block Logic: Publishes the specified quantity of each product.
                for i in range(quantity):
                    # Pre-condition: Attempts to publish the product.
                    sleep_time = self.marketplace.publish(producer_id, prod)
                    if sleep_time:
                        time.sleep(timee) # Inline: Waits on successful publish.
                    else:
                        # Block Logic: If publish fails, waits and retries the same product.
                        time.sleep(self.republish_wait_time)
                        i = i - 1 # Inline: Decrements `i` to retry publishing the same product.
