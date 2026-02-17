"""
A multi-threaded producer-consumer simulation modeling an e-commerce marketplace.

This script defines three classes:
- Marketplace: A thread-safe central hub for producers to publish products and
  consumers to purchase them.
- Producer: A thread that generates and publishes products to the marketplace.
- Consumer: A thread that simulates a customer adding products to a cart and
  placing an order.
"""

import time
from threading import Thread, Lock


class Consumer(Thread):
    """
    Represents a consumer thread that simulates purchasing items from the marketplace.

    Each consumer processes a list of carts, with each cart containing a series of
    'add' or 'remove' operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        :param carts: A list of carts, where each cart is a list of operations.
        :param marketplace: The Marketplace instance to interact with.
        :param retry_wait_time: Time to wait before retrying to add a product.
        :param kwargs: Keyword arguments, including the consumer's 'name'.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        """
        The main execution loop for the consumer thread.

        Iterates through its assigned carts, creates a new cart in the marketplace for each,
        performs the specified add/remove operations, and then places the order.
        """
        # Pre-condition: Each consumer is assigned a list of carts to process.
        for cart in self.carts:
            # Invariant: A new cart is created for each list of operations.
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                # Logic: Differentiates between adding and removing a product from the cart.
                if operation["type"] == "add":
                    # Pre-condition: An 'add' operation requires a product and a quantity.
                    i = 0
                    while i < operation["quantity"]:
                        # Attempt to add the product to the cart.
                        verify = self.marketplace.add_to_cart(cart_id, operation["product"])
                        # Invariant: Retry with a delay if the product is not available.
                        while not verify:
                            time.sleep(self.wait_time)
                            verify = self.marketplace.add_to_cart(cart_id, operation["product"])
                        i += 1
                elif operation["type"] == "remove":
                    # Pre-condition: A 'remove' operation requires a product and a quantity.
                    i = 0
                    while i < operation["quantity"]:
                        self.marketplace.remove_from_cart(cart_id, operation["product"])
                        i += 1
            # Finalizes the transaction for the current cart.
            orders = self.marketplace.place_order(cart_id)
            for order in orders:
                print("%s bought %s" % (self.name, order[0]))


class Marketplace:
    """
    A thread-safe marketplace that facilitates the interaction between producers and consumers.

    This class manages product inventory from multiple producers and customer shopping carts,
    using locks to ensure data consistency in a multi-threaded environment.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        :param queue_size_per_producer: The maximum number of products each producer can have for sale at one time.
        """
        self.queue_size = queue_size_per_producer
        self.producer_id = 0
        self.cart_id = 0
        self.producers_buffers = {}
        self.carts_list = {}
        # Locks to ensure thread-safe operations on shared data structures.
        self.lock_buffers = Lock()
        self.lock_carts = Lock()

    def register_producer(self):
        """
        Registers a new producer, providing them with a unique ID and an inventory buffer.

        :return: The new producer's unique ID.
        """
        with self.lock_buffers:
            self.producer_id += 1
            self.producers_buffers[self.producer_id] = []
            new_id = self.producer_id
        return new_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        :param producer_id: The ID of the producer publishing the product.
        :param product: The product to be published.
        :return: True if the product was published successfully, False if the producer's buffer is full.
        """
        self.lock_buffers.acquire()
        # Pre-condition: The producer's buffer must not be full.
        if len(self.producers_buffers[producer_id]) < self.queue_size:
            self.producers_buffers[producer_id].append(product)
            self.lock_buffers.release()
            return True
        self.lock_buffers.release()
        return False

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.

        :return: The unique ID of the newly created cart.
        """
        with self.lock_carts:
            self.cart_id += 1
            self.carts_list[self.cart_id] = []
            new_cart = self.cart_id
        return new_cart

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified shopping cart by taking it from a producer's inventory.

        :param cart_id: The ID of the cart to add the product to.
        :param product: The product to be added.
        :return: True if the product was found and added, False otherwise.
        """
        self.lock_buffers.acquire()
        # Invariant: Scans all producer buffers to find the requested product.
        for producer in self.producers_buffers:
            for prod in self.producers_buffers[producer]:
                if prod == product:
                    self.producers_buffers[producer].remove(prod)
                    self.carts_list[cart_id].append((prod, producer))
                    self.lock_buffers.release()
                    return True
        self.lock_buffers.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart and returns it to the original producer's inventory.

        :param cart_id: The ID of the cart to remove the product from.
        :param product: The product to be removed.
        """
        # Invariant: Scans the cart to find the specified product to remove.
        for (prod, producer) in self.carts_list[cart_id]:
            if prod == product:
                with self.lock_carts:
                    self.carts_list[cart_id].remove((prod, producer))
                # Returns the product to the producer's inventory.
                self.producers_buffers[producer].append(prod)
                break

    def place_order(self, cart_id):
        """
        Finalizes an order by returning the list of items in the cart.

        :param cart_id: The ID of the cart to be ordered.
        :return: A list of products in the cart.
        """
        return self.carts_list[cart_id]


class Producer(Thread):
    """
    Represents a producer thread that generates and publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        :param products: A list of products the producer will create, including quantity and creation time.
        :param marketplace: The Marketplace instance to interact with.
        :param republish_wait_time: Time to wait before retrying to publish a product if the buffer is full.
        :param kwargs: Keyword arguments for the thread, including 'name' and 'daemon'.
        """
        Thread.__init__(self, group=None, target=None, name=None, args=(), kwargs={},
                        daemon=kwargs.get("daemon"))
        self.products = products
        self.name = kwargs["name"]
        self.marketplace = marketplace      
        self.wait_time = republish_wait_time
        self.id_producer = 0

    def run(self):
        """
        The main execution loop for the producer thread.

        Registers with the marketplace and then enters an infinite loop to continuously
        produce and publish its products.
        """
        self.id_producer = self.marketplace.register_producer()
        # Invariant: The producer runs in an infinite loop to continuously supply products.
        while True:
            for prod in self.products:
                i = 0
                # Produces the specified quantity of the current product.
                while i < prod[1]:
                    verify = self.marketplace.publish(self.id_producer, prod[0])
                    # Invariant: If publishing fails, wait and retry.
                    while not verify:
                        time.sleep(self.wait_time)
                        verify = self.marketplace.publish(self.id_producer, prod[0])
                    # Simulates the time taken to produce the item.
                    time.sleep(prod[2])
                    i += 1