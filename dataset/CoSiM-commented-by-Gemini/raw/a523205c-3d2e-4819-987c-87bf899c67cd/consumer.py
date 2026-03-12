"""
Implements a producer-consumer simulation for an e-commerce marketplace.

This file defines three main classes:
- Producer: A thread that produces items and adds them to the marketplace.
- Consumer: A thread that adds items from the marketplace to a shopping cart and "buys" them.
- Marketplace: The central, shared resource that manages product inventory and shopping carts.
"""

import time
from threading import Thread, Lock


class Consumer(Thread):
    """
    Represents a consumer thread that shops at the marketplace.
    
    Each consumer processes a list of shopping lists ('carts'), adding and
    removing items before finally placing an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of "shopping lists", where each list contains
                          add/remove operations.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying a failed operation.
            **kwargs: Catches extra arguments, using "name" for the thread's name.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        """
        The main loop for the consumer. It processes each shopping cart in sequence.
        """
        for cart in self.carts:
            # Get a new, unique cart from the marketplace for this shopping session.
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                # Process add-to-cart operations.
                if operation["type"] == "add":
                    i = 0
                    while i < operation["quantity"]:
                        # Attempt to add the product to the cart.
                        verify = self.marketplace.add_to_cart(cart_id, operation["product"])
                        # If adding fails (e.g., product is out of stock), busy-wait and retry.
                        while not verify:
                            time.sleep(self.wait_time)
                            verify = self.marketplace.add_to_cart(cart_id, operation["product"])
                        i += 1
                
                # Process remove-from-cart operations.
                elif operation["type"] == "remove":
                    i = 0
                    while i < operation["quantity"]:
                        self.marketplace.remove_from_cart(cart_id, operation["product"])
                        i += 1

            # Finalize the purchase.
            orders = self.marketplace.place_order(cart_id)
            for order in orders:
                print("%s bought %s" % (self.name, order[0]))


class Marketplace:
    """
    The central marketplace that manages producers, products, and consumer carts.

    This class acts as the shared resource in the producer-consumer simulation.
    It synchronizes access to its internal state, but contains several
    performance bottlenecks and race conditions in its design.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products each
                                           producer can have listed at one time.
        """
        self.queue_size = queue_size_per_producer
        self.producer_id = 0
        self.cart_id = 0
        
        # In-memory storage for products and carts.
        self.producers_buffers = {}
        self.carts_list = {}
        
        # Locks for synchronizing access to shared state.
        self.lock_buffers = Lock()
        self.lock_carts = Lock()

    def register_producer(self):
        """
        Registers a new producer, giving them a unique ID and a product buffer.
        Returns:
            int: The new producer's unique ID.
        """
        with self.lock_buffers:
            self.producer_id += 1
            self.producers_buffers[self.producer_id] = []
            new_id = self.producer_id
        return new_id

    def publish(self, producer_id, product):
        """
        Allows a producer to list a product for sale.

        Note: The lock is held for the entire duration, which is not strictly
        necessary and could be optimized. Using `with` is safer than `acquire/release`.

        Returns:
            bool: True if publishing was successful, False if the producer's buffer is full.
        """
        self.lock_buffers.acquire()
        if len(self.producers_buffers[producer_id]) < self.queue_size:
            self.producers_buffers[producer_id].append(product)
            self.lock_buffers.release()
            return True

        self.lock_buffers.release()
        return False

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.
        
        Returns:
            int: The new cart's unique ID.
        """
        with self.lock_carts:
            self.cart_id += 1
            self.carts_list[self.cart_id] = []
            new_cart = self.cart_id
        return new_cart

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's cart by finding it in a producer's buffer.
        
        Note: This method holds a global lock (`lock_buffers`) during the entire
        search and transfer operation, serializing all "add to cart" actions across
        all consumers and creating a significant performance bottleneck.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        self.lock_buffers.acquire()
        for producer in self.producers_buffers:
            for prod in self.producers_buffers[producer]:
                if prod == product:
                    # Move product from producer buffer to consumer cart.
                    self.producers_buffers[producer].remove(prod)
                    self.carts_list[cart_id].append((prod, producer))
                    self.lock_buffers.release()
                    return True

        self.lock_buffers.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the original producer.

        CRITICAL FLAW: This method has a race condition. It modifies `self.carts_list`
        within a lock, but then modifies `self.producers_buffers` *without* holding
        `lock_buffers`, which can lead to data corruption if multiple threads call
        this concurrently.
        """
        for (prod, producer) in self.carts_list[cart_id]:
            if prod == product:
                # This part is thread-safe.
                with self.lock_carts:
                    self.carts_list[cart_id].remove((prod, producer))
                
                # This part is NOT thread-safe and is a race condition.
                self.producers_buffers[producer].append(prod)

                break

    def place_order(self, cart_id):
        """
        Finalizes an order by returning the contents of the cart.
        
        Note: This is not thread-safe if the cart could be modified elsewhere.
        """
        return self.carts_list[cart_id]


class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of (product, quantity, delay) tuples.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish.
            **kwargs: Catches extra arguments like "name" and "daemon".
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
        The main loop for the producer. It continuously tries to publish its products.
        """
        self.id_producer = self.marketplace.register_producer()
        while True:
            for prod in self.products:
                i = 0
                while i < prod[1]:
                    # Attempt to publish the product.
                    verify = self.marketplace.publish(self.id_producer, prod[0])
                    # If publishing fails, busy-wait and retry.
                    while not verify:
                        time.sleep(self.wait_time)
                        verify = self.marketplace.publish(self.id_producer, prod[0])
                    
                    time.sleep(prod[2])
                    i += 1