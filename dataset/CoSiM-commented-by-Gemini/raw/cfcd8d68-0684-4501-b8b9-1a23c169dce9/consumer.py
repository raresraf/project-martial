"""
Models a Producer-Consumer simulation with a Marketplace.

This module contains the classes for a multithreaded simulation of a marketplace
with producers and consumers. Producers create products and add them to the
marketplace, while consumers add products to carts and place orders. The
Marketplace class orchestrates the interactions and ensures thread safety.
"""

import time
from threading import Thread, Lock
from tema.product import Product


class Marketplace:
    """
    A thread-safe marketplace for producers and consumers.

    This class manages inventories of products from multiple producers and handles
    shopping carts for multiple consumers. It uses locks to ensure that concurrent
    operations are handled safely.
    """

    def __init__(self, queue_size_per_producer: int):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in the marketplace at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer

        # Each producer gets a dedicated queue and a lock.
        self.producer_queues = []
        # Each consumer gets a cart.
        self.consumer_carts = []

        # Locks to protect registration and cart creation.
        self.register_producer_lock = Lock()
        self.new_cart_lock = Lock()

    def register_producer(self) -> int:
        """
        Registers a new producer, allocating a dedicated queue and returning an ID.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        with self.register_producer_lock:
            producer_id = len(self.producer_queues)
            self.producer_queues.append(([], Lock()))
        return producer_id

    def publish(self, producer_id: int, product: Product) -> bool:
        """
        Adds a product to a specific producer's queue.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Product): The product to be published.

        Returns:
            bool: True if the product was published successfully, False if the
                  producer's queue was full.
        """
        queue, lock = self.producer_queues[producer_id]
        with lock:
            if len(queue) >= self.queue_size_per_producer:
                return False
            queue.append(product)
        return True

    def new_cart(self) -> int:
        """
        Creates a new, empty shopping cart for a consumer.

        Returns:
            int: The unique ID assigned to the new cart.
        """
        with self.new_cart_lock:
            cart_id = len(self.consumer_carts)
            self.consumer_carts.append([])
        return cart_id

    def add_to_cart(self, cart_id: int, product: Product) -> bool:
        """
        Moves a product from any producer's queue to a consumer's cart.

        This method iterates through all producer queues to find the requested
        product. The search is not guaranteed to be fair.

        Args:
            cart_id (int): The ID of the target cart.
            product (Product): The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        cart = self.consumer_carts[cart_id]
        # Iterate through producers to find the product.
        for producer_id, (queue, lock) in enumerate(self.producer_queues):
            with lock:
                try:
                    # Attempt to remove the product from the producer's queue.
                    queue.remove(product)
                except ValueError:
                    # Product not in this queue, try the next one.
                    continue
            
            # Product found, add it to the cart with its original producer_id.
            cart.append((product, producer_id))
            return True
        # Product was not found in any producer's queue.
        return False

    def remove_from_cart(self, cart_id: int, product: Product) -> bool:
        """
        Removes a product from a cart and returns it to its original producer's queue.

        Args:
            cart_id (int): The ID of the cart.
            product (Product): The product to remove.

        Returns:
            bool: True if the product was found and removed, False otherwise.
        """
        cart = self.consumer_carts[cart_id]
        for i, (prod, producer_id) in enumerate(cart):
            if prod == product:
                del cart[i]
                # Return the product to the original producer's queue.
                queue, lock = self.producer_queues[producer_id]
                with lock:
                    queue.append(prod)
                return True
        return False

    def place_order(self, cart_id) -> list:
        """
        Finalizes an order by returning the contents of the cart.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: A list of the products in the cart.
        """
        cart = self.consumer_carts[cart_id]
        return [product for product, producer_id in cart]


class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace.

    Each consumer processes a list of carts, with each cart containing a series
    of actions (add/remove products).
    """

    def __init__(self,
                 carts: list,
                 marketplace: Marketplace,
                 retry_wait_time: int,
                 **kwargs) 
    :
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of carts to be processed. Each cart is a list of actions.
            marketplace (Marketplace): The shared marketplace object.
            retry_wait_time (int): Time in seconds to wait before retrying to add a
                                 product if it's not available.
            **kwargs: Arguments for the Thread base class.
        """
        super().__init__(**kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution loop for the consumer.

        Iterates through its assigned carts, executes the add/remove actions for
        each product, and finally places the order.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for action in cart:
                type_ = action['type']
                product = action['product']
                qty = action['quantity']

                for _ in range(qty):
                    if type_ == 'add':
                        # If a product is not available, wait and retry.
                        while not self.marketplace.add_to_cart(cart_id, product):
                            time.sleep(self.retry_wait_time)
                    elif type_ == 'remove':
                        # Assumes the product is in the cart.
                        self.marketplace.remove_from_cart(cart_id, product)

            order = self.marketplace.place_order(cart_id)

            for product in order:
                print(f'{self.name} bought {product}')


class Producer(Thread):
    """
    Represents a producer thread that creates products and publishes them.
    """
    def __init__(self,
                 products: list,
                 marketplace: Marketplace,
                 republish_wait_time: int,
                 **kwargs) 
    :
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products to produce, where each item is a
                             tuple of (product, quantity, wait_time).
            marketplace (Marketplace): The shared marketplace object.
            republish_wait_time (int): Time in seconds to wait before retrying to
                                     publish a product if the queue is full.
            **kwargs: Arguments for the Thread base class.
        """
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Register with the marketplace to get a unique producer ID.
        self.id_ = self.marketplace.register_producer()

    def run(self):
        """
        The main execution loop for the producer.

        Continuously produces products and tries to publish them to the marketplace.
        If the marketplace queue is full, it waits and retries.
        """
        while True:
            for product, qty, wait_time in self.products:
                for _ in range(qty):
                    time.sleep(wait_time)
                    # If the queue is full, wait and retry publishing.
                    while not self.marketplace.publish(self.id_, product):
                        time.sleep(self.republish_wait_time)
