"""
This module simulates a multi-threaded producer-consumer model for an e-commerce
marketplace.

It defines four main components:
- Marketplace: The central shared resource where products are published and carts
  are managed. It handles synchronization between producers and consumers.
- Producer: A thread that produces and publishes items to the marketplace.
- Consumer: A thread that simulates a customer adding and removing items from a
  shopping cart and eventually placing an order.
- Product classes: Dataclasses representing the items being traded.
"""

import sys
import time
from threading import Lock, Thread, currentThread


class Consumer(Thread):
    """A thread that simulates a consumer interacting with the marketplace."""

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping lists. Each shopping list is a
                          sequence of operations (add/remove products).
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait in seconds before retrying a
                                     failed operation (e.g., product not available).
            **kwargs: Additional keyword arguments for the Thread constructor.
        """

        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # Map operation strings to marketplace methods for easy dispatch.
        self.operations = {"add": marketplace.add_to_cart,
                           "remove": marketplace.remove_from_cart}

    def run(self):
        """The main execution loop for the consumer.

        Processes each cart in its list of tasks. For each cart, it performs
        the specified add/remove operations and then places the order.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for operation in cart:
                quantity = operation["quantity"]
                product = operation["product"]
                operation_type = operation["type"]

                # Retry the operation until the desired quantity is fulfilled.
                while quantity > 0:
                    # Attempt the operation (e.g., add_to_cart).
                    if self.operations[operation_type](cart_id, product) is not False:
                        # If successful, decrement the remaining quantity.
                        quantity -= 1
                    else:
                        # If it fails (e.g., product unavailable), wait and retry.
                        time.sleep(self.retry_wait_time)

            self.marketplace.place_order(cart_id)


class Marketplace:
    """A thread-safe marketplace for producers and consumers.

    This class manages the inventory of products, producer capacities, and
    consumer shopping carts. It uses locks to ensure that concurrent access
    from multiple producer and consumer threads is handled safely.
    """

    def __init__(self, queue_size_per_producer):
        """Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in the marketplace at once.
        """
        self.carts_lock = Lock()
        self.carts = []  # Stores items in each active shopping cart.

        self.producers_lock = Lock()
        self.producers_capacity = queue_size_per_producer
        self.producers_sizes = []  # Tracks current publication count for each producer.
        self.products = []         # The general pool of available products.

    def register_producer(self):
        """Registers a new producer with the marketplace.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        with self.producers_lock:
            self.producers_sizes.append(0)
            return len(self.producers_sizes) - 1

    def publish(self, producer_id, product):
        """Allows a producer to publish a product to the marketplace.

        The operation will fail if the producer is already at its capacity.

        Args:
            producer_id (int): The ID of the producer.
            product (Product): The product to be published.

        Returns:
            bool: True if the product was published successfully, False otherwise.
        """
        with self.producers_lock:
            if self.producers_sizes[producer_id] == self.producers_capacity:
                return False  # Producer is at capacity.

            self.producers_sizes[producer_id] += 1
            self.products.append((product, producer_id))
            return True

    def new_cart(self):
        """Creates a new, empty shopping cart.

        Returns:
            int: The unique ID for the new cart.
        """
        with self.carts_lock:
            self.carts.append([])
            return len(self.carts) - 1

    def add_to_cart(self, cart_id, product):
        """Adds a product from the marketplace to a consumer's cart.

        This involves finding the product in the general pool and moving it.
        Note: The locking in this method is imperfect; it releases the
        producers_lock before modifying the carts list, which can lead to
        race conditions.

        Args:
            cart_id (int): The ID of the cart to add to.
            product (Product): The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """

        self.producers_lock.acquire()
        for (prod, prod_id) in self.products:
            if prod == product:
                self.producers_sizes[prod_id] -= 1
                self.products.remove((prod, prod_id))
                self.producers_lock.release()
                # This append is not protected by carts_lock and can cause a race condition.
                self.carts[cart_id].append((prod, prod_id))
                return True

        self.producers_lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a consumer's cart and returns it to the marketplace.
        
        Note: This method has potential race conditions as it iterates and modifies
        the cart list without acquiring the `carts_lock`.
        
        Args:
            cart_id (int): The ID of the cart.
            product (Product): The product to remove.
        """

        for (prod, prod_id) in self.carts[cart_id]:
            if prod == product:
                self.carts[cart_id].remove((prod, prod_id))
                self.producers_lock.acquire()
                self.products.append((prod, prod_id))
                self.producers_sizes[prod_id] += 1
                self.producers_lock.release()
                return

    def place_order(self, cart_id):
        """Finalizes an order and prints the contents of the cart.

        Note: This method accesses the cart list without acquiring the `carts_lock`,
        which can lead to race conditions if other threads are modifying the cart.
        """
        order = ""
        for (product, _) in self.carts[cart_id]:
            order += "{} bought {}\n".format(
                currentThread().getName(), product)
        sys.stdout.write(order)
        return self.carts[cart_id]


class Producer(Thread):
    """A thread that simulates a producer publishing products."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes a Producer thread.

        Args:
            products (list): A list of (product, quantity, wait_time) tuples.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish
                                         if the marketplace is full.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = marketplace.register_producer()

    def run(self):
        """The main execution loop for the producer.

        Continuously tries to publish its products to the marketplace.
        """
        # This is an infinite loop, so the producer never stops.
        while True:
            for (product, quantity, wait_time) in self.products:
                # Publish the specified quantity of the product.
                while quantity > 0:
                    if self.marketplace.publish(self.producer_id, product):
                        quantity -= 1
                        time.sleep(wait_time)
                    else:
                        # Wait if the producer's capacity in the marketplace is full.
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass for a Tea product, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass for a Coffee product, inheriting from Product."""
    acidity: str
    roast_level: str
