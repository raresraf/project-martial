"""
A multi-threaded producer-consumer marketplace simulation.

This script models a marketplace with producers who create products and consumers
who purchase them. The `Marketplace` class serves as the central hub for these
interactions. The simulation uses threads for concurrent producer and consumer
actions, with basic locking for ID generation but non-thread-safe methods for
core marketplace operations.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    Represents a consumer that processes a list of shopping carts.

    Each consumer thread is assigned a set of carts and executes the specified
    'add' or 'remove' operations for each product. If an operation fails
    (e.g., product is unavailable), it waits and retries.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        Args:
            carts (list): A list of shopping carts, with each cart containing
                          a list of operations to perform.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying
                                     a failed operation.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution logic for the consumer thread.

        Processes each cart by creating a new cart session in the marketplace,
        executing all add/remove operations, and finally placing the order.
        """
        for cart in self.carts:
            # Pre-condition: A new cart session is created for the transaction.
            c_id = self.marketplace.new_cart()

            # Block Logic: Iterates through each operation in the shopping list.
            for cons_op in cart:
                num_of_ops = 0

                # Block Logic: Repeats the operation until the desired quantity is met.
                while num_of_ops < cons_op["quantity"]:
                    if cons_op["type"] == "add":
                        ret = self.marketplace.add_to_cart(str(c_id), cons_op["product"])
                    else:
                        ret = self.marketplace.remove_from_cart(str(c_id), cons_op["product"])

                    # Invariant: The operation is retried after a delay upon failure.
                    if ret:
                        num_of_ops += 1
                    else:
                        time.sleep(self.retry_wait_time)

            self.marketplace.place_order(c_id)

from threading import Lock, currentThread


class Marketplace:
    """
    Manages product inventories and facilitates transactions.

    This class acts as the central exchange for products, maintaining separate
    inventories for each producer. It uses locks to ensure thread-safe
    generation of producer and cart IDs, but core product handling methods
    like `add_to_cart` and `publish` are not thread-safe.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of items each
                                           producer can list at one time.
        """
        self.max_size = queue_size_per_producer
        self.producers = {}
        self.p_index = 0
        self.carts = {}
        self.c_index = 0

        self.lock_register = Lock()
        self.lock_carts = Lock()


    def register_producer(self):
        """
        Thread-safely registers a new producer and provides a unique ID.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        with self.lock_register:
            p_id = self.p_index
            self.producers[p_id] = []
            self.p_index += 1

        return p_id


    def publish(self, producer_id, product):
        """
        Adds a product to a specific producer's inventory queue.
        
        This operation is not thread-safe.

        Args:
            producer_id (str): The ID of the producer.
            product: The product to add to the inventory.

        Returns:
            bool: True if the product was added, False if the queue was full.
        """
        p_id = int(producer_id)

        # Pre-condition: Check if the producer's queue has space.
        if len(self.producers[p_id]) >= self.max_size:
            return False

        self.producers[p_id].append(product)

        return True

    def new_cart(self):
        """
        Thread-safely creates a new shopping cart and returns its unique ID.

        Returns:
            int: The unique ID for the new cart.
        """
        with self.lock_carts:
            c_id = self.c_index
            self.carts[c_id] = []
            self.c_index += 1

        return c_id


    def add_to_cart(self, cart_id, product):
        """
        Searches all producer inventories for a product and adds it to a cart.
        
        This operation is not thread-safe as it iterates over and modifies
        producer queues without a lock.

        Args:
            cart_id (str): The ID of the cart to add the product to.
            product: The product to find and add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        c_id = int(cart_id)
        
        # Block Logic: Iterates through all producers and all their products
        # to find the requested item.
        for i in range(0, len(self.producers)):
            for prod in self.producers[i]:
                if product == prod:
                    self.producers[i].remove(product)
                    self.carts[c_id].append(product)
                    return True

        return False


    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to a producer's queue.

        This operation is not thread-safe. It contains a logical flaw where the
        product is always returned to the last registered producer's inventory.

        Args:
            cart_id (str): The ID of the cart.
            product: The product to remove.

        Returns:
            bool: True if the product was in the cart and removed, False otherwise.
        """
        c_id = int(cart_id)

        if product in self.carts[c_id]:
            self.carts[c_id].remove(product)
            # Returns the product to the last registered producer's queue.
            self.producers[self.p_index - 1].append(product)
            return True

        return False

    def place_order(self, cart_id):
        """
        Finalizes an order by retrieving cart items and printing them.

        Args:
            cart_id (int): The ID of the cart to be finalized.

        Returns:
            list: The list of products from the finalized cart.
        """
        c_id = int(cart_id)

        cart = self.carts.pop(c_id, None)

        for prod in cart:
            print("{} bought {}".format(currentThread().getName(), prod))

        return cart

import time
from threading import Thread


class Producer(Thread):
    """
    Represents a producer that generates and publishes products to the marketplace.
    
    The producer runs in an infinite loop, attempting to publish its designated
    products. It will wait and retry if the marketplace queue is full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer instance.

        Args:
            products (list): A list of products to generate, including quantity
                             and production time.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying a publish
                                         operation.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.p_id = self.marketplace.register_producer()

    def run(self):
        """The main execution logic, running in an infinite loop."""
        while 1:
            # Block Logic: Iterates through this producer's catalog of items.
            for p_list in self.products:
                num_of_p = 0

                # Block Logic: Produces the specified quantity of a single item.
                while num_of_p < p_list[1]:
                    ret = self.marketplace.publish(str(self.p_id), p_list[0])
                    
                    # Invariant: If publishing is successful, waits for the item's
                    # designated production time. Otherwise, waits and retries.
                    if ret:
                        time.sleep(p_list[2])
                        num_of_p += 1
                    else:
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing a type of tea."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass representing a type of coffee."""
    acidity: str
    roast_level: str