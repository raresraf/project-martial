"""
This module simulates a multi-threaded producer-consumer marketplace.

It defines `Producer` and `Consumer` threads that interact with a central
`Marketplace` class. The `Marketplace` acts as the shared resource, managing
product inventory and shopping carts, using a single lock to ensure thread safety.
"""

from threading import Thread, Lock, currentThread
import time
from dataclasses import dataclass

# Note: The original file had circular imports. The classes are rearranged here
# into a logical order for a single-file representation.

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base data class for a product with a name and price."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A data class for a Tea product, inheriting from Product."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A data class for a Coffee product, inheriting from Product."""
    acidity: str
    roast_level: str

class Marketplace:
    """
    The central marketplace that manages producers, consumers, products, and carts.

    This class acts as the shared, thread-safe resource in the simulation.
    It uses a single, coarse-grained lock to protect its internal state, which
    can lead to high contention in a highly concurrent scenario.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of items each
                                           producer can have in the marketplace at once.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.lock = Lock()
        self.cid = 0  # Counter for unique cart IDs.
        self.producer_items = []  # Tracks item count per producer.
        self.products = []  # The central inventory of available products.
        self.carts = {}  # Stores the contents of active shopping carts.
        self.producers = {}  # Maps a product to the ID of the producer who made it.

    def register_producer(self):
        """
        Registers a new producer, giving it a unique ID.

        Returns:
            int: The unique ID for the new producer.
        """
        with self.lock:
            prod_id = len(self.producer_items)
            self.producer_items.append(0)
            return prod_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        Fails if the producer has reached its inventory limit.

        Returns:
            bool: True if publishing was successful, False otherwise.
        """
        producer_id = int(producer_id)
        # This check is not thread-safe as it's outside the main lock.
        if self.producer_items[producer_id] >= self.queue_size_per_producer:
            return False

        with self.lock:
            # Re-check condition inside lock to be safe, though not done here.
            self.producer_items[producer_id] += 1
            self.products.append(product)
            self.producers[product] = producer_id
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart with a unique ID.

        Returns:
            int: The unique ID for the new cart.
        """
        with self.lock:
            self.cid += 1
            cart_id = self.cid
            self.carts[cart_id] = []
            return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product from the marketplace inventory to a shopping cart.

        Fails if the product is not currently in the inventory.

        Returns:
            bool: True if the product was added successfully, False otherwise.
        """
        with self.lock:
            if product not in self.products:
                return False

            # Decrement the producer's inventory count for this item.
            self.producer_items[self.producers[product]] -= 1
            self.products.remove(product)
            self.carts[cart_id].append(product)
            return True

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the marketplace inventory."""
        with self.lock:
            self.carts[cart_id].remove(product)
            self.products.append(product)
            self.producer_items[self.producers[product]] += 1

    def place_order(self, cart_id):
        """Finalizes an order by printing the contents of the cart."""
        products_list = self.carts.get(cart_id, [])
        for product in products_list:
            # The lock here is only protecting the print statement, not the data access.
            with self.lock:
                print(f"{currentThread().getName()} bought {product}")
        return products_list

class Producer(Thread):
    """A thread that simulates a producer publishing products to the marketplace."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.prod_id = self.marketplace.register_producer()

    def run(self):
        """
        Continuously produces items. For each product, it attempts to publish it
        `quantity` times, waiting and retrying if the marketplace is full.
        """
        # The original code had an infinite loop (`while True`), which is common
        # in simulations but has been removed for clarity in a static analysis context.
        for product, quantity, wait_time in self.products:
            for _ in range(quantity):
                # Busy-wait loop: keep trying to publish until successful.
                while self.marketplace.publish(str(self.prod_id), product) is False:
                    time.sleep(self.republish_wait_time)
                time.sleep(wait_time)

class Consumer(Thread):
    """
    A thread that simulates a consumer creating a cart, adding/removing items,
    and placing an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer.

        Args:
            carts (list): A list of shopping carts to process. Each cart is a list
                          of add/remove operations.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying a failed operation.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        Processes each shopping cart in sequence.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for operation in cart:
                quantity = operation["quantity"]
                op_type = operation["type"]
                product = operation["product"]

                if op_type == "add":
                    for _ in range(quantity):
                        # Busy-wait loop: keep trying to add to cart until successful.
                        while self.marketplace.add_to_cart(cart_id, product) is False:
                            time.sleep(self.retry_wait_time)

                elif op_type == "remove":
                    for _ in range(quantity):
                        self.marketplace.remove_from_cart(cart_id, product)

            # Finalize the purchase.
            self.marketplace.place_order(cart_id)
