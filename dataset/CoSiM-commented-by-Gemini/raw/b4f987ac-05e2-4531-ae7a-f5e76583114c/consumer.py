# -*- coding: utf-8 -*-
"""
#
# Module: consumer.py
#
# @brief Implements a producer-consumer simulation modeling an e-commerce marketplace.
#
# @details This module establishes a concurrent system with multiple producer threads (vendors)
# and consumer threads (customers) interacting through a central `Marketplace` class.
# The `Marketplace` acts as a monitor, coordinating thread-safe operations for
# publishing products, managing shopping carts, and placing orders. The simulation
# highlights classic concurrency patterns, including resource sharing and synchronization.
#
# @note The current implementation of the `Marketplace` is not fully thread-safe.
# Operations like `add_to_cart` and `remove_from_cart` lack atomicity, which could
# lead to race conditions under heavy concurrent load.
#
# Classes:
#     Consumer: A thread representing a customer who browses and buys products.
#     Marketplace: The central, thread-aware intermediary for all e-commerce operations.
#     Producer: A thread representing a vendor who creates and lists products.
#     Product, Tea, Coffee: Immutable dataclasses for representing product types.
#
# Concurrency Model:
#   - Producers and Consumers are implemented as `threading.Thread` subclasses.
#   - The `Marketplace` uses `threading.Lock` to protect critical sections,
#     specifically for generating unique IDs and ensuring atomic print operations.
#
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Represents a consumer that buys products from the marketplace.
    Each consumer operates in its own thread, simulating concurrent user activity.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of carts, where each cart is a list of items to process.
                          This represents a sequence of shopping sessions for the consumer.
            marketplace (Marketplace): The central marketplace to interact with.
            retry_wait_time (float): The time in seconds to wait before retrying a failed
                                     operation, implementing a simple backoff strategy.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution loop for the consumer thread.

        It processes each cart sequentially, simulating a user's shopping journey.
        For each cart, it attempts to add or remove items and finally places an order.
        """
        # Block Logic: Iterates through each shopping cart assigned to this consumer.
        # Pre-condition: `self.carts` contains a list of shopping sessions.
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            # Block Logic: Processes each item within a single shopping cart.
            # Invariant: `quantity` tracks the number of successful operations for the current item.
            for item in cart:
                quantity = 0
                # Block Logic: Implements a retry mechanism for cart operations.
                # This optimistic loop attempts an operation until it succeeds, which is crucial
                # in a concurrent environment where a product might be temporarily unavailable.
                while quantity < item["quantity"]:
                    if item["type"] == "add":
                        # Attempt to add a product to the cart.
                        ver = self.marketplace.add_to_cart(cart_id, item["product"])
                    else:
                        # Attempt to remove a product from the cart.
                        ver = self.marketplace.remove_from_cart(cart_id, item["product"])

                    if ver:
                        quantity += 1
                    else:
                        # If the operation fails, wait before retrying to avoid busy-waiting
                        # and to allow other threads to make progress.
                        time.sleep(self.retry_wait_time)

            # Finalizes the transaction for the current cart.
            self.marketplace.place_order(cart_id)


from threading import Lock, currentThread

class Marketplace:
    """
    The central marketplace that manages producers, consumers, and products.

    This class is designed to be a thread-safe monitor, using locks to protect
    shared data structures from race conditions. It orchestrates the entire
    e-commerce simulation.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have listed at one time. This
                                           acts as a form of back-pressure.
        """
        self.max_products_per_size = queue_size_per_producer
        self.carts = {}
        self.producers = {}
        self.reserved = {}

        self.id_cart = 0
        self.id_producer = 0

        # Synchronization Primitive: Protects access to the cart ID counter (`id_cart`)
        # to ensure that each new cart receives a unique identifier.
        self.lock_id_cart = Lock()
        # Synchronization Primitive: Protects access to the producer ID counter (`id_producer`)
        # to guarantee unique producer registration.
        self.lock_id_producer = Lock()
        # Synchronization Primitive: Ensures that console output from different threads
        # is not interleaved, making logs readable.
        self.lock_print = Lock()

    def register_producer(self):
        """
        Registers a new producer and returns a unique producer ID.
        This method is thread-safe.
        """
        # Critical Section: Ensures atomic increment and assignment of producer ID.
        with self.lock_id_producer:
            self.id_producer += 1
            prod_id = self.id_producer

        self.producers[prod_id] = []
        return prod_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        @note This operation is not guaranteed to be atomic if multiple producers
              were to share the same producer_id, though the current design
              assigns a unique ID to each producer thread.
        """
        prod_id = int(producer_id)
        # Block Logic: Enforces a limit on the number of products per producer,
        # preventing any single producer from flooding the marketplace.
        if len(self.producers[prod_id]) >= self.max_products_per_size:
            return False

        self.producers[prod_id].append(product)
        return True

    def new_cart(self):
        """
        Creates a new, empty cart and returns its ID.
        This method is thread-safe.
        """
        # Critical Section: Ensures atomic increment and assignment of cart ID.
        with self.lock_id_cart:
            self.id_cart += 1
            cart_id = self.id_cart
        self.carts[cart_id] = []
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified cart.

        This method moves the product from the producer's available stock
        to a reserved state, associated with the cart.

        @warning This method is NOT thread-safe. The check for product availability
                 and its subsequent removal are not an atomic operation. This can lead
                 to a race condition where two consumers might successfully "claim" the
                 same last item, leading to an inconsistent state. A lock is needed
                 to protect the `producers` and `reserved` dictionaries.
        """
        ver = False
        # Block Logic: Searches for the requested product across all producers' inventories.
        for _ in self.producers:
            if product in self.producers[_]:
                ver = True
                key = _
                break

        if not ver:
            return False

        # Non-atomic section: The product is removed from the public inventory
        # and moved to a reserved state pending order completion.
        self.producers[key].remove(product)
        if key in self.reserved.keys():
            self.reserved[key].append(product)
        else:
            self.reserved[key] = []
            self.reserved[key].append(product)

        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specified cart.

        This method moves the product from the reserved state back to the
        producer's available stock.

        @warning This method is NOT thread-safe. Similar to `add_to_cart`, the
                 read-then-write operations on shared `reserved` and `producers`
                 collections are not atomic and can lead to race conditions.
        """
        ver = True
        # Block Logic: Locates the product within the reserved items list.
        for key in self.reserved:
            for cnt in self.reserved[key]:
                if cnt == product:
                    ver = False
                    rem = key
                    break
            if not ver:
                break

        self.carts[cart_id].remove(product)

        # Non-atomic section: The product is moved from reserved back to available.
        self.producers[rem].append(product)
        self.reserved[rem].remove(product)
        return True

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.

        This method simulates the checkout process by printing the items bought
        and then clearing the cart. The print operation is thread-safe.
        """
        res = []
        res.extend(self.carts[cart_id])
        del self.carts[cart_id]

        # Block Logic: Iterates through the purchased items to log the transaction.
        # The use of `lock_print` ensures that output from concurrent orders is not mixed.
        for cnt in res:
            with self.lock_print:
                print("{} bought {}".format(currentThread().getName(), cnt))
        return res


from threading import Thread
import time

class Producer(Thread):
    """
    Represents a producer that creates and publishes products to the marketplace.
    Each producer operates in its own thread, simulating vendors adding inventory.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products to be produced, structured as tuples of
                             (product, quantity, production_time).
            marketplace (Marketplace): The marketplace to publish to.
            republish_wait_time (float): Time to wait before retrying to publish if the
                                         marketplace is at capacity for this producer.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Each producer registers itself with the marketplace to get a unique identifier.
        self.prod_id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution loop for the producer thread.

        It continuously produces and attempts to publish products to the marketplace,
        respecting production times and marketplace capacity limits.
        """
        # Block Logic: An infinite loop to continuously produce and publish items.
        # This simulates a vendor that is always restocking their inventory.
        while True:
            # Block Logic: Iterates through the catalog of products this producer can create.
            for(product, num_prod, wait_time) in self.products:
                # Block Logic: Produces a specific quantity of a single product.
                for quantity in range(num_prod):
                    # Attempt to publish a product to the marketplace.
                    if self.marketplace.publish(str(self.prod_id), product):
                        # Functional Utility: Simulates the time taken to produce one unit of the product.
                        time.sleep(wait_time)
                    else:
                        # If publishing fails (e.g., producer's queue is full), wait and then retry.
                        time.sleep(self.republish_wait_time)
                        # Inline: Decrementing the counter effectively forces a retry of the current
                        # product, ensuring that the target quantity is eventually produced.
                        quantity -= 1


from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    A simple dataclass for a generic product.
    `frozen=True` makes instances of this class immutable, which is a good
    practice for objects that are shared between threads.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    A dataclass for Tea, inheriting from Product.
    Represents a specific type of product with additional attributes.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    A dataclass for Coffee, inheriting from Product.
    Represents a specific type of product with additional attributes.
    """
    acidity: str
    roast_level: str