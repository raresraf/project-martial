
"""
@brief A flawed producer-consumer simulation of a marketplace.
@file consumer.py

This module simulates an e-commerce marketplace where `Producer` threads generate
and publish products, and `Consumer` threads purchase them. The `Marketplace`
class acts as the central hub. The module also defines a data model for products
using `dataclasses`.

WARNING: This implementation is critically flawed and not thread-safe.
1.  **No Synchronization**: The `Marketplace` class initializes several locks but
    NEVER uses them in the methods that modify shared data (`publish`,
    `add_to_cart`, `remove_from_cart`). This makes the entire class prone to
    severe race conditions.
2.  **Memory Leak**: The `place_order` method does not remove the cart after an
    order is completed, leading to unbounded memory growth in `self.carts`.
3.  **Inefficient Logic**: `add_to_cart` uses a highly inefficient nested loop to
    find products. Both `Producer` and `Consumer` use inefficient busy-wait
    loops with `time.sleep` instead of proper synchronization primitives like
    Condition variables or Events.
"""

from threading import Thread, Lock
import time
import collections

class Consumer(Thread):
    """
    Represents a consumer that buys products from the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes the consumer thread."""
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """The main loop for the consumer, processing each cart in its workload."""
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for product in cart:
                # Invariant: perform the specified quantity of add/remove operations.
                for _ in range(product["quantity"]):
                    if product["type"] == "add":
                        # Block Logic: Busy-wait until the product can be added.
                        while not self.marketplace.add_to_cart(cart_id, product["product"]):
                            time.sleep(self.retry_wait_time)

                    elif product["type"] == "remove":
                        self.marketplace.remove_from_cart(cart_id, product["product"])

            # Finalize the purchase and print the bought items.
            bought = self.marketplace.place_order(cart_id)
            for item in bought:
                print(self.kwargs['name'], "bought", item)

class Marketplace:
    """
    A central marketplace for producers and consumers.
    
    WARNING: This class is NOT thread-safe due to a failure to use its locks.
    """
    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace."""
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = 0
        self.cart_ids = 0
        # Stores products available from each producer. Access is not synchronized.
        self.producers_buffers = collections.defaultdict(list)
        # Stores items in each customer's cart. Access is not synchronized.
        self.carts = collections.defaultdict(list)
        
        # Locks are declared but never used in critical sections.
        self.register_producer_lock = Lock()
        self.new_cart_lock = Lock()
        self.add_to_cart_lock = Lock()
        self.remove_from_cart_lock = Lock()

    def register_producer(self):
        """Atomically registers a new producer and returns a unique ID."""
        with self.register_producer_lock:
            producer_id = self.id_producer
            self.id_producer += 1
        return str(producer_id)

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer.
        
        WARNING: NOT THREAD-SAFE. Concurrent calls can lead to race conditions
        when checking the buffer length and appending to it.
        """
        if producer_id in self.producers_buffers:
            if len(self.producers_buffers[producer_id]) >= self.queue_size_per_producer:
                return False
        
        self.producers_buffers[producer_id].append(product)
        return True

    def new_cart(self):
        """Atomically creates a new cart and returns its ID."""
        with self.new_cart_lock:
            cart_id = self.cart_ids
            self.cart_ids += 1
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart.
        
        WARNING: NOT THREAD-SAFE. Iterates and modifies shared producer buffers
        without a lock, which can cause exceptions and data corruption. Also
        uses a highly inefficient O(N*M) search.
        """
        for producer_id, products in self.producers_buffers.items():
            for prod in products:
                if product == prod:
                    self.carts[cart_id].append((product, producer_id))
                    products.remove(product)
                    return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart.
        
        WARNING: NOT THREAD-SAFE. Modifies both the cart and producer buffers
        without any locking.
        """
        for item in self.carts[cart_id]:
            if product == item[0]:
                self.carts[cart_id].remove(item)
                self.producers_buffers[item[1]].append(product)
                return

    def place_order(self, cart_id):
        """
        Returns the list of items in a cart.
        
        BUG: This method does not clear the cart after placing the order,
        resulting in a memory leak as the `self.carts` dictionary grows indefinitely.
        """
        return [i[0] for i in self.carts[cart_id]]


class Producer(Thread):
    """Represents a producer that generates and publishes products."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the producer thread."""
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs
        self.id_producer = self.marketplace.register_producer()

    def run(self):
        """
        Main loop for the producer, which continuously creates and publishes products.
        """
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    # Block Logic: Busy-wait until the product can be published.
                    while not self.marketplace.publish(self.id_producer, product[0]):
                        time.sleep(self.republish_wait_time)
                    time.sleep(product[2])

# The dataclasses below define a data model for products but are not directly
# used by the simulation logic, which only passes product names as strings.
from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base class for a product with a name and price."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A tea product with a specific type."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A coffee product with acidity and roast level."""
    acidity: str
    roast_level: str
