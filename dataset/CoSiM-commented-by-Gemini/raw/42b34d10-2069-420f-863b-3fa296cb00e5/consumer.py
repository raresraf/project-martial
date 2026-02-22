"""
This file models a multi-threaded producer-consumer simulation of a marketplace.

It appears to contain the contents of multiple files (`consumer.py`,
`marketplace.py`, `producer.py`, and `product.py`) concatenated together.
The code defines the core components of the simulation:
- Marketplace: The central shared resource where products are published and purchased.
- Consumer: A thread that simulates a user buying products.
- Producer: A thread that simulates a vendor publishing products.
- Product/Tea/Coffee: Data classes for the items being traded.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace.

    Each consumer runs in its own thread, processing a list of shopping carts.
    It performs 'add' and 'remove' operations and includes retry logic for when
    a product is not immediately available.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of carts, where each cart is a list of
                          product operations.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying
                                     to add a product to the cart.
            **kwargs: Keyword arguments passed to the Thread constructor (e.g., name).
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.consumer_name = kwargs["name"]

    def run(self):
        """The main execution loop for the consumer."""
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for data in cart:
                i = 0
                item = data["product"]
                operation = data["type"]

                while i < data["quantity"]:
                    if operation == "add":
                        # Attempt to add an item, retrying if it's not available.
                        available = self.marketplace.add_to_cart(cart_id, item)
                        if available:
                            i += 1
                        else:
                            time.sleep(self.retry_wait_time)

                    if operation == "remove":
                        self.marketplace.remove_from_cart(cart_id, item)
                        i += 1
            
            # Finalize the order and print the items.
            order = self.marketplace.place_order(cart_id)
            for item in order:
                print(self.consumer_name + " bought "+ str(item[0]))


from threading import Lock

class Marketplace:
    """
    The central marketplace that manages producers, products, and carts.

    This class attempts to provide thread-safe operations for a multi-producer,
    multi-consumer environment. While it correctly uses shared instance locks
    for some operations, several critical race conditions remain.
    """
    
    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace state and its synchronization locks."""
        self.queue_size_per_producer = queue_size_per_producer
        
        # State variables
        self.num_prod = 0
        self.num_carts = 0
        self.prod_num_items = []
        self.items = {} # {producer_id: [product1, product2, ...]}
        self.carts = {} # {cart_id: [(product, producer_id), ...]}

        # Shared locks for different operations
        self.register_lock = Lock()
        self.new_cart_lock = Lock()
        self.cart_lock = Lock()

    def register_producer(self):
        """Atomically registers a new producer and returns a unique producer ID."""
        with self.register_lock:
            prod_id = self.num_prod
            self.num_prod += 1

        self.prod_num_items.append(0)
        self.items[prod_id] = []
        return prod_id

    def publish(self, producer_id, product):
        """
        Publishes a product for a given producer.

        @warning This method has a race condition. The check for the producer's
                 queue size is performed *before* any lock is acquired. Two
                 threads could both read the count, find it to be under the limit,
                 and then both proceed to add an item, potentially violating the
                 `queue_size_per_producer` constraint. The list append and count
                 increment are also not protected by a lock.
        """
        if self.prod_num_items[producer_id] >= self.queue_size_per_producer:
            return False
        self.items[producer_id].append(product)
        self.prod_num_items[producer_id] += 1
        return True

    def new_cart(self):
        """Atomically creates a new shopping cart and returns a unique cart ID."""
        with self.new_cart_lock:
            cart_id = self.num_carts
            self.num_carts += 1
        self.carts[cart_id] = []
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Finds a product from any producer and reserves it for a specific cart.
        This operation is protected by a lock.
        """
        found = False
        with self.cart_lock:
            # Find the product from any producer's inventory.
            for i in self.items:
                if product in self.items[i]:
                    # Atomically remove from producer and note the source.
                    self.items[i].remove(product)
                    self.prod_num_items[i] -= 1
                    prod_id = i
                    found = True
                    break
        
        # If found, add the product and its original producer to the cart.
        if found:
            self.carts[cart_id].append((product, prod_id))
        return found

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to its original producer.

        @warning This method has a critical race condition. It modifies the cart
                 list (`self.carts[cart_id]`) and the producer's item list
                 (`self.items[prod_id]`) *outside* of any lock. Only the final
                 increment of the producer's item count is locked, leaving the
                 data structures vulnerable to corruption under concurrent access.
        """
        # Unsafe modification of the cart list.
        for item, producer in self.carts[cart_id]:
            if item is product:
                prod_id = producer
                self.carts[cart_id].remove((item, producer))
                break
        
        # Unsafe modification of the producer's item list.
        self.items[prod_id].append(product)

        # The lock only protects this single increment operation.
        with self.cart_lock:
            self.prod_num_items[prod_id] += 1

    def place_order(self, cart_id):
        """Finalizes an order by returning the cart's contents and removing the cart."""
        # This operation is atomic on the dictionary itself.
        res = self.carts.pop(cart_id)
        return res

# The following classes appear to belong in separate files.

from threading import Thread
import time

class Producer(Thread):
    """Represents a producer that publishes products to the marketplace."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """The main execution loop for the producer."""
        prod_id = self.marketplace.register_producer()
        while True:
            for (item, quantity, wait_time) in self.products:
                i = 0
                while i < quantity:
                    # Attempt to publish a product, retrying if the queue is full.
                    available = self.marketplace.publish(prod_id, item)
                    if available:
                        time.sleep(wait_time)
                        i += 1
                    else:
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base data class for a generic product."""
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
