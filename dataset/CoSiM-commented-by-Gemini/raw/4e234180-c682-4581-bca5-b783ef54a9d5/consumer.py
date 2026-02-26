"""
This module simulates a multi-threaded e-commerce marketplace.

It defines three main components:
- `Marketplace`: A central class that manages products and carts.
- `Producer`: A thread that publishes products to the marketplace.
- `Consumer`: A thread that simulates a user adding and removing items from a
  cart and placing an order.

It also defines simple data classes for the products being traded.

NOTE: This implementation has critical concurrency issues. The `Marketplace`
class is not thread-safe, and locking within the Producer/Consumer threads is
implemented incorrectly, which would lead to race conditions and data
corruption in a real multi-threaded environment. The comments describe the
intended functionality.
"""


from threading import Thread, Lock
import time


class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace.

    Each consumer processes a list of "carts", where each cart is a sequence of
    'add' or 'remove' operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.cart_id = 0

    def run(self):
        """The main logic for a consumer."""
        for cart in self.carts:
            # Note: This lock is created locally and is therefore ineffective
            # for providing mutual exclusion between different consumers.
            lock = Lock()
            lock.acquire()
            self.cart_id = self.marketplace.new_cart()
            lock.release()

            for ops in cart:
                type_operation = ops['type']
                product = ops['product']
                quantity = ops['quantity']
                i = 0

                if type_operation == "add":
                    # Keep trying to add the product until successful.
                    while i < quantity:
                        status = self.marketplace.add_to_cart(self.cart_id, product)
                        if not status:
                            # If add fails (e.g., product unavailable), wait and retry.
                            time.sleep(self.retry_wait_time)
                        else:
                            i += 1
                else: # type_operation == "remove"
                    while i < quantity:
                        self.marketplace.remove_from_cart(self.cart_id, product)
                        i += 1

            # Finalize the order and print the items.
            placed_order_cart = self.marketplace.place_order(self.cart_id)

            # This lock is also local and does not protect the shared `print`
            # resource from being interleaved by other threads.
            lock = Lock()
            for product_bought in placed_order_cart:
                lock.acquire()
                print("{} bought {}".format(self.name, product_bought))
                lock.release()



class Marketplace:
    """
    The central marketplace that manages producers, products, and carts.

    WARNING: This class is NOT thread-safe. Its methods directly manipulate
    shared lists without any locking, making it susceptible to race conditions.
    """

    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer
        self.count_producers = 0
        self.carts = []
        self.producer_products = []  # Products available for sale.
        self.reserved_products = []  # Products added to carts but not yet ordered.

    def register_producer(self):
        """Allocates a new ID and data structures for a producer."""
        self.producer_products.append([])
        self.reserved_products.append([])
        self.count_producers = self.count_producers + 1
        return self.count_producers - 1

    def publish(self, producer_id, product):
        """
        Allows a producer to list a product for sale.

        Returns:
            True if the product was successfully listed, False if the producer's
            queue is full.
        """
        if len(self.producer_products[producer_id]) < self.queue_size_per_producer:
            self.producer_products[producer_id].append(product)
            return True
        return False

    def new_cart(self):
        """Creates a new, empty cart and returns its ID."""
        self.carts.append([])
        return len(self.carts) - 1

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart if it's available from any producer.

        This involves moving the product from the producer's available stock to
        the cart and a reserved list.
        """
        # Inefficiently scans all producers for the product.
        for i in range(self.count_producers):
            if product in self.producer_products[i]:
                self.carts[cart_id].append(product)
                self.reserved_products[i].append(product)
                self.producer_products[i].remove(product)
                return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart, returning it to the producer's stock.
        """
        self.carts[cart_id].remove(product)

        # Scans to find which producer it was reserved from.
        for i in range(self.count_producers):
            if product in self.reserved_products[i]:
                self.reserved_products[i].remove(product)
                self.producer_products[i].append(product)
                return True
        return False

    def place_order(self, cart_id):
        """Finalizes an order, returning the list of products in the cart."""
        return self.carts[cart_id]


from threading import Thread, Lock
import time


class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = 0

    def run(self):
        """The main logic for a producer."""
        # This lock is local and ineffective for concurrency control.
        lock = Lock()
        lock.acquire()
        self.producer_id = self.marketplace.register_producer()
        lock.release()

        while True:
            for product in self.products:
                product_id = product[0]
                quantity = product[1]
                waiting_time = product[2]
                i = 0

                while i < quantity:
                    # Attempt to publish the product.
                    status = self.marketplace.publish(self.producer_id, product_id)
                    if not status:
                        # If queue is full, wait and retry.
                        time.sleep(self.republish_wait_time)
                    else:
                        i += 1
                        time.sleep(waiting_time)

# Simple, immutable data classes for representing products.
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