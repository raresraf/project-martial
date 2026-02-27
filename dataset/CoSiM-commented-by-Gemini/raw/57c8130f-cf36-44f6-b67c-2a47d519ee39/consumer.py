"""
This file contains a multi-threaded producer-consumer simulation using a central
Marketplace.

WARNING: The Marketplace implementation in this file has critical concurrency
bugs. It either uses local locks that provide no protection or completely omits
locking when modifying shared state, making it not thread-safe.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    A consumer thread that simulates purchasing items from the marketplace.
    """
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes the Consumer."""
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution logic. Creates a cart, performs all assigned operations
        with a busy-wait retry loop, and then places the order.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for order in cart:
                if order["type"] == "add":
                    for i in range(order["quantity"]):
                        # This nested loop structure is a convoluted way to retry
                        # the 'add' operation until it succeeds.
                        while True:
                            out = self.marketplace.add_to_cart(cart_id, order["product"])
                            while not out:
                                time.sleep(self.retry_wait_time)
                                out = self.marketplace.add_to_cart(cart_id, order["product"])
                            else:
                                break
                elif order["type"] == "remove":
                    for i in range(order["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, order["product"])
            self.marketplace.place_order(cart_id)


from threading import currentThread, Lock

class Marketplace:
    """
    A marketplace simulation that is NOT THREAD-SAFE due to incorrect locking.
    """
    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace."""
        self.max_q_per_prod = queue_size_per_producer
        self.producer_id = 0
        # Maps producer_id to a list of products they have published.
        self.products = {}
        self.cart_id = 0
        # Maps cart_id to a list of (product, producer_id) tuples.
        self.carts = {}
        # A redundant list of all available products.
        self.marketplace = []

    def register_producer(self):
        """
        Registers a new producer.

        WARNING: This method is NOT thread-safe. It creates a new local lock
        that protects nothing, leading to a race condition on `self.producer_id`.
        """
        lock = Lock()
        lock.acquire()
        self.producer_id += 1
        self.products[self.producer_id] = []
        lock.release()
        return self.producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product.

        WARNING: This method is NOT thread-safe. It modifies shared state
        (`self.marketplace`, `self.products`) without any locks.
        """
        num_prod = self.products[producer_id]
        if len(num_prod) >= self.max_q_per_prod:
            return False

        self.marketplace.append((product, producer_id))
        num_prod.append(product)
        return True

    def new_cart(self):
        """
        Creates a new cart.

        WARNING: This method is NOT thread-safe. It creates a new local lock
        that protects nothing, leading to a race condition on `self.cart_id`.
        """
        lock = Lock()
        lock.acquire()
        self.cart_id += 1
        cart_id = self.cart_id
        self.carts[cart_id] = []
        lock.release()
        return cart_id
    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart.

        WARNING: This method is NOT thread-safe. It modifies shared state
        (`self.marketplace`, `self.carts`, `self.products`) without any locks.
        """
        for (product_type, producer_id) in self.marketplace:
            if product_type == product:
                if product in self.products[producer_id]:
                    self.carts[cart_id].append((product, producer_id))
                    self.marketplace.remove((product_type, producer_id))
                    self.products[producer_id].remove(product)
                    return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart.

        WARNING: This method is NOT thread-safe. It modifies shared state
        (`self.carts`, `self.marketplace`, `self.products`) without any locks.
        """
        for (product_type, _producer_id) in self.carts[cart_id]:
            if product_type == product:
                self.carts[cart_id].remove((product, _producer_id))
                self.marketplace.append((product_type, _producer_id))
                self.products[_producer_id].append(product)
                break

    def place_order(self, cart_id):
        """
        Finalizes an order and prints the purchased items.

        WARNING: This method is NOT thread-safe. It reads from `self.carts`
        without a lock.
        """
        for (product, _producer_id) in self.carts[cart_id]:
            print("{} bought {}".format(currentThread().getName(), product))
        return self.carts.pop(cart_id, None)

from threading import Thread
import time

class Producer(Thread):
    """A producer thread that continuously publishes products."""
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the producer and registers it with the marketplace."""
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        Main loop: continuously produces and publishes its assigned products,
        retrying with a wait period upon failure.
        """
        while True:
            for (product, quantity, wait_time) in self.products:
                while quantity:
                    out = self.marketplace.publish(self.producer_id, product)
                    if not out:
                        time.sleep(self.republish_wait_time)
                    else:
                        quantity -= 1
                        time.sleep(wait_time)
