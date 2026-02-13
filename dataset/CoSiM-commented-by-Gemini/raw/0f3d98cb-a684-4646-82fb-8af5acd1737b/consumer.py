"""
This module simulates a marketplace with producers and consumers using a multi-threaded approach.

It defines three main classes:
- Producer: A thread that generates and publishes products to the marketplace.
- Consumer: A thread that browses products, adds/removes them from a personal cart, and
  eventually places an order.
- Marketplace: The central class that is intended to synchronize producers and consumers.

**Warning:** This implementation contains severe threading issues. The locking mechanisms
are implemented incorrectly, which will lead to race conditions and unpredictable behavior
under concurrent access. The comments will highlight these specific issues.
"""

from threading import Thread, currentThread, Lock
import time

class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace.

    Each consumer runs in its own thread, performing a series of operations
    defined in its cart.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of actions for the consumer to perform.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying an operation.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution loop for the consumer.

        Processes each operation in the cart, retrying if a product is not
        immediately available.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for order in cart:
                i = 0
                if order["type"] == "add":
                    # Loop until the desired quantity of the product is added.
                    while i < order["quantity"]:
                        # Inner loop to retry adding a product until successful.
                        while True:
                            out = self.marketplace.add_to_cart(cart_id, order["product"])
                            if out == False:
                                time.sleep(self.retry_wait_time)
                            else:
                                break
                        i += 1
                if order["type"] == "remove":
                    while i < order["quantity"]:
                        self.marketplace.remove_from_cart(cart_id, order["product"])
                        i += 1
            self.marketplace.place_order(cart_id)


class Marketplace:
    """
    Manages the inventory and transactions between producers and consumers.

    @warning This class is NOT thread-safe. The lock implementation is incorrect,
    and most methods that modify shared state do so without any synchronization,
    making it highly susceptible to race conditions.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can list.
        """
        self.max_q_per_prod = queue_size_per_producer
        self.producer_id = 0
        # `products` stores products per producer. Shared state, needs locking.
        self.products = {}
        self.cart_id = 0
        # `carts` stores consumer cart contents. Shared state, needs locking.
        self.carts = {}
        # `marketplace` is a list of all available products. Shared state, needs locking.
        self.marketplace = []

    def register_producer(self):
        """
        Registers a new producer.

        @warning Incorrect Locking: A new `Lock` is created on each call, so it provides
        no protection for the shared `self.producer_id`. The `lock.release()` is also
        unreachable code. This method is not thread-safe.
        """
        lock = Lock()
        lock.acquire()
        self.producer_id += 1
        self.products[self.producer_id] = [] 
        return self.producer_id
        lock.release()

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace.

        @warning Not Thread-Safe: This method modifies the shared `self.marketplace`
        and `self.products` lists without any locks, which will cause race conditions.
        """
        num_prod = self.products[producer_id]
        if len(num_prod) >= self.max_q_per_prod:
            return False

        self.marketplace.append((product, producer_id))
        num_prod.append(product)
        return True

    def new_cart(self):
        """
        Creates a new cart for a consumer.

        @warning Incorrect Locking: A new `Lock` is created on each call, providing
        no protection for the shared `self.cart_id`. The `lock.release()` is
        unreachable. This method is not thread-safe.
        """
        lock = Lock()
        lock.acquire()
        self.cart_id += 1
        cart_id = self.cart_id
        self.carts[cart_id] = [] 
        return cart_id
        lock.release()

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's cart.

        @warning Not Thread-Safe: This method reads from and writes to shared lists
        (`self.marketplace`, `self.carts`, `self.products`) without any locking,
        making it extremely vulnerable to race conditions and data corruption.
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
        Removes a product from a consumer's cart.

        @warning Not Thread-Safe: Modifies shared lists (`self.carts`,
        `self.marketplace`, `self.products`) without any synchronization.
        """
        for (product_type, _producer_id) in self.carts[cart_id]:
            if product_type == product:
                self.carts[cart_id].remove((product, _producer_id))
                self.marketplace.append((product_type, _producer_id))
                self.products[_producer_id].append(product)
                break

    def place_order(self, cart_id):
        """
        Finalizes an order, printing the bought items.

        @warning Not Thread-Safe: Reading from `self.carts` and calling `pop`
        without a lock can lead to issues if another thread modifies it.
        """
        for (product, _producer_id) in self.carts[cart_id]:
            print("{} bought {}".format(currentThread().getName(), product))

        return self.carts.pop(cart_id, None)


class Producer(Thread):
    """
    Represents a producer that supplies products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products for the producer to publish.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """The main execution loop for the producer."""
        # This loop runs indefinitely, attempting to publish products.
        while True:
            for (product, quantity, wait_time) in self.products:
                # Publish the specified quantity of the product.
                while quantity:
                    out = self.marketplace.publish(self.producer_id, product)
                    if out == False:
                        # If publishing fails (e.g., queue is full), wait and retry.
                        time.sleep(self.republish_wait_time)
                    else:
                        quantity -= 1
                        time.sleep(wait_time)
