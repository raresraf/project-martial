
"""
This module implements a producer-consumer simulation centered around a
Marketplace.

It defines `Producer` and `Consumer` threads that interact through the
`Marketplace` class, which manages product inventory and shopping carts.
This version uses instance-level state for the marketplace, but has some
flaws in its locking strategy.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    Represents a consumer thread that shops at the marketplace.
    
    Each consumer processes a list of shopping carts, adding and removing
    products according to a predefined list of operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.name = kwargs['name']
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """The main execution loop for the consumer."""
        for cart in self.carts:


            cart_id = self.marketplace.new_cart()

            
            for operation in cart:
                for _ in range(operation['quantity']):
                    if operation['type'] == 'add':
                        # If adding fails (product not found), wait and retry.
                        while not self.marketplace.add_to_cart(cart_id, operation['product']):
                            sleep(self.retry_wait_time)
                    elif operation['type'] == 'remove':
                        self.marketplace.remove_from_cart(cart_id, operation['product'])

            
            order = self.marketplace.place_order(cart_id)
            for product in order:
                print("%s bought %s" % (self.name, product))


from threading import Lock

class Marketplace:
    """
    Manages the inventory and transactions between producers and consumers.

    This implementation correctly uses instance variables for its state, meaning
    different Marketplace objects are independent. However, its locking is
    a mix of fine-grained and coarse-grained, with some operations lacking
    protection entirely.
    """
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.last_producer_id = 0
        self.last_cart_id = 0
        
        self.products_per_producer = {}
        
        self.carts = {}
        # A lock to protect the cart ID counter and carts dictionary.
        self.cart_lock = Lock()
        # A lock to protect the producer ID counter.
        self.producer_id_lock = Lock()
        # A single, coarse-grained lock for all add-to-cart operations.
        self.add_to_cart_lock = Lock()

    def register_producer(self):
        """Registers a new producer and returns a unique ID."""
        with self.producer_id_lock:
            self.last_producer_id += 1
            self.products_per_producer[self.last_producer_id] = []
            return self.last_producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product. This operation is not locked
        and assumes it is only called by a single producer thread at a time.
        """

        if len(self.products_per_producer[producer_id]) == self.queue_size_per_producer:
            return False

        self.products_per_producer[producer_id].append(product)
        return True

    def new_cart(self):
        """Creates a new, empty shopping cart for a consumer."""
        with self.cart_lock:
            self.last_cart_id += 1
            self.carts[self.last_cart_id] = []
            return self.last_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart by searching all producers.
        
        This method uses a single, coarse-grained lock, which serializes all
        `add_to_cart` operations across all consumers, potentially creating a
        bottleneck.
        """
        
        
        for producer_id, products in self.products_per_producer.items():


            with self.add_to_cart_lock:
                if product in products:
                    products.remove(product)
                    self.carts[cart_id].append((producer_id, product))
                    return True

        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the original producer.

        CRITICAL FLAW: This method is not thread-safe. It does not acquire
        any locks, creating a race condition. If multiple threads call this
        method concurrently, or if it runs at the same time as `add_to_cart`,
        the `self.carts` and `self.products_per_producer` collections could be
        corrupted.
        """
        
        
        producer_id = 0
        for cart_producer_id, cart_product in self.carts[cart_id]:
            if cart_product == product:
                producer_id = cart_producer_id

        
        self.carts[cart_id].remove((producer_id, product))
        self.products_per_producer[producer_id].append(product)

    def place_order(self, cart_id):
        """Finalizes an order by returning a list of products in the cart."""
        return [product for _, product in self.carts[cart_id]]


from threading import Thread
from time import sleep

class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = -1

    def run(self):
        """
        The main execution loop for the producer.
        
        It registers with the marketplace and then continuously publishes its
        products, waiting and retrying if a product queue is full.
        """
        self.producer_id = self.marketplace.register_producer()

        
        
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    while not self.marketplace.publish(self.producer_id, product[0]):
                        sleep(self.republish_wait_time)

                    sleep(product[2])
