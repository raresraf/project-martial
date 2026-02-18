"""
Models a multi-producer, multi-consumer marketplace simulation.

This module implements a producer-consumer system using a global product pool
architecture. Unlike other versions with per-producer queues, this `Marketplace`
maintains a single list of all available products. The implementation attempts
fine-grained locking but suffers from numerous race conditions and logic errors,
making its concurrent operation fundamentally unsafe.
"""
from threading import Thread
import time

class Consumer(Thread):
    """
    A thread that simulates a consumer purchasing products from the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.
        
        Args:
            carts (list): A list of shopping actions to perform.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying an action.
            **kwargs: Keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        Executes the consumer's shopping simulation by processing a list of carts,
        with each cart containing a series of add/remove operations.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for ops in cart:
                ops_nr = 0

                ops_type = ops["type"]
                product = ops["product"]
                quantity = ops["quantity"]

                # Perform the 'add' or 'remove' operation for the specified quantity.
                while ops_nr < quantity:
                    if ops_type == "add":
                        operation = self.marketplace.add_to_cart(cart_id, product)
                    else:
                        operation = self.marketplace.remove_from_cart(cart_id, product)

                    # Retry logic if the operation failed (e.g., product not available).
                    # NOTE: `operation is None` is likely a logic error, as the methods
                    # do not appear to return None.
                    if operation or operation is None:
                        ops_nr += 1
                    else:
                        time.sleep(self.retry_wait_time)

            
            self.marketplace.place_order(cart_id)

from threading import Lock, currentThread

class Marketplace:
    """
    The central marketplace, acting as a broker between producers and consumers.
    
    This implementation uses a global pool of products rather than per-producer
    queues. It is not thread-safe due to improper locking.
    """
    def __init__(self, queue_size_per_producer):
        

        self.queue_size_per_producer = queue_size_per_producer

        self.carts = {} 
        self.product_producer = {} 
        self.products = [] 
        self.products_number = [] 

        self.lock_new = Lock() 
        self.lock_print = Lock() 
        self.lock_product = Lock() 
        self.lock_id = Lock() 

        self.carts_nr = 0

    def register_producer(self):
        """Thread-safely registers a new producer, returning a unique ID."""
        self.lock_id.acquire()
        producer_id = len(self.products_number)
        
        self.products_number.append(0)
        self.lock_id.release()

        return producer_id

    def publish(self, producer_id, product):
        """
        Adds a product to the global product pool.
        
        RACE CONDITION: This method is not thread-safe. Multiple producers
        calling this concurrently can lead to inconsistent state because access
        to shared lists (`self.products`, `self.products_number`) is not synchronized.
        """
        prod_id = int(producer_id)

        
        if self.products_number[prod_id] >= self.queue_size_per_producer:
            return False

        self.products_number[prod_id] += 1

        self.products.append(product)
        self.product_producer[product] = prod_id 

        return True

    def new_cart(self):
        """Thread-safely creates a new cart and returns its ID."""
        self.lock_new.acquire()
        self.carts_nr += 1
        cart_id = self.carts_nr
        self.lock_new.release()

        
        self.carts[cart_id] = []

        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Moves a product from the global pool to a consumer's cart.
        This operation is intended to be atomic but is flawed.
        """
        with self.lock_product:
            if product not in self.products:
                return False

            
            producer_id = self.product_producer[product]
            self.products_number[producer_id] -= 1
            self.products.remove(product)

        
        self.carts[cart_id].append(product)

        return True

    def remove_from_cart(self, cart_id, product):
        """
        Moves a product from a cart back to the global pool.
        
        RACE CONDITION: This method is not thread-safe. It modifies the shared
        `self.products` list without acquiring `self.lock_product`, which can
        corrupt the state if another thread calls `add_to_cart` simultaneously.
        """
        self.carts[cart_id].remove(product)
        self.products.append(product)

        self.lock_product.acquire()
        producer_id = self.product_producer[product]
        self.products_number[producer_id] += 1
        self.lock_product.release()


    def place_order(self, cart_id):
        """Finalizes an order by printing the items bought."""

        list_all = []
        for product in self.carts[cart_id]:
            # The lock ensures that printing is atomic per item.
            self.lock_print.acquire()


            print(str(currentThread().getName()) + " bought " + str(product))
            list_all.append(product)
            self.lock_print.release()

        return list_all


from threading import Thread
import time


class Producer(Thread):
    """A thread that simulates a producer creating and publishing products."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """Continuously produces items according to a schedule."""
        while True:
            for (product, quantity, product_wait_time) in self.products:
                i = 0

                while i < quantity:
                    # Retry publishing until successful.
                    pub = self.marketplace.publish(str(self.producer_id), product)

                    if pub is True:
                        time.sleep(product_wait_time)
                        i += 1
                    else:
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    
    acidity: str
    roast_level: str
