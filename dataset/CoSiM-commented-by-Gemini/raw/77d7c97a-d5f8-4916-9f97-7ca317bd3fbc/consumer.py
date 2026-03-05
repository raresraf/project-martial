"""
This module implements a multi-producer, multi-consumer marketplace simulation.

This version defines product types using Python's `dataclasses` and attempts
to manage a marketplace where producers publish goods that consumers can then
add to a cart.

NOTE: This implementation is fundamentally broken due to severe concurrency issues.
The `Marketplace` class is not thread-safe. Critical methods like `publish`
operate on shared data without any locking, and other methods (`add_to_cart`,
`remove_from_cart`) use locks inconsistently and incorrectly. This will lead
to race conditions and data corruption in a multi-threaded environment.
"""

from threading import Thread, Lock, currentThread
import time
from dataclasses import dataclass


class Consumer(Thread):
    """
    Represents a consumer thread that processes shopping carts.
    """
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer.

        Args:
            carts (list): A list of shopping lists to process.
            marketplace (Marketplace): The shared marketplace object.
            retry_wait_time (float): Time to wait before retrying a failed action.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts 
        self.marketplace = marketplace 
        self.retry_wait_time = retry_wait_time

    def run(self):
        """Main execution loop for the consumer."""
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for products in cart:
                now_quantity = 0
                # Loop until the desired quantity of a product is acquired.
                while now_quantity < products["quantity"]:
                    check = False
                    if products["type"] == "add":
                        # Attempt to add the product to the cart.
                        check = self.marketplace.add_to_cart(cart_id, products["product"])
                    elif products["type"] == "remove":
                        check = self.marketplace.remove_from_cart(cart_id, products["product"])
                    
                    if check is False:
                        # If the operation failed, wait and retry.
                        time.sleep(self.retry_wait_time)
                    else:
                        now_quantity += 1
            # Once the cart is filled, place the order.
            self.marketplace.place_order(cart_id)


class Marketplace:
    """
    A centralized marketplace that is NOT thread-safe.

    This class manages producers and consumers but fails to use locks correctly
    to protect its shared data structures, making it prone to race conditions.
    """
    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer
        self.no_of_producers = 0
        self.producers = {} # Maps producer_id to their current inventory count.
        self.no_of_carts = 0
        self.carts = {}
        # Maps a product to a producer_id. This is a flawed model, implying
        # only one producer can ever make a given type of product.
        self.producers_products = {} 
        # A single global list of all products. This is a major bottleneck.
        self.available_products = [] 
        
        self.lock_reg_producers = Lock() 
        self.lock_carts = Lock() 
        self.lock_producers = Lock() 

    def register_producer(self):
        """Registers a new producer and returns a unique ID."""
        with self.lock_reg_producers:
            self.no_of_producers += 1
            producer_id = self.no_of_producers
            self.producers[producer_id] = 0
        return producer_id

    def publish(self, producer_id, product):
        """
        Adds a product to the marketplace.

        BUG: This method is NOT thread-safe. It modifies shared state
        (`self.producers`, `self.producers_products`, `self.available_products`)
        without acquiring any locks, which will lead to data corruption when
        called by multiple producer threads.
        """
        if self.producers[int(producer_id)] >= self.queue_size_per_producer:
            return False

        self.producers[int(producer_id)] += 1
        self.producers_products[product] = int(producer_id)
        self.available_products.append(product)
        return True

    def new_cart(self):
        """Creates a new cart and returns its ID."""
        with self.lock_carts:
            self.no_of_carts += 1
            cart_id = self.no_of_carts
            self.carts[cart_id] = []
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart.

        BUG: This method is NOT thread-safe. It races with the `publish` method
        (which has no lock) and also modifies `self.carts` outside of the
        `lock_carts` protection.
        """
        with self.lock_producers:
            if product not in self.available_products:
                return False

            prod_id = self.producers_products[product]
            self.producers[prod_id] -= 1
            self.available_products.remove(product)
            # This modification is not protected by the `lock_carts`.
            self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart. NOT thread-safe.
        """
        # Unsafe modification of the cart list.
        self.carts[cart_id].remove(product)
        # Unsafe modification of the global product list, though partially locked.
        self.available_products.append(product)
        with self.lock_producers:
            self.producers[self.producers_products[product]] += 1

    def place_order(self, cart_id):
        """Finalizes an order."""
        # The pop operation is not protected by `lock_carts`.
        prod_list = self.carts.pop(cart_id)
        for product in prod_list:
            print("{} bought {}".format(currentThread().getName(), product))


class Producer(Thread):
    """Represents a producer that creates and publishes products."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products 
        self.marketplace = marketplace 
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """Main producer loop."""
        while True:
            for sublist in self.products:
                count = 0
                while count < sublist[1]:
                    # `publish` is not thread-safe and will cause race conditions.
                    check = self.marketplace.publish(str(self.producer_id), sublist[0])
                    if check:
                        time.sleep(sublist[2])
                        count += 1
                    else:
                        time.sleep(self.republish_wait_time)


# --- Data Models for Products ---
@dataclass(frozen=True)
class Product:
    """A generic product with a name and price."""
    name: str
    price: int

@dataclass(frozen=True)
class Tea(Product):
    """A 'Tea' type of product."""
    type: str

@dataclass(frozen=True)
class Coffee(Product):
    """A 'Coffee' type of product."""
    acidity: str
    roast_level: str
