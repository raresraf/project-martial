"""
This module simulates a marketplace with producers, consumers, and various
product types using a multi-threaded approach.

It defines classes for:
- Consumer: A thread that simulates a customer adding and removing items from a
  shopping cart and placing an order.
- Marketplace: The central class intended to manage the inventory and cart
  transactions in a shared environment.
- Producer: A thread that simulates a producer publishing products to the marketplace.
- Product, Tea, Coffee: Dataclasses for representing products.

Warning: The Marketplace class is not thread-safe. While some methods use a
mutex, several critical methods that modify shared state do so without any
locking, which will lead to race conditions and incorrect behavior.
"""

from threading import Thread, Lock
from time import sleep
from dataclasses import dataclass

class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace.

    Each consumer runs in its own thread, performing a series of "add" or "remove"
    operations before placing a final order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of cart operations for the consumer to perform.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying an operation.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """The main execution loop for the consumer."""
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            
            for command in cart:
                command_type = command["type"]
                product = command["product"]
                quantity = command["quantity"]
                
                if command_type == "add":
                    # Tries to add the specified quantity of a product to the cart.
                    for _ in range(quantity):
                        # Retries until the product is successfully added.
                        while True:
                            added = self.marketplace.add_to_cart(cart_id, product)
                            if added:
                                break
                            sleep(self.retry_wait_time)
                elif command_type == "remove":
                    # Tries to remove the specified quantity of a product.
                    for _ in range(quantity):
                        while True:
                            removed = self.marketplace.remove_from_cart(cart_id, product)
                            if removed:
                                break
                            sleep(self.retry_wait_time)
            
            # Places the order and prints the items bought.
            products_bought = self.marketplace.place_order(cart_id)
            for product in products_bought:
                print(f"{self.getName()} bought {product}")


class Marketplace:
    """
    Manages inventory and transactions between producers and consumers.

    @warning Not Thread-Safe: This class has severe race conditions. While a
    mutex is used in `register_producer` and `new_cart`, it is NOT used in
    `publish`, `add_to_cart`, or `remove_from_cart`, which all modify shared
    lists and dictionaries concurrently.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): Max items a producer can have listed.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0
        self.cart_id = 0 
        # `queues`: Stores products for each producer. This is a shared resource.
        self.queues = [] 
        # `carts`: Stores products in each consumer's cart. This is a shared resource.
        self.carts = [] 
        # A single mutex to protect some ID generation.
        self.mutex = Lock() 
        # `products_dict`: Maps a product to its producer. This is a shared resource.
        self.products_dict = {} 

    def register_producer(self):
        """
        Registers a new producer, returning a unique ID.
        This method is partially thread-safe for ID generation.
        """
        self.mutex.acquire() 
        producer_id = self.producer_id
        self.producer_id += 1 
        self.queues.append([]) 
        self.mutex.release()
        return str(producer_id)

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace.

        @warning Not Thread-Safe: This method modifies `self.queues` and
        `self.products_dict` without a lock, leading to race conditions.
        """
        producer_index = int(producer_id) 
        if len(self.queues[producer_index]) == self.queue_size_per_producer: 
            return False
        self.queues[producer_index].append(product) 
        self.products_dict[product] = producer_index 
        return True

    def new_cart(self):
        """
        Creates a new cart for a consumer.

        @note While cart ID generation is protected by a mutex, the modification
        of the `self.carts` list happens outside the lock, which is not ideal.
        """
        self.mutex.acquire()
        cart_id = self.cart_id
        self.cart_id += 1
        self.mutex.release()
        self.carts.append([]) 
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified consumer's cart.

        @warning Not Thread-Safe: This method checks for a product's existence
        and then removes it from a shared list (`self.queues`) without a lock.
        This is a classic check-then-act race condition. Two threads could find
        the same product, and one would fail with a `ValueError` on `remove`.
        """
        is_product = False
        for queue in self.queues:
            if product in queue:
                is_product = True
                queue.remove(product) 
                break
        if not is_product:
            return False
        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the producer's stock.
        
        @warning Not Thread-Safe: This method modifies shared lists (`self.carts`,
        `self.queues`) without any synchronization.
        """
        if product not in self.carts[cart_id]:
            return False
        producer_idx = self.products_dict[product] 
        if len(self.queues[producer_idx]) == self.queue_size_per_producer: 
            return False
        self.carts[cart_id].remove(product) 
        self.queues[producer_idx].append(product) 
        return True

    def place_order(self, cart_id):
        """
        Finalizes the transaction for a given cart.

        @warning Not Thread-Safe: Reads from and modifies `self.carts` without a lock.
        """
        cart_content = self.carts[cart_id] 
        self.carts[cart_id] = [] 
        return cart_content 


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
        while True:
            for product in self.products:
                quantity = product[1]
                production_time = product[2]
                
                for _ in range(0, quantity):
                    # Retries publishing until successful.
                    while True:
                        published = self.marketplace.publish(self.producer_id, product[0])
                        if published:
                            sleep(production_time)
                            break
                        sleep(self.republish_wait_time)


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A dataclass for representing a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass for representing a Tea product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass for representing a Coffee product."""
    acidity: str
    roast_level: str
