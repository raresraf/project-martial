"""
@file consumer.py
@brief A multithreaded producer-consumer simulation of an e-commerce marketplace.

This script models a marketplace where multiple producer threads generate products
and multiple consumer threads purchase them. It demonstrates the use of threading,
locks, and shared data structures to manage concurrent access to resources
in a simulated e-commerce environment.

Algorithm:
- The `Marketplace` class acts as a central hub, managing producers, consumers,
  products, and carts. It uses locks to ensure thread-safe operations.
- `Producer` threads generate and publish products to the marketplace.
- `Consumer` threads create carts, add/remove items, and place orders.
- The simulation uses `time.sleep` to model processing delays and waiting times.

Domain-Specific Awareness:
- **Concurrency**: The script heavily relies on `threading.Lock` to prevent race
  conditions when accessing shared data like product availability, producer capacity,
  and shopping carts. Each operation that modifies shared state is protected by a lock.
- **Producer-Consumer Pattern**: This is a classic example of the producer-consumer
  problem, where producers and consumers operate asynchronously and share a common,
  fixed-size buffer (or in this case, a marketplace with limited capacity).
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    Represents a consumer thread that simulates purchasing products from the marketplace.

    Each consumer processes a list of carts, with each cart containing a sequence
    of add/remove operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        Args:
            carts: A list of carts, where each cart is a list of items to be
                   added or removed.
            marketplace: The shared Marketplace instance.
            retry_wait_time: The time to wait before retrying an operation that
                             failed (e.g., adding an out-of-stock item).
            **kwargs: Keyword arguments for the Thread constructor, including 'name'.
        """
        Thread.__init__(self, **kwargs)


        self.name = kwargs["name"]
        self.carts = carts
        self.marketplace = marketplace
        self.wait_time = retry_wait_time
        

    def run(self):
        """
        The main execution loop for the consumer thread.

        Iterates through its assigned carts, processes the items in each cart,
        and places the order.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for item in cart:
                if item["type"] == "add":
                    for _ in range(item["quantity"]):
                        # Block Logic: Continuously tries to add an item to the cart until successful.
                        # This simulates waiting for a product to become available.
                        while not self.marketplace.add_to_cart(cart_id, item["product"]):
                            time.sleep(self.wait_time)

                elif item["type"] == "remove":
                    for _ in range(item["quantity"]):
                        # Block Logic: Continuously tries to remove an item from the cart until successful.
                        while not self.marketplace.remove_from_cart(cart_id, item["product"]):
                            time.sleep(self.wait_time)

            final_cart = self.marketplace.place_order(cart_id)
            for item in final_cart:
                if item is not None:
                    print(self.name + " bought " + str(item))
    


from threading import Lock

class Marketplace:
    """
    Manages the state of the marketplace, including producers, products, and carts.

    This class acts as a thread-safe intermediary between producers and consumers.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace instance.

        Args:
            queue_size_per_producer: The number of products each producer can
                                     publish before needing to wait.
        """
        self.queue_size = queue_size_per_producer
        self.producer_count = 0
        self.cart_count = 0
        self.carts = {}
        self.cart_suppliers = {}
        self.producer_lock_univ = Lock()
        self.producer_lock = []
        self.cart_lock = Lock()
        self.producer_capacity = []
        self.item_lock = {}
        self.product_availability = {}
        self.product_suppliers = {}
        

    def register_producer(self):
        """
        Registers a new producer with the marketplace.

        Returns:
            The ID of the newly registered producer.
        """
        
        with self.producer_lock_univ:
            self.producer_capacity.append(self.queue_size)
            retval = self.producer_count
            self.producer_count += 1
            self.producer_lock += [Lock()]
            return retval

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer to the marketplace.

        Args:
            producer_id: The ID of the producer publishing the product.
            product: The product to be published.

        Returns:
            True if the product was successfully published, False otherwise
            (e.g., if the producer's queue is full).
        """
        
        with self.producer_lock[producer_id]:
            if self.producer_capacity[producer_id] > 0:
                amount = self.product_availability.setdefault(product[0], 0)
                producers = self.product_suppliers.setdefault(product[0], [])
                self.product_suppliers.update({product[0]: producers + [producer_id]})

                self.product_availability.update({product[0]: 1 + amount})
                self.producer_capacity[producer_id] -= 1
                return True

        return False

    def new_cart(self):
        """
        Creates a new shopping cart.

        Returns:
            The ID of the new cart.
        """
        
        with self.cart_lock:
            retval = self.cart_count
            self.carts.setdefault(self.cart_count, [])
            self.cart_suppliers.setdefault(self.cart_count, [])
            self.cart_count += 1
            return retval

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a shopping cart.

        Args:
            cart_id: The ID of the cart to add the product to.
            product: The product to add.

        Returns:
            True if the product was successfully added, False otherwise
            (e.g., if the product is out of stock).
        """
        
        lock = self.item_lock.setdefault(product, Lock())

        with lock:
            amount = self.product_availability.setdefault(product, 0)

            if amount == 0:
                return False

            producers = self.product_suppliers.get(product)
            if producers is not None:
                self.producer_capacity[producers[0]] += 1
                self.cart_suppliers[cart_id].append(producers[0])
                producers.pop(0)


            self.product_availability.update({product: amount - 1})
            self.carts[cart_id].append(product)
            return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart.

        Args:
            cart_id: The ID of the cart to remove the product from.
            product: The product to remove.

        Returns:
            True if the product was successfully removed, False otherwise.
        """
        
        lock = self.item_lock.setdefault(product, Lock())

        with lock:
            amount = self.product_availability.setdefault(product, 0)
            producers = self.product_suppliers.setdefault(product, [])

            product_idx = self.carts[cart_id].index(product)
            producer_id = self.cart_suppliers[cart_id][product_idx]
            with self.producer_lock[producer_id]:
                self.product_suppliers.update({product: producers + [producer_id]})
                self.producer_capacity[producer_id] -= 1
                self.product_availability.update({product: amount + 1})
                self.cart_suppliers[cart_id][product_idx] = None
                self.carts[cart_id][product_idx] = None
                return True

        return False



    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.

        Args:
            cart_id: The ID of the cart to place the order for.

        Returns:
            The list of products in the cart.
        """
        
        return self.carts[cart_id]


import time
from threading import Thread


class Producer(Thread):
    """
    Represents a producer thread that generates and publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer instance.

        Args:
            products: A list of products that this producer can generate.
            marketplace: The shared Marketplace instance.
            republish_wait_time: The time to wait before retrying to publish a product.
            **kwargs: Keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.wait_time = republish_wait_time
        

    def run(self):
        """
        The main execution loop for the producer thread.

        Continuously publishes products to the marketplace.
        """
        this_id = self.marketplace.register_producer()

        while True:
            for item in self.products:
                for _ in range(item[1]):
                    # Block Logic: Continuously tries to publish a product until successful.
                    # This simulates waiting for capacity to become available.
                    while not self.marketplace.publish(this_id, item):
                        time.sleep(self.wait_time)

                    time.sleep(item[2])
        


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    A data class representing a generic product with a name and price.
    """
    
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    A data class representing a type of tea, inheriting from Product.
    """
    
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    A data class representing a type of coffee, inheriting from Product.
    """
    
    acidity: str
    roast_level: str