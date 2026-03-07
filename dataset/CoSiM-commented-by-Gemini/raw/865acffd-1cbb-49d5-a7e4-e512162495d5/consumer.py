"""
This module simulates a multi-threaded producer-consumer marketplace.

It defines classes for a Marketplace, Producers who publish products, and
Consumers who buy them. The simulation uses threads to run producers and
consumers concurrently, and locks to manage access to shared marketplace data.
The system models basic e-commerce operations like adding items to a cart,
placing an order, and managing product inventory.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    A thread representing a consumer that interacts with the marketplace.

    Each consumer processes a list of shopping carts, where each cart contains
    a sequence of actions (add/remove products).
    """
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of carts to be processed. Each cart is a list
                          of product operations.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying a
                                     failed marketplace operation.
            **kwargs: Keyword arguments passed to the Thread constructor,
                      such as 'name'.
        """

        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs
        

    def run(self):
        """
        The main execution loop for the consumer.

        Processes each assigned cart by executing its operations, and finally
        places the order.
        """

        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for operation in cart:
                type_opp = operation["type"]
                quantity = operation["quantity"]
                product = operation["product"]

                # Retry loop to fulfill the required quantity for the operation.
                while quantity > 0:
                    if type_opp == "add":
                        result = self.marketplace.add_to_cart(cart_id, product)
                    elif type_opp == "remove":
                        result = self.marketplace.remove_from_cart(cart_id, product)

                    if result or result is None:
                        quantity -= 1
                    else:
                        # If the operation failed (e.g., product not available), wait and retry.
                        time.sleep(self.retry_wait_time)

            order = self.marketplace.place_order(cart_id)
            for product in order:
                result = self.kwargs["name"] + " bought " + str(product)
                print(result)

from threading import Lock
from collections import defaultdict

class Marketplace:
    """
    The central marketplace that manages producers, products, and carts.

    This class contains the shared state and the logic for all marketplace
    operations. It uses locks to handle concurrent access from multiple
    producer and consumer threads.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products any
                                           single producer can have listed at one time.
        """

        
        self.queue_size_per_producer = queue_size_per_producer

        
        self.number_of_producers = 0

        
        self.producers_queue_sizes = dict()

        
        self.products = []

        
        self.number_of_carts = 0

        
        self.carts = defaultdict()

        
        self.lock_register = Lock()

        
        self.lock_cart = Lock()

        
        self.lock_product = Lock()

    def register_producer(self):
        """
        Registers a new producer, assigning it a unique ID.

        Returns:
            int: The new producer's unique ID.
        """

        
        with self.lock_register:
            producer_id = self.number_of_producers
            self.number_of_producers += 1
            self.producers_queue_sizes[producer_id] = 0
        return producer_id


    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        The operation will fail if the producer has already reached its
        publication limit (`queue_size_per_producer`).

        Args:
            producer_id (str): The ID of the producer publishing the product.
            product (Product): The product to be published.

        Returns:
            bool: True if publication was successful, False otherwise.
        """
        
        id_producer = int(producer_id)

        
        product.owner = id_producer

        if self.producers_queue_sizes[id_producer] < self.queue_size_per_producer:
            
            self.producers_queue_sizes[id_producer] += 1
            
            self.products.append(product)
            return True

        return False

    def new_cart(self):
        """
        Creates a new, empty shopping cart and returns its ID.

        Returns:
            int: The new cart's unique ID.
        """
        
        with self.lock_cart:
            id_cart = self.number_of_carts
            self.number_of_carts += 1
            self.carts[id_cart] = []
        return id_cart


    def add_to_cart(self, cart_id, product):
        """
        Adds a product from the marketplace to a consumer's cart.

        This operation is atomic. It removes the product from the main
        product list and adds it to the specified cart.

        Args:
            cart_id (int): The ID of the cart to add to.
            product (Product): The product to add.

        Returns:
            bool: True if the product was successfully added, False if the
                  product was not available.
        """
        
        with self.lock_product:
            if product not in self.products:
                self.lock_product.release()
                return False
            
            producer_id = product.owner


            self.producers_queue_sizes[producer_id] -= 1

            
            self.products.remove(product)

            
            self.carts[cart_id].append(product)
        return True


    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the marketplace.

        NOTE: This method appears to lack locking, which could lead to race
        conditions if multiple threads access `self.products` or
        `self.producers_queue_sizes` concurrently.
        """
        
        
        
        self.carts[cart_id].remove(product)
        self.products.append(product)

        
        producer_id = product.owner
        self.producers_queue_sizes[producer_id] += 1



    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.

        Returns the list of products that were in the cart and clears the cart.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: A list of products in the order.
        """
        
        
        prod_list = self.carts[cart_id]
        
        self.carts[cart_id] = []

        return prod_list


from threading import Thread
import time

class Producer(Thread):
    """A thread representing a producer that publishes products to the marketplace."""
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products for the producer to publish.
                             Each item is a tuple of (product, quantity, production_time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish
                                         if the marketplace queue is full.
            **kwargs: Keyword arguments for the Thread constructor.
        """

        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

        

    def run(self):
        """
        The main execution loop for the producer.

        Continuously attempts to publish its products to the marketplace.
        """
        
        producer_id = self.marketplace.register_producer()
        while True:
            for product in self.products:
                type_prod = product[0]
                quantity = product[1]
                wait_time = product[2]

                while quantity > 0:
                    ret = self.marketplace.publish(str(producer_id), type_prod)

                    if ret:
                        time.sleep(wait_time)
                        quantity -= 1
                    else:
                        # If publishing failed, wait before retrying.
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=False)
class Product:
    """A base dataclass for a generic product."""
    
    name: str
    price: int
    owner = -1


@dataclass(init=True, repr=True, order=False, frozen=False)
class Tea(Product):
    """A dataclass for a Tea product, inheriting from Product."""
    
    type: str
    owner = -1

@dataclass(init=True, repr=True, order=False, frozen=False)
class Coffee(Product):
    """A dataclass for a Coffee product, inheriting from Product."""
    
    acidity: str
    roast_level: str
    owner = -1
