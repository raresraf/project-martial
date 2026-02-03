"""
This module simulates a producer-consumer marketplace using threads.

It defines `Producer` and `Consumer` classes that interact with a shared
`Marketplace`. Producers publish products, and consumers add them to carts and
place orders. The simulation uses locks to ensure thread-safe access to shared
resources.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace.

    Each consumer processes a list of carts, adding and removing products
    before placing an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a new Consumer instance."""
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        """
        The main execution logic for the consumer.

        It iterates through its assigned carts, performs add/remove operations,
        and then places an order.
        """
        
        # Invariant: Processes each cart assigned to the consumer.
        for cart in range(len(self.carts)):
            cart_id = self.marketplace.new_cart()
            for opr in range(len(self.carts[cart])):
                # Block-level comment: Handles 'add' operations by attempting
                # to add the specified quantity of a product to the cart,
                # retrying if the product is not immediately available.
                if self.carts[cart][opr]['type'] == 'add':
                    for _ in range(self.carts[cart][opr]['quantity']):
                        while True:
                            aux = self.marketplace.add_to_cart(cart_id,
                                                               self.carts[cart][opr]['product'])
                            if aux:
                                break


                            time.sleep(self.retry_wait_time)
                elif self.carts[cart][opr]['type'] == 'remove':
                    for _ in range(self.carts[cart][opr]['quantity']):
                        self.marketplace.remove_from_cart(cart_id,
                                                          self.carts[cart][opr]['product'])
            for product in self.marketplace.place_order(cart_id):
                print(self.name + " bought " + str(product))

from threading import Lock


class Marketplace:
    """
    Represents the shared marketplace where producers and consumers interact.

    It manages the inventory of products and the state of shopping carts. Access
    to shared data is protected by locks.
    """
    
    def __init__(self, queue_size_per_producer):
        """Initializes a new Marketplace instance."""
        self.queue_size_per_producer = queue_size_per_producer
        self.products = []  
        self.carts = []  
        self.lock_producer = Lock()  
        self.lock_cart = Lock()  
        self.lock_operations = Lock()  

    def register_producer(self):
        """
        Registers a new producer with the marketplace.
        """
        
        with self.lock_producer:
            self.products.append([])
        return len(self.products) - 1

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace.

        Returns True if the product was successfully published, False otherwise
        (e.g., if the producer's queue is full).
        """
        
        with self.lock_operations:
            if len(self.products[producer_id]) < self.queue_size_per_producer:
                self.products[producer_id].append(product)
                return True
        return False

    def new_cart(self):
        """
        Creates a new shopping cart.
        """
        
        with self.lock_cart:
            self.carts.append([])
        return len(self.carts) - 1

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a shopping cart.

        This involves finding the product in a producer's inventory and moving
        it to the cart.
        """
        # Invariant: Iterates through all producer inventories to find the
        # requested product.
        for i in range(len(self.products)):
            
            
            with self.lock_operations:
                if product in self.products[i]:
                    self.carts[cart_id].append((product, i))
                    self.products[i].remove(product)
                    return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart.

        This moves the product from the cart back to the producer's inventory.
        """
        for prod in self.carts[cart_id]:
            
            with self.lock_operations:
                if prod[0] == product:
                    self.carts[cart_id].remove(prod)
                    self.products[prod[1]].append(prod[0])
                    break

    def place_order(self, cart_id):
        """
        Finalizes an order, returning the list of products in the cart.
        """
        
        result = []
        for product in self.carts[cart_id]:
            result.append(product[0])
        return result

import time
from threading import Thread


class Producer(Thread):
    """
    Represents a producer that publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes a new Producer instance."""
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        The main execution logic for the producer.

        It continuously tries to publish its products to the marketplace,
        waiting if the marketplace is full.
        """
        producer_id = self.marketplace.register_producer()
        
        # Invariant: This loop continuously attempts to publish all products.
        while True:
            for product in range(len(self.products)):
                for _ in range(self.products[product][1]):
                    aux = self.marketplace.publish(producer_id, self.products[product][0])
                    if aux:
                        time.sleep(self.products[product][2])
                    else:
                        time.sleep(self.republish_wait_time)