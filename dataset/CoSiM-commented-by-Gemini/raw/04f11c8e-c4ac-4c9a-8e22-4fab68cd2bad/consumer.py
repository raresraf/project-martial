
"""
A simulation of a marketplace with producers and consumers.

This module implements a multi-threaded producer-consumer model to simulate
an e-commerce marketplace. It includes three main classes:
- Marketplace: The central entity that manages product inventories and carts.
- Producer: A thread that adds products to the marketplace.
- Consumer: A thread that adds products to a cart and "buys" them.

The simulation demonstrates handling concurrent operations, such as multiple
producers publishing products and multiple consumers shopping simultaneously,
using threading and semaphores for synchronization.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    Represents a consumer that buys products from the marketplace.

    Each consumer runs in its own thread, processing a list of shopping operations
    (add/remove products) for a series of carts, and then places an order for each.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        Args:
            carts (list): A list of carts, where each cart is a list of product
                          operations (add/remove dictionaries).
            marketplace (Marketplace): The marketplace instance from which to buy.
            retry_wait_time (float): The time in seconds to wait before retrying
                                     to add a product if it's not available.
            **kwargs: Arbitrary keyword arguments, expects 'name'.
        """
        super().__init__()
        self.name = kwargs["name"]
        self.retry_wait_time = retry_wait_time
        self.id_cart = -1  
        self.carts = carts
        self.marketplace = marketplace

    def run(self):
        """
        The main execution loop for the consumer thread.

        Iterates through each assigned cart, performs the add/remove operations,
        and finally places the order. It implements a retry mechanism with a
        timeout for adding products that are not immediately available.
        """
        for cart in self.carts:
            # Logic: A new cart is created in the marketplace for each set of transactions.
            self.id_cart = self.marketplace.new_cart()
            for command in cart:
                comm_type = command["type"]


                product = command["product"]
                quantity = command["quantity"]

                if comm_type == "add":
                    # Block Logic: Attempts to add the specified quantity of a product to the cart.
                    for i in range(quantity):
                        add_result = self.marketplace.add_to_cart(self.id_cart, product)
                        # Invariant: Continuously retries adding the product until successful.
                        while True:
                            
                            if not add_result:


                                time.sleep(self.retry_wait_time)
                                add_result = self.marketplace.add_to_cart(self.id_cart, product)
                            else:
                                
                                break
                elif comm_type == "remove":
                    # Block Logic: Removes a specified quantity of a product from the cart.
                    for i in range(quantity):
                        
                        remove_result = self.marketplace.remove_from_cart(self.id_cart, product)
                        if not remove_result:  
                            print("INVALID OPERATION RESULT! REMOVED FAILED! EXITING THREAD")
                            return
                else:  
                    print("INVALID OUTPUT! EXITING THREAD")
                    return
            # Functional Utility: Finalizes the purchase and prints the bought items.
            cart_list = self.marketplace.place_order(self.id_cart)
            for item in cart_list:
                print(f"{self.name} bought {item}")


import threading


class Marketplace(object):
    """
    Manages producers, product inventories, and customer carts in a thread-safe manner.

    This class acts as the central hub for the simulation, using semaphores to
    coordinate access to shared resources like product queues and shopping carts,
    preventing race conditions between concurrent producers and consumers.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products each
                                           producer can have in their inventory at a time.
        """
        self.queues = {}  # Stores producer inventories and their semaphores.
                                                                
        self.capacity = queue_size_per_producer
        self.id_producer = -1
        self.id_cart = -1
        self.general_semaphore = threading.Semaphore(1)  # Protects shared marketplace data structures.
                                                         
        self.carts = {}  # Stores active shopping carts.

    def register_producer(self):
        """
        Registers a new producer, assigning them a unique ID and an inventory queue.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        self.general_semaphore.acquire()


        self.id_producer += 1
        self.queues[self.id_producer] = {}
        self.queues[self.id_producer]["products"] = []
        self.queues[self.id_producer]["semaphore"] = threading.Semaphore(1)
        self.general_semaphore.release()
        return self.id_producer

    def publish(self, producer_id, product):
        """
        Allows a producer to add a product to their inventory.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (any): The product to be added to the inventory.

        Returns:
            bool: True if the product was successfully published, False if the
                  producer's inventory is full.
        """


        self.queues[producer_id]["semaphore"].acquire()
        if len(self.queues[producer_id]["products"]) < self.capacity:
            
            self.queues[producer_id]["products"].append(product)
            self.queues[producer_id]["semaphore"].release()
            return True
        self.queues[producer_id]["semaphore"].release()
        
        return False

    def new_cart(self):
        """
        Creates a new, empty shopping cart.

        Returns:
            int: The unique ID for the newly created cart.
        """
        self.id_cart += 1
        self.carts[self.id_cart] = []  # A cart stores tuples of (producer_id, product).
        return self.id_cart

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified shopping cart by taking it from a producer's inventory.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (any): The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """

        # Block Logic: Iterates through all producer queues to find the requested product.
        for id_producer, queue_producer in self.queues.items():
            queue_producer["semaphore"].acquire()
            for queue_product in queue_producer["products"]:
                if product == queue_product:
                    
                    # Logic: Moves the product from the producer's queue to the consumer's cart.
                    queue_producer["products"].remove(queue_product)
                    
                    self.carts[cart_id].append((id_producer, product))
                    queue_producer["semaphore"].release()
                    return True
            queue_producer["semaphore"].release()

        
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart and returns it to the producer's inventory.

        Args:
            cart_id (int): The ID of the cart to remove the product from.
            product (any): The product to remove.

        Returns:
            bool: True if the product was found and removed, False otherwise.
        """
        # Block Logic: Iterates through the items in the cart to find the product.
        for id_producer, cart_product in self.carts[cart_id]:
            if product == cart_product:
                self.carts[cart_id].remove((id_producer, cart_product))
                # Logic: Returns the product to the original producer's inventory.
                self.queues[id_producer]["semaphore"].acquire()
                self.queues[id_producer]["products"].append(product)
                self.queues[id_producer]["semaphore"].release()
                return True
        return False

    def place_order(self, cart_id):
        """
        Finalizes an order, returning the list of products in the cart.

        Args:
            cart_id (int): The ID of the cart to place an order for.

        Returns:
            list: A list of products that were in the cart.
        """
        result = []
        for _, product in self.carts[cart_id]:
            result.append(product)

        return result


from threading import Thread
import time


class Producer(Thread):
    """
    Represents a producer that supplies products to the marketplace.

    Each producer runs in its own thread, continuously attempting to publish a
    list of products to the marketplace, each with a specific quantity and
    production time.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer instance.

        Args:
            products (list): A list of products to produce, where each product
                             is a tuple of (product_name, quantity, production_time).
            marketplace (Marketplace): The marketplace instance to publish to.
            republish_wait_time (float): The time in seconds to wait before
                                         retrying to publish if the inventory is full.
            **kwargs: Arbitrary keyword arguments, expects 'name'.
        """
        super().__init__(daemon=True)
        self.products = products
        self.marketplace = marketplace
        self.name = kwargs["name"]
        self.republish_wait_time = republish_wait_time
        self.id_producer = -1  # Will be assigned by the marketplace.
                               

    def run(self):
        """
        The main execution loop for the producer thread.

        Registers with the marketplace and then enters an infinite loop to produce
        and publish its assigned products. It respects the production time for
        each item and retries publishing if the marketplace inventory is full.
        """
        self.id_producer = self.marketplace.register_producer()

        # Invariant: The producer runs in an infinite loop to continuously supply products.
        while True:
            for product_struct in self.products:
                product = product_struct[0]  
                quantity = product_struct[1]  
                sleep_time = product_struct[2]  

                for _ in range(quantity):
                    publish_result = self.marketplace.publish(self.id_producer, product)

                    # Block Logic: If publishing fails (e.g., inventory is full),
                    # it retries after a specified wait time.
                    if not publish_result:
                        while True:
                            
                            time.sleep(self.republish_wait_time)
                            publish_result = self.marketplace.publish(self.id_producer, product)
                            if publish_result:
                                
                                time.sleep(sleep_time)
                                break
                    else:
                        time.sleep(sleep_time)
