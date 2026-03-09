"""
This module contains a producer-consumer simulation. It includes classes for
Consumer, Marketplace, Producer, and data classes for different product types.
The Marketplace implementation in this version is complex and has notable
inefficiencies and potential logic flaws in how it tracks product ownership.
"""
from threading import Thread, currentThread, Lock
import time
from dataclasses import dataclass


class Consumer(Thread):
    """
    A thread that simulates a consumer purchasing products from the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of "shopping trips", where each trip is a list
                          of operations (add/remove products).
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying a failed 'add'.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main logic for the consumer. Processes each shopping trip, adding and
        removing items from a cart, and then printing the purchased items.
        """
        # A consumer can perform multiple shopping trips.
        for cart in self.carts:
            # A new cart ID is created for each trip.
            id_cart = self.marketplace.new_cart()
            for operation in cart:
                op_count = 0
                # Invariant: Loop until the desired quantity for the operation is met.
                while op_count < operation['quantity']:
                    if operation['type'] == 'add':
                        # Pre-condition: Try to add a product. If it fails (product not
                        # available), wait and retry.
                        if self.marketplace.add_to_cart(id_cart, operation['product']) is False:
                            time.sleep(self.retry_wait_time)        
                        else:
                            op_count += 1
                    elif operation['type'] == 'remove':
                        self.marketplace.remove_from_cart(id_cart, operation['product'])
                        op_count += 1

            # Finalize the order and print the items bought in this trip.
            products_in_cart = self.marketplace.place_order(id_cart)
            for product in products_in_cart:                               
                print(currentThread().getName() + " bought " + str(product))

class Marketplace:
    """
    Manages producers, consumers, and product inventory.

    WARNING: This implementation is inefficient. Adding/removing items requires
    linearly scanning through producers' inventories. State is also duplicated
    across several data structures, making it hard to maintain.
    """
    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer
        self.all_carts = {} # Stores items for each cart ID.
        self.id_carts_lock = Lock()
        self.id_cart = -1
        self.id_producer = -1
        self.id_producer_lock = Lock()
        # The following three structures create redundant state.
        self.products_in_marketplace = [] # A flat list of all available products.
        self.producers_queues = {} # Tracks current queue size for each producer.
        self.producers_products = {} # Tracks the specific products published by each producer.
        self.add_remove_lock = Lock()

    def register_producer(self):
        """Registers a new producer and returns a unique ID."""
        self.id_producer_lock.acquire()
        self.id_producer += 1               
        self.id_producer_lock.release()

        self.producers_products[self.id_producer] = []     
        self.producers_queues[self.id_producer] = 0        

        return self.id_producer

    def publish(self, producer_id, product):
        """Allows a producer to publish a product to the marketplace."""
        # Pre-condition: Check if the producer's queue is full.
        if not self.producers_queues[int(producer_id)] < self.queue_size_per_producer:
            return False

        # Add the product to two different tracking structures.
        self.producers_queues[int(producer_id)] += 1
        self.products_in_marketplace.append(product)                   
        self.producers_products[int(producer_id)].append(product)

        return True

    def new_cart(self):
        """Creates a new, empty cart and returns its ID."""
        self.id_carts_lock.acquire()
        self.id_cart += 1                       
        self.id_carts_lock.release()
        self.all_carts[self.id_cart] = []
        return self.id_cart

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart.

        This method is highly inefficient as it requires a linear scan of all
        producers' products to find the item and update their queue count.
        """
        with self.add_remove_lock:
            if product not in self.products_in_marketplace:
                return False

            self.products_in_marketplace.remove(product)
            # Inefficient Search: Find which producer "owns" the product to decrement their count.
            for producer in self.producers_products:
                if product in self.producers_products[producer]:            
                    self.producers_queues[producer] -= 1                    
                    self.producers_products[producer].remove(product)       
                    break

        self.all_carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the marketplace."""
        self.all_carts[cart_id].remove(product)

        # The logic to return a product is also inefficient.
        with self.add_remove_lock:
            self.products_in_marketplace.append(product)
            # This just finds the first producer who *could* have made the product.
            for producer in self.producers_products:
                if product in self.producers_products[producer]:        
                    self.producers_queues[producer] += 1                
                    self.producers_products[producer].append(product)   
                    break

    def place_order(self, cart_id):
        """Returns the list of products in the specified cart."""
        return self.all_carts[cart_id]                             


class Producer(Thread):
    """A thread that simulates a producer creating and publishing products."""
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Register with the marketplace upon creation.
        self.producerID = self.marketplace.register_producer()

    def run(self):
        """
        Continuously produces items and attempts to publish them to the marketplace,
        waiting and retrying if the marketplace queue is full.
        """
        while True:
            for product in self.products:
                quantity = 0
                while quantity < product[1]:
                    # Pre-condition: Check if the product can be published.
                    if self.marketplace.publish(str(self.producerID), product[0]):
                        time.sleep(product[2])                     
                        quantity += 1                              
                    else:
                        # If the producer's queue is full, wait before retrying.
                        time.sleep(self.republish_wait_time)


# --- Data models for products ---
@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base class for a product with a name and price."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A tea product with an additional 'type' attribute."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A coffee product with acidity and roast level attributes."""
    acidity: str
    roast_level: str