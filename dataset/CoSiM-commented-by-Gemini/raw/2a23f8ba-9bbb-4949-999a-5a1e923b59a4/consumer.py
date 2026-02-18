"""
Models a multi-producer, multi-consumer marketplace simulation.

This module implements a system where Producer threads create products and add
them to per-producer queues within a central `Marketplace`. Consumer threads
then request products from the marketplace, which are fulfilled from any
available producer.

CRITICAL NOTE: This implementation is not thread-safe. The core `publish` and
`add_to_cart` methods do not use locks when modifying shared data structures,
leading to severe race conditions that would corrupt the marketplace state under
concurrent access.
"""
from threading import Thread
import time

class Consumer(Thread):
    """
    A thread that simulates a consumer ordering products from the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        """
        Processes a list of shopping carts, where each cart is a series of
        add/remove operations, and then places the order.
        """
        for check in self.carts:
            
            id_cart = self.marketplace.new_cart()
            
            for demand in check:
                if demand["type"] == "add":


                    i = 0
                    
                    while i < (demand["quantity"]):
                        # Attempt to add a product to the cart, retrying on failure.
                        add_cart = self.marketplace.add_to_cart(id_cart, demand["product"])
                        while not add_cart:
                            time.sleep(self.retry_wait_time)
                            add_cart = self.marketplace.add_to_cart(id_cart, demand["product"])
                        i = i + 1


                elif demand["type"] == "remove":
                    i = 0
                    while i < (demand["quantity"]):
                        self.marketplace.remove_from_cart(id_cart, demand["product"])
                        i = i + 1
            # Finalize the order and print the items bought.
            place_order = self.marketplace.place_order(id_cart)

            for product in place_order:
                print(self.name + " bought " + str(product))

from threading import Lock
class Marketplace:
    """
    The central marketplace that manages producers and consumer carts.

    This class is intended to be the thread-safe broker, but it has critical
    concurrency flaws due to missing locks in key methods.
    """
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer

        self.id_producer = 0 
        self.id_cart = 0 

        # Inventory is a dictionary mapping producer IDs to their list of products.
        self.producers_dictionary = {}  
        self.carts_dictionary = {}  

        self.nr_prod = []   
        # A single coarse-grained mutex intended to protect shared state.
        self.mutex = Lock()

    def register_producer(self):
        """
        Registers a new producer and initializes their product list.
        Uses a lock to safely generate a new producer ID.
        """
        with self.mutex:
            
            self.id_producer = len(self.nr_prod)
            self.producers_dictionary[self.id_producer] = []
        
        self.nr_prod.append(0)
        return self.id_producer

    def publish(self, producer_id, product):
        """
        Adds a product to a specific producer's inventory.

        RACE CONDITION: This method is NOT thread-safe. It modifies the shared
        `producers_dictionary` and `nr_prod` list without acquiring a lock,
        which will lead to data corruption if called by multiple producers.
        """
        if self.nr_prod[producer_id] >= self.queue_size_per_producer:
            return False
        
        self.producers_dictionary[producer_id].append(product)
        self.nr_prod[producer_id] += 1
        return True

    def new_cart(self):
        """Thread-safely creates a new cart and returns its ID."""
        with self.mutex:
            
            self.id_cart = self.id_cart + 1
            self.carts_dictionary[self.id_cart] = []
        return self.id_cart

    def add_to_cart(self, cart_id, product):
        """
        Searches all producer inventories for a product and moves it to a cart.

        RACE CONDITION: This method is NOT thread-safe. It reads and modifies
        the shared `producers_dictionary` and `nr_prod` list without any locking,
        creating race conditions between consumers and with producers.
        """
        for prod_count in self.producers_dictionary:
            if product in self.producers_dictionary[prod_count]:
                self.carts_dictionary[cart_id].append((product, prod_count))
                self.producers_dictionary[prod_count].remove(product)
                
                if self.nr_prod[prod_count]:
                    self.nr_prod[prod_count] -= 1
                return True

        return False


    def remove_from_cart(self, cart_id, product):
        """Moves a product from a cart back to the corresponding producer's inventory."""
        try:
            # This lock protects the modification of the cart and producer dictionaries.
            with self.mutex:
                for cart in self.carts_dictionary[cart_id]:
                    if product == cart[0]:
                        self.carts_dictionary[cart_id].remove(cart)
                        self.producers_dictionary[cart[1]].append(product)
                        self.nr_prod[cart[1]] += 1
                        return

        except KeyboardInterrupt:
            # This exception handling is unusual and likely not part of the core logic.
            print('Caught KeyboardInterrupt')

    def place_order(self, cart_id):
        """Finalizes an order by creating a list of bought products and deleting the cart."""
        cart_dic = self.carts_dictionary[cart_id]
        products_ordered = []
        
        for cart in cart_dic:
            products_ordered.append(cart[0])
        # The cart is removed from the marketplace after the order is placed.
        self.carts_dictionary.pop(cart_id)

        return products_ordered


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
            for prod in self.products:
                i = 0
                


                while i < (prod[1]):
                    # Retry publishing until successful.
                    add_prod = self.marketplace.publish(self.producer_id, prod[0])

                    
                    while not add_prod:
                        time.sleep(self.republish_wait_time)
                        add_prod = self.marketplace.publish(self.producer_id, prod[0])

                    time.sleep(prod[2])
                    i = i + 1
