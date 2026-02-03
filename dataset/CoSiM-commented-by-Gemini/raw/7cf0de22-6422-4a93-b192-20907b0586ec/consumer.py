"""
This module simulates a producer-consumer marketplace using threads.

It defines `Producer` and `Consumer` classes that interact with a shared
`Marketplace`. Producers publish products, and consumers add them to carts and
place orders. The simulation uses locks to ensure thread-safe access to shared
resources.
"""

from threading import Thread
import time


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
    
    
    
    
    
    def run(self):
        """
        The main execution logic for the consumer.

        It iterates through its assigned carts, performs add/remove operations,
        and then places an order.
        """
        # Invariant: Processes each cart assigned to the consumer.
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for action in cart:
                for _ in range(action["quantity"]):
                    # Block-level comment: Handles 'add' operations by
                    # attempting to add a product to the cart, retrying with a
                    # delay if the product is not immediately available.
                    if action["type"] == "add":
                        ret = self.marketplace.add_to_cart(cart_id, action["product"])
                        while ret is False:
                            time.sleep(self.retry_wait_time)
                            ret = self.marketplace.add_to_cart(cart_id, action["product"])
                    elif action["type"] == "remove":
                        self.marketplace.remove_from_cart(cart_id, action["product"])
            self.marketplace.place_order(cart_id)

from threading import Lock
from threading import currentThread

class Marketplace:
    """
    Represents the shared marketplace where producers and consumers interact.

    It manages the inventory of products and the state of shopping carts. Access
    to shared data is protected by locks.
    """
    
    def __init__(self, queue_size_per_producer):
        """Initializes a new Marketplace instance."""
        self.queue_size_per_producer = queue_size_per_producer
        self.nr_of_producers = 0 
        self.nr_carts = 0 
        self.carts = [] 
        self.products = [] 
        self.map_between_product_and_id = {} 
        self.register = Lock() 
        self.publ = Lock() 
        self.for_cart = Lock() 
        self.for_action = Lock() 

    def register_producer(self):
        """
        Registers a new producer with the marketplace.
        """
        
        with self.register:
            self.nr_of_producers += 1

        
        self.products.append([])
        return self.nr_of_producers-1

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace.

        Returns True if the product was successfully published, False otherwise
        (e.g., if the producer's queue is full).
        """
        with self.publ:

            if len(self.products[producer_id]) >= self.queue_size_per_producer:
                return False
        
        self.products[producer_id].append(product)
        self.map_between_product_and_id[product] = producer_id
        return True
    def new_cart(self):
        """
        Creates a new shopping cart.
        """
        with self.for_cart:
            self.nr_carts += 1

        
        self.carts.append([])

        return self.nr_carts-1

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a shopping cart.

        This involves finding the product in a producer's inventory and moving
        it to the cart.
        """
        
        with self.for_action:
            for lst in self.products:
                if product in lst:


                    if product in self.products[self.map_between_product_and_id[product]]:
                        self.products[self.map_between_product_and_id[product]].remove(product)
                        self.carts[cart_id].append(product)
                        return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart.

        This moves the product from the cart back to the producer's inventory.
        """
        if product in self.carts[cart_id]:
            self.carts[cart_id].remove(product)
            self.products[self.map_between_product_and_id[product]].append(product)

    def place_order(self, cart_id):
        """
        Finalizes an order, printing the products bought by the consumer.
        """
        
        lst = self.carts[cart_id]
        for prod in lst:
            print("{} bought {}".format(currentThread().getName(), prod))

        return lst


from threading import Thread
import time


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
        self.prod_id = self.marketplace.register_producer()
    
    
    
    def run(self):
        """
        The main execution logic for the producer.

        It continuously tries to publish its products to the marketplace,
        waiting and retrying if the marketplace is full.
        """
        # Invariant: This loop continuously attempts to publish all products.
        while 1:
            for prod in self.products:
                for _ in range(prod[1]):
                    if self.marketplace.publish(self.prod_id, prod[0]):
                        time.sleep(prod[2])
                    else:
                        while self.marketplace.publish(self.prod_id, prod[0]) is False:
                            time.sleep(self.republish_wait_time)