"""
Models a multi-producer, multi-consumer marketplace simulation.

This module implements a system with Producers that publish products and Consumers
that purchase them. The `Marketplace` class acts as the central shared resource,
managing product inventory from multiple producers and consumer shopping carts. The
implementation uses threading to simulate concurrent producers and consumers.

NOTE: This implementation contains several significant logical bugs, particularly
in the inventory management within the `Marketplace` class, which would prevent
it from functioning correctly.
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
            retry_wait_time (float): Time in seconds to wait before retrying to add a product.
            **kwargs: Keyword arguments for the Thread constructor (e.g., name).
        """
        Thread.__init__(self, name=kwargs['name'])
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        

    def run(self):
        """
        Executes the consumer's shopping simulation.
        
        For each cart of actions, it adds/removes products and finally places the order.
        """
        for cart in self.carts:
            my_cart_id = self.marketplace.new_cart()
            i = 0
            # Process each action in the cart (add/remove).
            while i < len(cart):
                prod = cart[i]['product']
                quantity = cart[i]['quantity']
                if cart[i]['type'] == 'add':
                    # Attempt to add the specified quantity of a product to the cart.
                    while quantity != 0:
                        # Retry adding to cart until successful.
                        while not self.marketplace.add_to_cart(my_cart_id, prod):
                            time.sleep(self.retry_wait_time)
                        quantity = quantity - 1
                else: # 'remove' action
                    while quantity != 0:
                        self.marketplace.remove_from_cart(my_cart_id, prod)
                        quantity = quantity - 1
                i = i + 1
            # Finalize the transaction.
            self.marketplace.place_order(my_cart_id)


from threading import RLock
import threading

class Marketplace:
    """
    The central marketplace that manages producers, products, and consumer carts.
    
    This class is the shared state between all producer and consumer threads and is
    responsible for coordinating all actions. It contains several critical bugs.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.
        
        Args:
            queue_size_per_producer (int): The maximum number of products each
                                           producer can have listed at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.queues = []
        self.nr_producers = 0
        # NOTE: Romanian variable names are used, e.g., 'nr_produse_producator'
        # means 'number_of_products_per_producer'.
        self.nr_produse_producator = {}
        self.lock_per_queue = []
        self.carts = {}
        self.nr_carts = 0

    def register_producer(self):
        """
        Registers a new producer, providing them with a unique ID.
        
        Returns:
            int: The new producer's ID.
        """
        self.lock_per_queue.append(RLock())
        self.queues.append([])
        self.nr_produse_producator[self.nr_producers] = 0
        self.nr_producers = self.nr_producers + 1
        return self.nr_producers - 1

    def publish(self, producer_id, product):
        """
        Allows a producer to list a new product.
        
        BUG: The check `> queue_size_per_producer` should likely be `>=`. As written,
        it allows a producer to publish one more item than the specified limit.

        Returns:
            bool: True if the product was published successfully, False if the
                  producer's queue is full.
        """
        if self.nr_produse_producator[producer_id] > self.queue_size_per_producer:
            return False


        self.lock_per_queue[producer_id].acquire()
        self.queues[producer_id].append(product)
        self.nr_produse_producator[producer_id] = self.nr_produse_producator[producer_id] + 1
        self.lock_per_queue[producer_id].release()
        return True

    def new_cart(self):
        """Creates a new, empty shopping cart and returns its ID."""
        self.carts[self.nr_carts] = []
        self.nr_carts = self.nr_carts + 1
        return self.nr_carts - 1

    def add_to_cart(self, cart_id, product):
        """
        Finds a product from any producer and adds it to a cart.
        
        BUG: The line `self.queues[i] = self.queues[:i] + self.queues[i+1:]`
        is incorrect. It doesn't remove the product from the producer's queue (`self.queues[i]`)
        but instead corrupts the list of queues itself (`self.queues`). The correct logic
        to remove the j-th product from the i-th queue would be `self.queues[i].pop(j)`.
        This bug breaks the entire inventory system.
        """
        found = False
        for i in range(0, self.nr_producers):
            self.lock_per_queue[i].acquire()
            for j in range(0, len(self.queues[i])):
                if self.queues[i][j] == product:
                    # This line contains a critical bug that corrupts the `queues` list.
                    self.queues[i] = self.queues[:i] + self.queues[i+1:]
                    # BUG: The product count for the producer is not decremented here,
                    # but rather in `place_order`, which is incorrect.
                    self.nr_produse_producator[i] = self.nr_produse_producator[i] - 1
                    self.carts[cart_id].append((i, product))
                    found = True
                    break
            self.lock_per_queue[i].release()
            if found:
                break
        return found

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the producer's inventory."""
        for i in range(0, len(self.carts[cart_id])):
            (nr_prod, prod) = self.carts[cart_id][i]
            if prod == product:
                self.lock_per_queue[nr_prod].acquire()
                self.queues[nr_prod].append(product)
                self.nr_produse_producator[nr_prod] = self.nr_produse_producator[nr_prod] + 1
                self.lock_per_queue[nr_prod].release()
                self.carts[cart_id] = self.carts[cart_id][:i] + self.carts[cart_id][i+1:]
                break

    def place_order(self, cart_id):
        """
        Finalizes an order, 'consuming' the products.
        
        NOTE: The producer's product count is incorrectly decremented here. It should be
        decremented when an item is successfully added to a cart to accurately reflect
        availability. As is, products in carts still count against a producer's publish limit.
        """
        for item in self.carts[cart_id]:
            (nr_prod, prod) = item
            self.nr_produse_producator[nr_prod] = self.nr_produse_producator[nr_prod] - 1
            print(threading.currentThread().getName(), "bought", prod)


from threading import Thread
import time

class Producer(Thread):
    """
    A thread that simulates a producer creating and publishing products.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, name=kwargs['name'], daemon=kwargs['daemon'])
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        

    def run(self):
        """Continuously produces items according to a schedule."""
        my_id = self.marketplace.register_producer()
        while True:
            for product in self.products:
                (prod, quantity, duration) = product
                while quantity != 0:
                    time.sleep(duration)
                    # Retry publishing until successful.
                    while not self.marketplace.publish(my_id, prod):
                        time.sleep(self.republish_wait_time)
                    quantity = quantity - 1


from dataclasses import dataclass


# Data classes for defining product types.
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
