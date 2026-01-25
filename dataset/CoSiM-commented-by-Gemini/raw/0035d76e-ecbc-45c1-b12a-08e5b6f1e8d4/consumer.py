"""
This module implements a producer-consumer simulation centered around a marketplace.
It defines Producer and Consumer threads that interact with a shared Marketplace
to publish and purchase products. The simulation uses locks to ensure thread-safe
operations on shared data structures.
"""


from threading import Thread
import time


class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace to buy products.
    Each consumer has a list of operations (add or remove items from a cart)
    and places an order at the end.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping lists, where each list contains actions.
            marketplace (Marketplace): The shared marketplace object.
            retry_wait_time (float): The time to wait before retrying to add a product.
            **kwargs: Additional arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.market = marketplace
        self.wait_time = retry_wait_time
        self.carts = carts
        

    def run(self):
        """
        The main execution logic for the consumer. It processes a list of actions
        (adding/removing products from a cart) and finally places an order.
        """
        cart = self.market.new_cart()
        
        # Iterates through all assigned shopping actions.
        for cart_list in self.carts:
            for act in cart_list:
                
                
                # Pre-condition: 'act' specifies an action ('add' or 'remove') on a product.
                if act["type"] == "add":
                    cantitate = act["quantity"]
                    while cantitate > 0 :
                        # Attempts to add a product to the cart. If unavailable, it waits and retries.
                        ok = self.market.add_to_cart(cart, act["product"])
                        if ok:
                            cantitate -= 1
                        else :
                            
                            time.sleep(self.wait_time)
                else:
                    
                    
                    # Removes a specified quantity of a product from the cart.
                    cantitate = act["quantity"]
                    for i in range(cantitate):
                        self.market.remove_from_cart(cart, act["product"])
        # Places the final order with the contents of the cart.
        comanda = self.market.place_order(cart)
        
        for product in comanda :
            print(self.name + " bought " + str(product))



from threading import Lock


class Marketplace:
    """
    A thread-safe marketplace that facilitates the exchange of products between
    producers and consumers. It manages product inventory and shopping carts.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in the marketplace at once.
        """
        self.consumer_lock = Lock()
        self.buffer = []
        self.carts = []
        self.producer_lock = Lock()
        self.producer_id = -1


        self.cart_id = -1
        self.queue_size = queue_size_per_producer


    def register_producer(self):
        """
        Registers a new producer, assigning it a unique ID and a dedicated buffer space.
        Returns:
            int: The ID of the newly registered producer.
        """
        self.producer_lock.acquire()
        self.producer_id += 1
        self.buffer.append([])
        
        self.producer_lock.release()
        return self.producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace from a specific producer.
        The operation is thread-safe and respects the producer's queue size limit.
        """
        self.producer_lock.acquire()
        
        # Checks if the producer's buffer is full.
        if len(self.buffer[producer_id]) >= self.queue_size :
            self.producer_lock.release()
            return False
        to_add = {
            'product' : product,
            'producer_id' : producer_id
        }
        self.buffer[producer_id].append(to_add)
        self.producer_lock.release()
        return True

    def new_cart(self):
        """
        Creates a new shopping cart for a consumer, assigning a unique cart ID.
        """
        self.consumer_lock.acquire()
        self.cart_id += 1
        self.carts.append([])
        self.consumer_lock.release()
        return self.cart_id


    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's cart. It searches for the product across all
        producers' buffers.
        """
        self.consumer_lock.acquire()
        
        # Searches for the requested product in the marketplace buffer.
        for i in range(len(self.buffer)):
            for product_aux in self.buffer[i]:
                if product == product_aux['product']:
                    
                    # Moves the product from the buffer to the cart.
                    self.buffer[i].remove(product_aux)
                    self.carts[cart_id].append(product_aux)
                    self.consumer_lock.release()
                    return True
        self.consumer_lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a consumer's cart and returns it to the
        original producer's buffer.
        """
        
        
        self.consumer_lock.acquire()
        for product_aux in self.carts[cart_id]:
            if product_aux['product'] == product: 
                
                # Returns the product to the producer's buffer.
                self.buffer[product_aux['producer_id']].append(product_aux)
                self.carts[cart_id].remove(product_aux)
                break
        self.consumer_lock.release()
       



    def place_order(self, cart_id):
        """
        Finalizes an order, returning the list of products in the cart.
        """
        order = []
        for prod in self.carts[cart_id] :
            order.append(prod['product'])
        return order
        


from threading import Thread
import time

from httplib2 import ProxiesUnavailableError


class Producer(Thread):
    """
    Represents a producer that creates products and publishes them to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products to be produced.
            marketplace (Marketplace): The shared marketplace object.
            republish_wait_time (float): The time to wait before retrying to publish.
            **kwargs: Additional arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.market = marketplace
        self.wait_time = republish_wait_time
        self.products = products
        
        self.id = self.market.register_producer()

    def run(self):
        """
        The main execution logic for the producer. It continuously produces and
        publishes products to the marketplace.
        """
        contor = 0
        
        time.sleep(self.products[contor][2])
        while True:
            # Block Logic: Continuously publishes products based on the provided list.
            for product in self.products:
                for _ in range(product[1]):
                    # If publishing fails (e.g., buffer is full), it waits and retries.
                    while not self.market.publish(self.id, product[0]):
                        time.sleep(self.wait_time)
                    
                    time.sleep(product[2])



from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A simple data class for a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A data class representing Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A data class representing Coffee, inheriting from Product."""
    acidity: str
    roast_level: str