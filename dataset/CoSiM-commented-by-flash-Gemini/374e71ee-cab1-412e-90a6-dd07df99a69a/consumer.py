

from threading import Thread, Lock
import time

class Consumer(Thread):
    """
    Represents a consumer in a marketplace simulation.

    Each consumer operates as a separate thread, creating a shopping cart,
    adding and removing products based on a list of operations, and finally
    placing an order. It handles retries for adding products to the cart.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a new Consumer instance.

        Args:
            carts (list): A list of lists of product operations (add/remove)
                          that this consumer will perform.
            marketplace (Marketplace): The marketplace object with which
                                       this consumer will interact.
            retry_wait_time (float): The time in seconds to wait before
                                     retrying a failed 'add to cart' operation.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self._carts = carts
        self._market = marketplace
        self.retry_time = retry_wait_time
        self._lock = Lock() # A lock to synchronize printing output (optional, but good practice).

    def run(self):
        """
        The main execution method for the consumer thread.

        It creates a new shopping cart in the marketplace, then processes
        a series of product operations. For 'add' operations, it repeatedly
        tries to add the product, waiting if the operation fails. For 'remove'
        operations, it simply removes the product. After processing all
        operations, it places the order and prints the purchased items.
        """
        cart_id = self._market.new_cart()
        for op_list in self._carts:

            # Iterate through each operation in the current list of operations.
            for op in op_list:

                op_type = op["type"]
                prod = op["product"]
                quantity = op["quantity"]

                if op_type == "add":
                    # Continuously try to add the product until the desired quantity is met.
                    while quantity > 0:
                        ret = self._market.add_to_cart(cart_id, prod)

                        if ret == True:
                            # If successful, decrement the remaining quantity.
                            quantity -= 1
                        else:
                            # If failed (e.g., product not available), wait and retry.
                            time.sleep(self.retry_time)

                if op_type == "remove":
                    # Remove the product from the cart the specified number of times.
                    while quantity > 0:
                        self._market.remove_from_cart(cart_id, prod)
                        quantity -= 1

            # Acquire the lock before placing the order and printing to prevent mixed output.
            with self._lock:
                # Place the order and get the list of products bought.
                products_list = self._market.place_order(cart_id)
                # Print each product that was successfully bought by this consumer.
                for prod in products_list:
                    print("cons" + str(cart_id) + " bought " + str(prod))

from threading import Lock

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self._producers_queue = {}  
        self._carts = {}            
        self._id_carts = 0          


        self._id_producers = 0      
        self._products = []         
        self._product_producer = {} 
        self._queue_size = queue_size_per_producer
        self._lock0 = Lock()
        self._lock1 = Lock()
        self._lock2 = Lock()

    def register_producer(self):
        
        with self._lock0:
            self._id_producers += 1

        return self._id_producers

    def publish(self, producer_id, product):
        
        
        if producer_id not in self._producers_queue:
            self._producers_queue[producer_id] = 0

        
        if self._producers_queue[producer_id] >= self._queue_size:
            return False

        
        self._producers_queue[producer_id] += 1

        
        self._products.append(product)

        
        self._product_producer[product] = producer_id

        return True

    def new_cart(self):
        
        with self._lock1:
            self._id_carts += 1

        return self._id_carts

    def add_to_cart(self, cart_id, product):
        

        
        
        with self._lock2:
            if cart_id not in self._carts:
                self._carts[cart_id] = []
            
            


            if product not in self._products:
                return False
            
            
            self._products.remove(product)

            
            pid = self._product_producer[product]
            self._producers_queue[pid] -= 1

            
            self._carts[cart_id].append(product)
        
        return True


    def remove_from_cart(self, cart_id, product):
        
        
        self._carts[cart_id].remove(product)

        


        pid = self._product_producer[product]
        self._producers_queue[pid] += 1

        
        self._products.append(product)

    def place_order(self, cart_id):
                
        cart_prods_copy = self._carts[cart_id].copy()
        self._carts[cart_id] = []
        
        return cart_prods_copy

import time
from threading import Thread

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self._prods = products
        self._market = marketplace
        self._id = marketplace.register_producer()
        self._rwait_time = republish_wait_time

    def run(self):
        while True:

            
            for product in self._prods:
                prod = product[0]
                quantity = product[1]
                repub_time = product[2]

                
                while quantity > 0:
                    ret = self._market.publish(self._id, prod)
                    if ret is True:
                        time.sleep(self._rwait_time)
                        quantity -= 1
                    else:
                        time.sleep(repub_time)


from dataclasses import dataclass


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