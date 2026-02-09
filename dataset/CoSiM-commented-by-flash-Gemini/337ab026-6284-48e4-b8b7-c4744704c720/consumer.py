"""
Module: consumer.py
Description: Semantic documentation for consumer.py.
             Detailed semantic analysis will be applied later.
"""



from threading import Thread
import time

class Consumer(Thread):
    

    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    '''
    Functional Utility: Describe purpose of run here.
    '''
    def run(self):
        
        
        
        
        
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for operation in cart:
                count = 0
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                while count < operation["quantity"]:
                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    if operation["type"] == "add":
                        result = self.marketplace.add_to_cart(cart_id, operation["product"])
                        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                        # Invariant: State condition that holds true before and after each iteration/execution
                        if result:
                            count += 1
                        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                        # Invariant: State condition that holds true before and after each iteration/execution
                        else:
                            time.sleep(self.retry_wait_time)
                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    elif operation["type"] == "remove":
                        self.marketplace.remove_from_cart(cart_id, operation["product"])
                        count += 1
            self.marketplace.place_order(cart_id)>>>> file: marketplace.py

from threading import BoundedSemaphore, currentThread


class Marketplace:
    

    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        
        self.producer_ids = -1
        
        self.cart_ids = -1
        
        self.products = []
        
        self.producers_capacity = {}
        
        self.producers = {}
        
        self.carts = {}
        
        self.semaphore = BoundedSemaphore(1)

    '''
    Functional Utility: Describe purpose of register_producer here.
    '''
    def register_producer(self):
        
        
        self.semaphore.acquire()
        self.producer_ids = self.producer_ids + 1
        self.semaphore.release()

        
        
        self.producers_capacity[self.producer_ids] = 0
        return self.producer_ids

    '''
    Functional Utility: Describe purpose of publish here.
    '''
    def publish(self, producer_id, product):
        
        
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if self.producers_capacity[int(producer_id)] < self.queue_size_per_producer:
            self.producers_capacity[int(producer_id)] += 1
            
            self.products.append(product)
            self.producers[product] = int(producer_id)
            return True
        return False

    '''
    Functional Utility: Describe purpose of new_cart here.
    '''
    def new_cart(self):
        
        
        self.semaphore.acquire()
        self.cart_ids = self.cart_ids + 1
        self.semaphore.release()
        self.carts[self.cart_ids] = []
        return self.cart_ids

    '''
    Functional Utility: Describe purpose of add_to_cart here.
    '''
    def add_to_cart(self, cart_id, product):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if product in self.products:
            self.semaphore.acquire()


            
            
            self.producers_capacity[self.producers[product]] -= 1
            self.semaphore.release()
            
            self.carts[cart_id].append(product)
            return True
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        else:
            return False

    '''
    Functional Utility: Describe purpose of remove_from_cart here.
    '''
    def remove_from_cart(self, cart_id, product):
        
        
        self.carts[cart_id].remove(product)


        self.products.append(product)
        
        
        self.semaphore.acquire()
        self.producers_capacity[self.producers[product]] += 1
        self.semaphore.release()

    '''
    Functional Utility: Describe purpose of place_order here.
    '''
    def place_order(self, cart_id):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for product in self.carts[cart_id]:
            self.semaphore.acquire()
            print(F"{currentThread().getName()} bought {product}")
            self.semaphore.release()

        return self.carts[cart_id]


from threading import Thread
import time

class Producer(Thread):
    

    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    '''
    Functional Utility: Describe purpose of run here.
    '''
    def run(self):
        
        
        producer_id = str(self.marketplace.register_producer())
        
        
        
        
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        while True:
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for product in self.products:
                count = product[1]
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                while count > 0:
                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    if self.marketplace.publish(producer_id, product[0]):
                        time.sleep(product[2])
                        count -= 1
                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    else:
                        time.sleep(self.republish_wait_time)>>>> file: product.py


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