"""
Module: consumer.py
Description: Semantic documentation for consumer.py.
             Detailed semantic analysis will be applied later.
"""



from threading import Thread
from time import sleep


class Consumer(Thread):
    

    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

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
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                if operation.get("type") == "add":
                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    for quantity in range(operation.get("quantity")):
                        status = self.marketplace.add_to_cart(cart_id, operation.get("product"))
                        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                        # Invariant: State condition that holds true before and after each iteration/execution
                        while not status:
                            sleep(self.retry_wait_time)
                            status = self.marketplace.add_to_cart(cart_id, operation.get("product"))
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                elif operation.get("type") == "remove":
                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    for quantity in range(operation.get("quantity")):
                        self.marketplace.remove_from_cart(cart_id, operation.get("product"))
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for product in self.marketplace.place_order(cart_id):
                print(self.kwargs.get("name") + " bought " + product.__str__())>>>> file: marketplace.py

from threading import Lock, Thread

class Marketplace:
    
    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_index = -1
        self.consumer_cart_index = -1
        self.producer_queue_size_list = []
        self.products_list = []
        self.producers_to_products_map = {}
        self.carts = []
        self.lock_register = Lock()
        self.lock_publish = Lock()
        self.lock_new_cart = Lock()
        self.lock_add_cart = Lock()
        self.lock_remove_cart = Lock()

    '''
    Functional Utility: Describe purpose of register_producer here.
    '''
    def register_producer(self):
        
        self.lock_register.acquire()
        self.producer_index += 1
        self.lock_register.release()
        self.producer_queue_size_list.append(0)
        return self.producer_index


    '''
    Functional Utility: Describe purpose of publish here.
    '''
    def publish(self, producer_id, product):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if self.producer_queue_size_list[producer_id] <= self.queue_size_per_producer:
            self.producer_queue_size_list[producer_id] += 1
            self.products_list.append(product)
            self.lock_publish.acquire()
            self.producers_to_products_map[product] = producer_id
            self.lock_publish.release()
            return True
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        else:
            return False

    '''
    Functional Utility: Describe purpose of new_cart here.
    '''
    def new_cart(self):
        
        self.lock_new_cart.acquire()
        self.consumer_cart_index += 1
        self.lock_new_cart.release()
        self.carts.append([])
        return self.consumer_cart_index

    '''
    Functional Utility: Describe purpose of add_to_cart here.
    '''
    def add_to_cart(self, cart_id, product):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if product in self.products_list:
            self.products_list.remove(product)


            self.lock_add_cart.acquire()
            self.producer_queue_size_list[self.producers_to_products_map[product]] -= 1
            self.lock_add_cart.release()
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
        
        self.products_list.append(product)


        self.carts[cart_id].remove(product)
        self.lock_remove_cart.acquire()
        self.producer_queue_size_list[self.producers_to_products_map[product]] += 1
        self.lock_remove_cart.release()

    '''
    Functional Utility: Describe purpose of place_order here.
    '''
    def place_order(self, cart_id):
        
        return self.carts[cart_id]


from threading import Thread
from time import sleep


class Producer(Thread):
    

    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    '''
    Functional Utility: Describe purpose of run here.
    '''
    def run(self):
        producer_id = self.marketplace.register_producer()
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        while True:
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for (product, quantity, wait_time) in self.products:
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                while quantity > 0:
                    status = self.marketplace.publish(producer_id, product)
                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    while not status:
                        sleep(self.republish_wait_time)
                        status = self.marketplace.publish(producer_id, product)
                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    if status:
                        quantity -= 1
                        sleep(wait_time)



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
