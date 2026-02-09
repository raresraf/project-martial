"""
Module: consumer.py
Description: Semantic documentation for consumer.py.
             Detailed semantic analysis will be applied later.
"""


import time
from threading import Thread


class Consumer(Thread):
    

    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    '''
    Functional Utility: Describe purpose of run here.
    '''
    def run(self):
        
        


        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for cart in range(len(self.carts)):
            cart_id = self.marketplace.new_cart()
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for opr in range(len(self.carts[cart])):
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                if self.carts[cart][opr]['type'] == 'add':
                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    for _ in range(self.carts[cart][opr]['quantity']):
                        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                        # Invariant: State condition that holds true before and after each iteration/execution
                        while True:
                            aux = self.marketplace.add_to_cart(cart_id,
                                                               self.carts[cart][opr]['product'])
                            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                            # Invariant: State condition that holds true before and after each iteration/execution
                            if aux:
                                break


                            time.sleep(self.retry_wait_time)
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                elif self.carts[cart][opr]['type'] == 'remove':
                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    for _ in range(self.carts[cart][opr]['quantity']):
                        self.marketplace.remove_from_cart(cart_id,
                                                          self.carts[cart][opr]['product'])
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for product in self.marketplace.place_order(cart_id):
                print(self.name + " bought " + str(product))

from threading import Lock


class Marketplace:
    
    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.products = []  
        self.carts = []  
        self.lock_producer = Lock()  
        self.lock_cart = Lock()  
        self.lock_operations = Lock()  

    '''
    Functional Utility: Describe purpose of register_producer here.
    '''
    def register_producer(self):
        
        
        with self.lock_producer:
            self.products.append([])
        return len(self.products) - 1

    '''
    Functional Utility: Describe purpose of publish here.
    '''
    def publish(self, producer_id, product):
        
        
        with self.lock_operations:
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if len(self.products[producer_id]) < self.queue_size_per_producer:
                self.products[producer_id].append(product)
                return True
        return False

    '''
    Functional Utility: Describe purpose of new_cart here.
    '''
    def new_cart(self):
        
        
        with self.lock_cart:
            self.carts.append([])
        return len(self.carts) - 1

    '''
    Functional Utility: Describe purpose of add_to_cart here.
    '''
    def add_to_cart(self, cart_id, product):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for i in range(len(self.products)):
            
            
            with self.lock_operations:
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                if product in self.products[i]:
                    self.carts[cart_id].append((product, i))
                    self.products[i].remove(product)
                    return True
        return False

    '''
    Functional Utility: Describe purpose of remove_from_cart here.
    '''
    def remove_from_cart(self, cart_id, product):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for prod in self.carts[cart_id]:
            
            with self.lock_operations:
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                if prod[0] == product:
                    self.carts[cart_id].remove(prod)
                    self.products[prod[1]].append(prod[0])
                    break

    '''
    Functional Utility: Describe purpose of place_order here.
    '''
    def place_order(self, cart_id):
        
        
        result = []
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for product in self.carts[cart_id]:
            result.append(product[0])
        return result

import time
from threading import Thread


class Producer(Thread):
    

    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
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
            for product in range(len(self.products)):
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                for _ in range(self.products[product][1]):
                    aux = self.marketplace.publish(producer_id, self.products[product][0])
                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    if aux:
                        time.sleep(self.products[product][2])
                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    else:
                        time.sleep(self.republish_wait_time)
