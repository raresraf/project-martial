

import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        
        


        for cart in range(len(self.carts)):
            cart_id = self.marketplace.new_cart()
            for opr in range(len(self.carts[cart])):
                if self.carts[cart][opr]['type'] == 'add':
                    for _ in range(self.carts[cart][opr]['quantity']):
                        while True:
                            aux = self.marketplace.add_to_cart(cart_id,
                                                               self.carts[cart][opr]['product'])
                            if aux:
                                break


                            time.sleep(self.retry_wait_time)
                elif self.carts[cart][opr]['type'] == 'remove':
                    for _ in range(self.carts[cart][opr]['quantity']):
                        self.marketplace.remove_from_cart(cart_id,
                                                          self.carts[cart][opr]['product'])
            for product in self.marketplace.place_order(cart_id):
                print(self.name + " bought " + str(product))

from threading import Lock


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.products = []  
        self.carts = []  
        self.lock_producer = Lock()  
        self.lock_cart = Lock()  
        self.lock_operations = Lock()  

    def register_producer(self):
        
        
        with self.lock_producer:
            self.products.append([])
        return len(self.products) - 1

    def publish(self, producer_id, product):
        
        
        with self.lock_operations:
            if len(self.products[producer_id]) < self.queue_size_per_producer:
                self.products[producer_id].append(product)
                return True
        return False

    def new_cart(self):
        
        
        with self.lock_cart:
            self.carts.append([])
        return len(self.carts) - 1

    def add_to_cart(self, cart_id, product):
        
        for i in range(len(self.products)):
            
            
            with self.lock_operations:
                if product in self.products[i]:
                    self.carts[cart_id].append((product, i))
                    self.products[i].remove(product)
                    return True
        return False

    def remove_from_cart(self, cart_id, product):
        
        for prod in self.carts[cart_id]:
            
            with self.lock_operations:
                if prod[0] == product:
                    self.carts[cart_id].remove(prod)
                    self.products[prod[1]].append(prod[0])
                    break

    def place_order(self, cart_id):
        
        
        result = []
        for product in self.carts[cart_id]:
            result.append(product[0])
        return result

import time
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        


        producer_id = self.marketplace.register_producer()
        
        while True:
            for product in range(len(self.products)):
                for _ in range(self.products[product][1]):
                    aux = self.marketplace.publish(producer_id, self.products[product][0])
                    if aux:
                        time.sleep(self.products[product][2])
                    else:
                        time.sleep(self.republish_wait_time)
