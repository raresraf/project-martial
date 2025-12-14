

import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                
                if operation["type"] == "add":


                    i = 0
                    while i < operation["quantity"]:
                        verify = self.marketplace.add_to_cart(cart_id, operation["product"])
                        while not verify:
                            
                            time.sleep(self.wait_time)
                            verify = self.marketplace.add_to_cart(cart_id, operation["product"])
                        i += 1
                
                elif operation["type"] == "remove":
                    i = 0
                    while i < operation["quantity"]:
                        self.marketplace.remove_from_cart(cart_id, operation["product"])
                        i += 1

            orders = self.marketplace.place_order(cart_id)
            for order in orders:
                
                print("%s bought %s" % (self.name, order[0]))

from threading import Lock


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size = queue_size_per_producer
        self.producer_id = 0
        self.cart_id = 0
        
        self.producers_buffers = {}
        
        self.carts_list = {}
        
        self.lock_buffers = Lock()
        
        self.lock_carts = Lock()

    def register_producer(self):
        
        with self.lock_buffers:
            self.producer_id += 1
            
            self.producers_buffers[self.producer_id] = []
            new_id = self.producer_id
        return new_id

    def publish(self, producer_id, product):
        
        self.lock_buffers.acquire()
        if len(self.producers_buffers[producer_id]) < self.queue_size:
            self.producers_buffers[producer_id].append(product)
            self.lock_buffers.release()
            return True

        self.lock_buffers.release()
        return False

    def new_cart(self):
        
        with self.lock_carts:
            self.cart_id += 1
            
            self.carts_list[self.cart_id] = []
            new_cart = self.cart_id
        return new_cart

    def add_to_cart(self, cart_id, product):
        
        self.lock_buffers.acquire()
        for producer in self.producers_buffers:
            for prod in self.producers_buffers[producer]:
                if prod == product:
                    
                    self.producers_buffers[producer].remove(prod)
                    
                    self.carts_list[cart_id].append((prod, producer))

                    self.lock_buffers.release()
                    return True

        self.lock_buffers.release()
        return False

    def remove_from_cart(self, cart_id, product):
        
        for (prod, producer) in self.carts_list[cart_id]:
            if prod == product:
                
                with self.lock_carts:
                    self.carts_list[cart_id].remove((prod, producer))
                
                self.producers_buffers[producer].append(prod)

                break

    def place_order(self, cart_id):
        
        return self.carts_list[cart_id]


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        
        Thread.__init__(self, group=None, target=None, name=None, args=(), kwargs={},
                        daemon=kwargs.get("daemon"))
        self.products = products
        self.name = kwargs["name"]
        self.marketplace = marketplace      
        self.wait_time = republish_wait_time


        self.id_producer = 0

    def run(self):
        
        self.id_producer = self.marketplace.register_producer()
        while True:
            for prod in self.products:
                i = 0


                while i < prod[1]:
                    verify = self.marketplace.publish(self.id_producer, prod[0])
                    while not verify:
                        
                        time.sleep(self.wait_time)
                        verify = self.marketplace.publish(self.id_producer, prod[0])
                    
                    time.sleep(prod[2])
                    i += 1
