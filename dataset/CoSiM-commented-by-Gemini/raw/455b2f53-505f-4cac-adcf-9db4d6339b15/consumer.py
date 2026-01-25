


import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        


        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.operations = {"add": marketplace.add_to_cart,
                           "remove": marketplace.remove_from_cart}

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for operation in cart:
                quantity = operation["quantity"]

                while quantity > 0:
                    operation_type = operation["type"]
                    product = operation["product"]

                    if self.operations[operation_type](cart_id, product) is not False:
                        quantity -= 1
                    else:
                        time.sleep(self.retry_wait_time)

            self.marketplace.place_order(cart_id)

import sys
from threading import Lock, currentThread


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.carts_lock = Lock()
        self.carts = []

        self.producers_lock = Lock()
        self.producers_capacity = queue_size_per_producer
        self.producers_sizes = []
        self.products = []

    def register_producer(self):
        
        with self.producers_lock:
            self.producers_sizes.append(0)
            return len(self.producers_sizes) - 1

    def publish(self, producer_id, product):
        
        with self.producers_lock:
            if self.producers_sizes[producer_id] == self.producers_capacity:
                return False

            self.producers_sizes[producer_id] += 1
            self.products.append((product, producer_id))
            return True

    def new_cart(self):
        
        with self.carts_lock:
            self.carts.append([])
            return len(self.carts) - 1

    def add_to_cart(self, cart_id, product):
        


        self.producers_lock.acquire()
        for (prod, prod_id) in self.products:
            if prod == product:
                self.producers_sizes[prod_id] -= 1
                self.products.remove((prod, prod_id))
                self.producers_lock.release()
                self.carts[cart_id].append((prod, prod_id))
                return True

        self.producers_lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        


        for (prod, prod_id) in self.carts[cart_id]:
            if prod == product:
                self.carts[cart_id].remove((prod, prod_id))
                self.producers_lock.acquire()
                self.products.append((prod, prod_id))
                self.producers_sizes[prod_id] += 1
                self.producers_lock.release()
                return

    def place_order(self, cart_id):
        


        order = ""
        for (product, _) in self.carts[cart_id]:
            order += "{} bought {}\n".format(
                currentThread().getName(), product)
        sys.stdout.write(order)
        return self.carts[cart_id]


import time
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time


        self.producer_id = marketplace.register_producer()

    def run(self):
        while True:
            for (product, quantity, wait_time) in self.products:
                while quantity > 0:
                    if self.marketplace.publish(self.producer_id, product):
                        quantity -= 1
                        time.sleep(wait_time)
                    else:
                        time.sleep(self.republish_wait_time)


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
