

import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):

        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for product in cart:
                i = 0

                while i < product["quantity"]:

                    if product["type"] == "remove":
                        res = self.marketplace.remove_from_cart(cart_id, product["product"])

                        if res == 1:
                            i += 1
                        else:
                            time.sleep(self.retry_wait_time)

                    else:



                        res = self.marketplace.add_to_cart(cart_id, product["product"])

                        if res:
                            i += 1
                        else:
                            time.sleep(self.retry_wait_time)

            self.marketplace.place_order(cart_id)

from threading import Lock, currentThread


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.max_queue_size = queue_size_per_producer
        self.producer_dictionary = {}
        self.current_producer_id = -1
        self.all_carts = {}
        self.add_lock = Lock()
        self.remove_lock = Lock()
        self.carts_lock = Lock()
        self.register_lock = Lock()

    def register_producer(self):
        
        self.register_lock.acquire()
        self.current_producer_id += 1
        self.register_lock.release()
        self.producer_dictionary[self.current_producer_id] = []
        return self.current_producer_id

    def publish(self, producer_id, product):
        
        p_id = int(producer_id)

        if len(self.producer_dictionary[p_id]) >= self.max_queue_size:
            return False

        self.producer_dictionary[p_id].append(product)

        return True

    def new_cart(self):
        
        self.carts_lock.acquire()
        cart_id = len(self.all_carts) + 1
        self.carts_lock.release()
        self.all_carts[cart_id] = []

        return cart_id

    def add_to_cart(self, cart_id, product):
        

        self.add_lock.acquire()
        ok_add = 0
        if self.producer_dictionary[self.current_producer_id].count(product) > 0:
            self.producer_dictionary[self.current_producer_id].remove(product)
            self.all_carts[cart_id].append(product)
            ok_add = 1
        else:
            for (_, queue) in self.producer_dictionary.items():
                if queue.count(product) > 0:
                    queue.remove(product)
                    self.all_carts[cart_id].append(product)
                    ok_add = 1
                    break
        self.add_lock.release()

        if ok_add == 0:
            return False
        return True

    def remove_from_cart(self, cart_id, product):
        
        ok_remove = 0
        if len(self.producer_dictionary[self.current_producer_id]) < self.max_queue_size:
            self.producer_dictionary[self.current_producer_id].append(product)
            ok_remove = 1
        else:
            for (_, queue) in self.producer_dictionary.items():
                if len(queue) < self.max_queue_size:
                    queue.append(product)
                    ok_remove = 1
                    break
        if ok_remove == 1:
            self.remove_lock.acquire()
            self.all_carts[cart_id].remove(product)
            self.remove_lock.release()
        return ok_remove

    def place_order(self, cart_id):
        
        for prod in self.all_carts[cart_id]:
            print(str(currentThread().getName()) + " bought " + str(prod))

import time
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.product_id = self.marketplace.register_producer()

    def run(self):
        while True:


            for elem in self.products:
                curr_prod = 0

                while curr_prod < elem[1]:
                    publish_ok = self.marketplace.publish(str(self.product_id), elem[0])

                    if publish_ok:
                        time.sleep(elem[2])
                        curr_prod += 1
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
