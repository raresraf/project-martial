/**
 * @file consumer.py
 * @brief Semantic documentation for consumer.py. 
 *        This is a placeholder. Detailed semantic analysis will be applied later.
 */



from threading import Thread
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        for cart in self.carts:
            c_id = self.marketplace.new_cart()

            for cons_op in cart:
                num_of_ops = 0

                while num_of_ops < cons_op["quantity"]:
                    if cons_op["type"] == "add":
                        ret = self.marketplace.add_to_cart(str(c_id), cons_op["product"])
                    else:


                        ret = self.marketplace.remove_from_cart(str(c_id), cons_op["product"])

                    if ret:
                        num_of_ops += 1
                    else:
                        time.sleep(self.retry_wait_time)

            self.marketplace.place_order(c_id)

from threading import Lock, currentThread


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.max_size = queue_size_per_producer
        self.producers = {}
        self.p_index = 0
        self.carts = {}
        self.c_index = 0

        self.lock_register = Lock()
        self.lock_carts = Lock()


    def register_producer(self):
        
        with self.lock_register:
            p_id = self.p_index
            self.producers[p_id] = []
            self.p_index += 1

        return p_id


    def publish(self, producer_id, product):
        
        p_id = int(producer_id)

        if len(self.producers[p_id]) >= self.max_size:
            return False

        self.producers[p_id].append(product)

        return True

    def new_cart(self):
        
        with self.lock_carts:
            c_id = self.c_index
            self.carts[c_id] = []
            self.c_index += 1

        return c_id


    def add_to_cart(self, cart_id, product):
        
        c_id = int(cart_id)

        for i in range(0, len(self.producers)):
            for prod in self.producers[i]:
                if product == prod:
                    self.producers[i].remove(product)
                    self.carts[c_id].append(product)
                    return True

        return False


    def remove_from_cart(self, cart_id, product):
        
        c_id = int(cart_id)

        if product in self.carts[c_id]:
            self.carts[c_id].remove(product)
            self.producers[self.p_index - 1].append(product)
            return True

        return False

    def place_order(self, cart_id):
        
        c_id = int(cart_id)

        cart = self.carts.pop(c_id, None)

        for prod in cart:
            print("{} bought {}".format(currentThread().getName(), prod))

        return cart

import time
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.p_id = self.marketplace.register_producer()

    def run(self):
        while 1:


            for p_list in self.products:
                num_of_p = 0

                while num_of_p < p_list[1]:
                    ret = self.marketplace.publish(str(self.p_id), p_list[0])

                    if ret:
                        time.sleep(p_list[2])
                        num_of_p += 1
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
