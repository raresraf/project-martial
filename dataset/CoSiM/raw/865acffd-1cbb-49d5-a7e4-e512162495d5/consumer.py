


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):

        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs
        

    def run(self):

        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for operation in cart:
                type_opp = operation["type"]
                quantity = operation["quantity"]
                product = operation["product"]

                while quantity > 0:
                    if type_opp == "add":
                        result = self.marketplace.add_to_cart(cart_id, product)
                    elif type_opp == "remove":
                        result = self.marketplace.remove_from_cart(cart_id, product)

                    if result or result is None:
                        quantity -= 1
                    else:
                        time.sleep(self.retry_wait_time)

            order = self.marketplace.place_order(cart_id)
            for product in order:
                result = self.kwargs["name"] + " bought " + str(product)
                print(result)

from threading import Lock
from collections import defaultdict

class Marketplace:
    
    def __init__(self, queue_size_per_producer):

        
        self.queue_size_per_producer = queue_size_per_producer

        
        self.number_of_producers = 0

        
        self.producers_queue_sizes = dict()

        
        self.products = []

        
        self.number_of_carts = 0

        
        self.carts = defaultdict()

        
        self.lock_register = Lock()

        
        self.lock_cart = Lock()

        
        self.lock_product = Lock()

    def register_producer(self):

        
        self.lock_register.acquire()

        
        producer_id = self.number_of_producers

        self.number_of_producers += 1
        


        self.producers_queue_sizes[producer_id] = 0

        self.lock_register.release()

        return producer_id

    def publish(self, producer_id, product):
        
        id_producer = int(producer_id)

        
        product.owner = id_producer

        if self.producers_queue_sizes[id_producer] < self.queue_size_per_producer:
            
            self.producers_queue_sizes[id_producer] += 1
            
            self.products.append(product)
            return True

        return False

    def new_cart(self):
        
        self.lock_cart.acquire()

        
        id_cart = self.number_of_carts

        self.number_of_carts += 1

        self.lock_cart.release()

        
        self.carts[id_cart] = []

        return id_cart

    def add_to_cart(self, cart_id, product):
        
        self.lock_product.acquire()

        if product not in self.products:
            self.lock_product.release()
            return False
        
        producer_id = product.owner


        self.producers_queue_sizes[producer_id] -= 1

        
        self.products.remove(product)

        
        self.carts[cart_id].append(product)

        self.lock_product.release()

        return True

    def remove_from_cart(self, cart_id, product):
        
        
        
        self.carts[cart_id].remove(product)
        self.products.append(product)

        
        producer_id = product.owner
        self.producers_queue_sizes[producer_id] += 1



    def place_order(self, cart_id):
        
        
        prod_list = self.carts[cart_id]
        
        self.carts[cart_id] = []

        return prod_list


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):

        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

        

    def run(self):
        
        producer_id = self.marketplace.register_producer()
        while True:
            for product in self.products:
                type_prod = product[0]
                quantity = product[1]
                wait_time = product[2]

                while quantity > 0:
                    ret = self.marketplace.publish(str(producer_id), type_prod)

                    if ret:
                        time.sleep(wait_time)
                        quantity -= 1
                    else:
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=False)
class Product:
    
    name: str
    price: int
    owner = -1


@dataclass(init=True, repr=True, order=False, frozen=False)
class Tea(Product):
    
    type: str
    owner = -1

@dataclass(init=True, repr=True, order=False, frozen=False)
class Coffee(Product):
    
    acidity: str
    roast_level: str
    owner = -1
