


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        

        
        
        

        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):

        for cart in self.carts:

            cart_id = self.marketplace.new_cart()

            for product_info in cart:



                op = product_info['type']
                product = product_info['product']
                quantity = product_info['quantity']

                while quantity > 0:

                    if op == "add":
                        added = self.marketplace.add_to_cart(cart_id, product)

                        if added:
                            quantity = quantity - 1
                        else:
                            time.sleep(self.retry_wait_time)



                    if op == "remove":
                        self.marketplace.remove_from_cart(cart_id, product)
                        quantity = quantity - 1

            products = self.marketplace.place_order(cart_id)
        
            
            for prod in products:
                print(self.name + " bought " + str(prod))

import random
from threading import Lock

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        

        self.queue_size_per_producer = queue_size_per_producer
        self.prod_seed = 0
        self.cart_seed = 0

        
        
        self.producers = []

        
        
        
        self.items_by_producers = {}

        
        self.carts = []

        
        self.p_seed = Lock()
        self.c_seed = Lock()

    def register_producer(self):
        

        
        

        self.p_seed.acquire()

        random.seed(self.prod_seed)
        producer_id = random.randint(10000, 99999)
        self.prod_seed = self.prod_seed + 1

        self.p_seed.release()

        

        products = []
        self.items_by_producers[str(producer_id)] = products
        self.producers.append(str(producer_id)) 

        return str(producer_id)

    def publish(self, producer_id, product):
        

        products = self.items_by_producers[producer_id]

        
        if len(self.items_by_producers[producer_id]) >= self.queue_size_per_producer:
            return False

        products.append(product)

        self.items_by_producers[producer_id] = products

        return True

    def new_cart(self):
        

        self.c_seed.acquire()

        cart_id = self.cart_seed
        self.cart_seed = self.cart_seed + 1

        self.c_seed.release()

        new_cart = []
        self.carts.append(new_cart)

        return cart_id

    def add_to_cart(self, cart_id, product):
        

        for producer_id in self.producers:
            for item in self.items_by_producers[producer_id]:
                if item == product:

                    
                    self.items_by_producers[producer_id].remove(item)

                    self.carts[cart_id].append([product, producer_id])
                    
                    return True

        return False

    def remove_from_cart(self, cart_id, product):
        

        
        found = False
        for prod in self.carts[cart_id]:
            if prod[0] == product:

                
                found = True
                put_back = prod[0]
                producer_id = prod[1]

                
                self.carts[cart_id].remove(prod)

                
                self.items_by_producers[producer_id].append(put_back)

                break

        if not found:
            return False 
        return True

    def place_order(self, cart_id):
        

        list_of_products = []

        
        for prod in self.carts[cart_id]:

            
            list_of_products.append(prod[0])

        return list_of_products


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        

        
        
        

        Thread.__init__(self)
        self.products = products
        self.marketplace = marketplace


        self.republish_wait_time = republish_wait_time

    def run(self):

        producer_id = self.marketplace.register_producer()

        while True:
            for (product, quantity, waiting_time) in self.products:

                while quantity > 0:

                    published = self.marketplace.publish(producer_id, product)

                    if published:
                        quantity = quantity - 1
                        time.sleep(waiting_time)
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
