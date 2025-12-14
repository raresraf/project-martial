




from threading import Thread
import time

QUANTITY = "quantity"
PRODUCT = "product"
TYPE = "type"
ADD = "add"
REMOVE = "remove"

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def do_action(self, id_element, item):
        
        if item["type"] == "add":
            val = self.marketplace.add_to_cart(id_element, item["product"])
        else:
            val = self.marketplace.remove_from_cart(id_element, item["product"])
        return val

    def run(self):

        val = False
        
        for i in range(0, len(self.carts)):
            
            element = self.carts[i]
            
            id_element = self.marketplace.new_cart()

            
            for j in range(0, len(element)):
                item = element[j]
                
                for k in range(0, item["quantity"]):
                    val = self.do_action(id_element, item)

                    
                    if not val:
                        
                        while True:
                            
                            time.sleep(self.retry_wait_time)
                            val = self.do_action(id_element, item)
                            if val:
                                break
            
            self.marketplace.place_order(id_element)

import uuid
from threading import Lock, currentThread

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer

        
        self.items_in_cart = {}
        
        self.number_of_carts = 0

        
        self.lock_carts = Lock()
        self.lock_remove = Lock()
        self.lock_print = Lock()
        self.lock_add = Lock()

        
        
        self.producer = {}

    def register_producer(self):
        
        
        
        id_producer = uuid.uuid4()
        element = [[], 0]
        self.producer[id_producer] = element
        return id_producer

    def publish(self, producer_id, product):
        

        
        if self.producer[producer_id][1] >= self.queue_size_per_producer:
            return False

        
        self.producer[producer_id][1] += 1
        
        self.producer[producer_id][0].append(product)
        return True

    def new_cart(self):
        
        
        with self.lock_carts:
            self.number_of_carts += 1
            cart_id = self.number_of_carts

        
        element = ["", []]
        self.items_in_cart[cart_id] = element
        return cart_id

    def add_to_cart(self, cart_id, product):
        
        

        with self.lock_add:
            id_prod = ""
            id_p = [x for x in self.producer.keys() if product in self.producer[x][0]]
            if len(id_p) == 0:
                return False

            id_prod = id_p[0]
            self.producer[id_prod][1] -= 1

        
        self.producer[id_prod][0].remove(product)

        
        self.items_in_cart[cart_id][0] = id_prod
        self.items_in_cart[cart_id][1].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        
        
        
        self.producer[self.items_in_cart[cart_id][0]][0].append(product)

        
        with self.lock_remove:
            self.producer[self.items_in_cart[cart_id][0]][1] += 1

        
        self.items_in_cart[cart_id][1].remove(product)
        return True
        
    def place_order(self, cart_id):
        
        my_prods = self.items_in_cart.pop(cart_id, None)

        for elem in my_prods[1]:


            with self.lock_print:
                print(currentThread().getName() + " bought " + str(elem))

        return my_prods>>>> file: producer.py


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.id_ = self.marketplace.register_producer()

    def run(self):
        
        prods = self.products

        
        while True:
            
            for i in range(0, len(prods)):

                
                nr_products = prods[i][1]
                product = prods[i][0]
                time_wait = prods[i][2]

                
                for j in range(0, nr_products):
                    val = self.marketplace.publish(self.id_, product)

                    
                    if val:
                        time.sleep(time_wait)
                    else:
                        
                        while True:
                            time.sleep(self.republish_wait_time)
                            val = self.marketplace.publish(self.id_, product)
                            if val:
                                break


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
