


from threading import Thread
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs
        self.name = kwargs['name']

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for param in cart:
                if param["type"] == "add":


                    i = 0
                    while i < param["quantity"]:
                        response = self.marketplace.add_to_cart(cart_id, param["product"])

                        while not response:
                            time.sleep(self.retry_wait_time)
                            response = self.marketplace.add_to_cart(cart_id, param["product"])
                        i += 1
                elif param["type"] == "remove":
                    for i in range(param["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, param["product"])

            checkout = self.marketplace.place_order(cart_id)
            for i in checkout:
                print(self.name + " bought " + str(i))



from threading import Semaphore
from collections import defaultdict


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.products = defaultdict(list)
        self.consumers = defaultdict(list)
        self.producer_id = 0
        self.consumer_id = 0
        self.producer_lock = Semaphore(1)
        self.consumer_lock = Semaphore(1)
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        
        self.producer_lock.acquire()
        id_producer = self.producer_id
        self.products[id_producer] = []
        self.producer_id += 1
        self.producer_lock.release()

        return id_producer

    def publish(self, producer_id, product):
        
        products_list = self.products.get(producer_id)
        length = len(products_list)
        if length < self.queue_size_per_producer:
            self.products[producer_id].append(product)
            return True
        return False

    def new_cart(self):
        
        self.consumer_lock.acquire()
        cart_id = self.consumer_id
        self.consumers[cart_id] = []
        self.consumer_id += 1
        self.consumer_lock.release()

        return cart_id

    def add_to_cart(self, cart_id, product):
        

        for key, value in self.products.items():
            if product in value:
                self.consumers[cart_id].append((product, key))
                product_list = self.products.get(key)
                product_list.remove(product)
                self.products[key] = product_list
                return True
        return False

    def remove_from_cart(self, cart_id, product):
        

        for key, value in self.consumers.items():
            if key == cart_id:
                for prod in self.consumers.get(key):
                    if product == prod[0]:
                        self.consumers[cart_id].remove(prod)
                        self.products[prod[1]].append(product)
                        return

    def place_order(self, cart_id):
        
        order_list = []
        for key, value in self.consumers.items():
            if key == cart_id:
                for prod in value:
                    order_list.append(prod[0])

        new_dict = {key: val for key, val in self.consumers.items() if key != cart_id}
        self.consumers = new_dict
        return order_list


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs


        self.producer_id = 0

    def run(self):
        self.producer_id = self.marketplace.register_producer()
        while True:
            for entry in self.products:


                for i in range(entry[1]):
                    response = self.marketplace.publish(self.producer_id, entry[0])

                    while not response:
                        time.sleep(self.republish_wait_time)
                        response = self.marketplace.publish(self.producer_id, entry[0])

                    time.sleep(entry[2])


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
