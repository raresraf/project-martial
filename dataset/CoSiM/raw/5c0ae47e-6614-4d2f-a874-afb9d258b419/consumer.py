

import time
from threading import Thread
from threading import Lock


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.print_locked = Lock()

    def run(self):
        


        for task_cart in self.carts:
            
            current_cart = self.marketplace.new_cart()
            for task in task_cart:
                looper = task['quantity']
                while looper > 0:
                    if task['product'] in self.marketplace.market_stock:
                        self.execute_task(task['type'], current_cart, task['product'])
                        looper -= 1
                    else:
                        time.sleep(self.retry_wait_time)

            
            order = self.marketplace.place_order(current_cart)
            with self.print_locked:
                for product in order:
                    print(self.getName(), "bought", product)

    def execute_task(self, task_type, cart_id, product):
        
        if task_type == 'remove':
            self.marketplace.remove_from_cart(cart_id, product)


        elif task_type == 'add':
            self.marketplace.add_to_cart(cart_id, product)

from threading import Lock


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer  
        self.no_of_producers = -1  
        self.no_of_carts = -1  
        self.product_creator = {}  
        self.market_stock = []  
        self.product_counter = []  
        self.cart = [[]]  

        
        self.register_locked = Lock()  
        self.cart_locked = Lock()  
        self.add_locked = Lock()  
        self.remove_locked = Lock()  
        self.publish_locked = Lock()  
        self.market_locked = Lock()  

    def register_producer(self):
        
        with self.register_locked:
            self.no_of_producers += 1
            new_prod_id = self.no_of_producers

        
        self.product_counter.append(0)
        return new_prod_id

    def publish(self, producer_id, product):
        
        if self.product_counter[producer_id] >= self.queue_size_per_producer:
            return False

        self.market_stock.append(product)

        with self.publish_locked:
            self.product_counter[producer_id] += 1
            self.product_creator[product] = producer_id

        return True

    def new_cart(self):
        
        with self.cart_locked:
            self.no_of_carts += 1
            new_cart_id = self.no_of_carts

        
        self.cart.append([])
        return new_cart_id

    def add_to_cart(self, cart_id, product):
        
        if product not in self.market_stock:
            return False
        self.cart[cart_id].append(product)
        with self.add_locked:
            self.product_counter[self.product_creator[product]] -= 1
        with self.market_locked:
            if product in self.market_stock:
                self.market_stock.remove(product)
        return True

    def remove_from_cart(self, cart_id, product):
        
        if product in self.cart[cart_id]:
            with self.cart_locked:
                self.product_counter[self.product_creator[product]] += 1
            self.cart[cart_id].remove(product)
            self.market_stock.append(product)

    def place_order(self, cart_id):
        
        return self.cart[cart_id]

import time
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.prod_id = self.marketplace.register_producer()

    def run(self):
        


        while True:
            for (product, quantity, wait_time) in self.products:
                looper = quantity
                while looper > 0:
                    response = self.marketplace.publish(self.prod_id, product)
                    if response:
                        time.sleep(wait_time)
                        looper -= 1
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
