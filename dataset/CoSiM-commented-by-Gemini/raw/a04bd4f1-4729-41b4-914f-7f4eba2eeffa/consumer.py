

import time
from threading import Thread, Lock


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.print_locked = Lock()

    def run(self):
        


        for cart in self.carts:
            my_cart = self.marketplace.new_cart()
            for to_do in cart:
                repeat = to_do['quantity']
                while repeat > 0:
                    if to_do['product'] in self.marketplace.market_stock:
                        self.execute_task(to_do['type'], my_cart, to_do['product'])
                        repeat -= 1
                    else:
                        time.sleep(self.retry_wait_time)

            order = self.marketplace.place_order(my_cart)
            with self.print_locked:
                for product in order:
                    print(self.getName(), "bought", product)

    def execute_task(self, task_type, cart_id, product):
        
        if task_type == 'add':
            self.marketplace.add_to_cart(cart_id, product)
        else:


            if task_type == 'remove':
                self.marketplace.remove_from_cart(cart_id, product)

from threading import Lock


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer  
        self.num_producers = -1  
        self.register_locked = Lock()  
        self.market_stock = []  
        self.product_counter = []  
        self.product_owner = {}  
        self.num_consumers = -1  
        self.cart = [[]]  
        self.cart_locked = Lock()
        self.add_locked = Lock()
        self.remove_locked = Lock()
        self.publish_locked = Lock()
        self.market_locked = Lock()

    def register_producer(self):
        
        with self.register_locked:  
            self.num_producers += 1
            new_producer_id = self.num_producers
        self.product_counter.append(0)
        return new_producer_id

    def publish(self, producer_id, product):
        
        if self.product_counter[producer_id] >= self.queue_size_per_producer:
            return False
        self.market_stock.append(product)
        with self.publish_locked:
            self.product_counter[producer_id] += 1
            self.product_owner[product] = producer_id
        return True

    def new_cart(self):
        
        with self.cart_locked:
            self.num_consumers += 1
            new_consumer_cart_id = self.num_consumers
        self.cart.append([])
        return new_consumer_cart_id

    def add_to_cart(self, cart_id, product):
        

        if product not in self.market_stock:
            return False
        self.cart[cart_id].append(product)
        with self.add_locked:
            self.product_counter[self.product_owner[product]] -= 1
        with self.market_locked:
            if product in self.market_stock:
                element_index = self.market_stock.index(product)
                del self.market_stock[element_index]
        return True

    def remove_from_cart(self, cart_id, product):
        

        if product in self.cart[cart_id]:
            with self.remove_locked:
                self.product_counter[self.product_owner[product]] += 1
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
        self.my_id = self.marketplace.register_producer()

    def run(self):
        


        while True:
            for (product, quantity, seconds) in self.products:
                repeat = quantity
                while repeat > 0:
                    wait = self.marketplace.publish(self.my_id, product)
                    if wait:
                        time.sleep(seconds)
                        repeat -= 1
                    else:
                        time.sleep(self.republish_wait_time)
