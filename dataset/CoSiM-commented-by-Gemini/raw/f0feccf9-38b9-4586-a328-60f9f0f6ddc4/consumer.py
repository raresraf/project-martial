


from threading import Thread
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        customer_id = self.marketplace.new_cart()
        for cart in self.carts:
            for action in cart:
                if action["type"] == "add":
                    i = 0
                    while i < action["quantity"]:
                        if self.marketplace.add_to_cart(customer_id, action["product"]):
                            i += 1
                        else:
                            time.sleep(self.retry_wait_time)
                else:
                    for i in range(action["quantity"]):
                        self.marketplace.remove_from_cart(customer_id, action["product"])
        order = self.marketplace.place_order(customer_id)
        for product in order:
            print(self.name, "bought", product)

from threading import Semaphore
import random

class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer

        self.producers = []
        self.customers = []


        self.carts = {}
        self.products = {}
        self.sem = Semaphore(1)
        self.sem1 = Semaphore(1)
        self.sem2 = Semaphore(1)
        self.sem3 = Semaphore(1)
        self.sem4 = Semaphore(1)

    def register_producer(self):
        
        while True:
            rand = random.randint(0, 5000)
            if rand not in self.producers:
                self.producers.append(rand)
                self.products[rand] = []
                return rand

    def publish(self, producer_id, product):
        
        self.sem.acquire()
        length = len(self.products[producer_id])
        if length < self.queue_size_per_producer:
            self.products[producer_id].append(product)
            self.sem.release()
            return True
        self.sem.release()
        return False

    def new_cart(self):
        
        while True:
            rand = random.randint(0, 5000)
            if rand not in self.producers:
                self.carts[rand] = []
                return rand

    def add_to_cart(self, cart_id, product):
        
        self.sem1.acquire()
        for producer in self.producers:
            products = self.products[producer]


            if product in products:
                self.carts[cart_id].append((product, producer))
                self.products[producer].remove(product)
                self.sem1.release()
                return True
        self.sem1.release()
        return False

    def remove_from_cart(self, cart_id, product):
        
        self.sem3.acquire()
        producer_id = -1
        for aux in self.carts[cart_id]:
            if aux[0] == product:
                producer_id = aux[1]
                self.carts[cart_id].remove((product, producer_id))
                self.products[producer_id].append(product)
                self.sem3.release()
                return

    def place_order(self, cart_id):
        
        self.sem4.acquire()
        order = []
        for aux in self.carts[cart_id]:
            order.append(aux[0])
        self.sem4.release()
        return order


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        while True:
            producer_id = self.marketplace.register_producer()
            for [prod, quantity, timee] in self.products:
                for i in range(quantity):
                    sleep_time = self.marketplace.publish(producer_id, prod)
                    if sleep_time:
                        time.sleep(timee)
                    else:
                        time.sleep(self.republish_wait_time)
                        i = i - 1
