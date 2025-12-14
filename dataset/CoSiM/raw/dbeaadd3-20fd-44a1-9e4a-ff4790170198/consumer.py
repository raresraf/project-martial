

import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        super().__init__()
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    
    
    def add_command(self, id_cart, product, quantity):
        


        for _ in range(quantity):
            status = False
            while not status:
                status = self.marketplace.add_to_cart(id_cart, product)
                if not status:
                    time.sleep(self.retry_wait_time)

    
    
    def remove_command(self, id_cart, product, quantity):
        


        for _ in range(quantity):
            self.marketplace.remove_from_cart(id_cart, product)

    def run(self):
        for carts in self.carts:
            id_cart = self.marketplace.new_cart()
            for i in carts:
                command = i.get('type')
                if command == 'add':
                    self.add_command(id_cart, i.get('product'), i.get('quantity'))
                else:
                    self.remove_command(id_cart, i.get('product'), i.get('quantity'))

            return_list = self.marketplace.place_order(id_cart)

            for i in enumerate(return_list):
                res = self.kwargs.get('name') + " bought " + format(i[1])
                print(res)

import time
from threading import Semaphore, RLock


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = -1
        self.id_carts = -1
        self.producers_list = []  
        self.market_contains = []  
        self.carts_contains = []  
        self.lock_producers = RLock()
        self.lock_consumers = RLock()
        self.number_of_orders_placed = -1  
        self.consumers_semaphore = Semaphore(0)

    def register_producer(self):
        
        self.market_contains.append([])
        self.producers_list.append(self.queue_size_per_producer)
        with self.lock_producers:
            self.id_producer += 1
            return self.id_producer

    
    
    
    
    def publish(self, producer_id, product, wait_time_for_making_product):
        

        if self.producers_list[producer_id] != 0:
            self.market_contains[producer_id].append([product, True])
            self.producers_list[producer_id] -= 1
            self.consumers_semaphore.release()
            time.sleep(wait_time_for_making_product)
            return True
        return False

    def new_cart(self):
        
        with self.lock_consumers:
            self.id_carts += 1
            self.carts_contains.append([])
            return self.id_carts

    def add_to_cart(self, cart_id, product):
        
        self.consumers_semaphore.acquire()
        for lists in self.market_contains:
            for item in lists:
                if item[0] is product and item[1] is True:
                    self.carts_contains[cart_id].append(product)
                    with self.lock_consumers:
                        self.producers_list[self.market_contains.index(lists)] += 1
                        item[1] = False
                    return True
        return False

    def remove_from_cart(self, cart_id, product):
        
        self.carts_contains[cart_id].remove(product)
        for lists in self.market_contains:
            for item in lists:
                if item[0] is product and item[1] is False:
                    with self.lock_consumers:
                        self.producers_list[self.market_contains.index(lists)] -= 1


                        item[1] = True
        self.consumers_semaphore.release()

    def place_order(self, cart_id):
        
        with self.lock_consumers:
            self.number_of_orders_placed += 1
            return_list = self.carts_contains[cart_id]
            return return_list

    
    
    def number_of_orders(self):
        
        with self.lock_producers:
            if self.number_of_orders_placed == self.id_carts:
                return False
            return True

import time
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        super().__init__()
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    
    
    
    def helper_run(self, producer_id, command_info):
        
        for _ in range(command_info[1]):
            status = False
            while not status:
                status = self.marketplace.publish(producer_id, command_info[0], command_info[2])
                if not status:
                    time.sleep(self.republish_wait_time)
                if not self.marketplace.number_of_orders():
                    status = True

    def run(self):
        id_prod = self.marketplace.register_producer()
        time_to_run = True
        while time_to_run:
            for i in self.products:
                self.helper_run(id_prod, i)
            time_to_run = self.marketplace.number_of_orders()
