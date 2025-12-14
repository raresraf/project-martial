


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.new_cart = self.marketplace.new_cart


        self.add_to_cart = self.marketplace.add_to_cart
        self.remove_from_cart = self.marketplace.remove_from_cart
        self.place_order = self.marketplace.place_order


    def run(self):
        for current_cart in self.carts:

            id_cart = self.new_cart()

            for cart in current_cart:

                quantity = cart['quantity']
                product = cart['product']
                op_type = cart['type']
                step = 1

                while step <= quantity:
                    success = False
                    if op_type == "add":
                        success = self.add_to_cart(id_cart, product)
                    elif op_type == "remove":
                        success = self.remove_from_cart(id_cart, product)

                    if success is None or success:
                        step += 1
                        continue

                    sleep(self.retry_wait_time)

            self.place_order(id_cart)

from __future__ import print_function
from threading import Lock, currentThread


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer

        self.lock_register_prod = Lock() 
        self.lock_q_size = Lock() 
        self.lock_carts = Lock() 
        self.lock_printing = Lock() 

        self.carts_total = 0 
        self.producer_index = 0 
        self.prod_sizes = [] 
        self.all_products = [] 
        self.all_producers = {} 
        self.all_carts = {} 


    def register_producer(self):
        
        self.lock_register_prod.acquire()
        self.producer_index += 1
        self.prod_sizes.append(0)
        self.lock_register_prod.release()

        return self.producer_index - 1

    def publish(self, producer_id, product):
        
        id_prod = int(producer_id)

        if self.queue_size_per_producer > self.prod_sizes[id_prod]:

            self.all_products.append(product) 
            self.all_producers[product] = id_prod
            self.prod_sizes[id_prod] += 1
            return True

        return False

    def new_cart(self):
        
        self.lock_carts.acquire()

        self.carts_total += 1
        id_cart = self.carts_total

        self.all_carts[id_cart] = []

        self.lock_carts.release()

        return id_cart


    def add_to_cart(self, cart_id, product):
        

        self.lock_q_size.acquire()

        if product in self.all_products:
            ignore = False
        else:
            ignore = True

        if ignore is False:
            self.all_products.remove(product)
            self.prod_sizes[self.all_producers[product]] -= 1

        self.lock_q_size.release()

        if ignore is True:
            return False

        self.all_carts[cart_id].append(product) 

        return True

    def remove_from_cart(self, cart_id, product):
        
        self.lock_q_size.acquire()
        index = self.all_producers[product]
        self.prod_sizes[index] += 1
        self.lock_q_size.release()

        self.all_carts[cart_id].remove(product)
        self.all_products.append(product)


    def place_order(self, cart_id):
        

        prods = self.all_carts.pop(cart_id)

        for prod in prods:
            self.lock_printing.acquire()
            thread_name = currentThread().getName()
            print('{0} bought {1}'.format(thread_name, prod))
            self.lock_printing.release()

        return prods


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        self.producer_id = self.marketplace.register_producer()

    def run(self):

        while True:

            for current_product in self.products:
                step = 0
                product = current_product[0]
                products_no = current_product[1]
                waiting_time = current_product[2]

                while True:
                    published = self.marketplace.publish(str(self.producer_id), product)

                    if published is True:
                        step += 1
                        sleep(waiting_time)
                    else:
                        sleep(self.republish_wait_time)

                    if step == products_no:
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
