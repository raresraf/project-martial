


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
            cart_id = self.marketplace.new_cart()

            for ops in cart:
                ops_nr = 0

                
                ops_type = ops["type"]
                product = ops["product"]
                quantity = ops["quantity"]

                while ops_nr < quantity:
                    if ops_type == "add":
                        operation = self.marketplace.add_to_cart(cart_id, product)
                    else:
                        operation = self.marketplace.remove_from_cart(cart_id, product)

                    if operation or operation is None:
                        ops_nr += 1
                    else:
                        time.sleep(self.retry_wait_time)

            
            self.marketplace.place_order(cart_id)

from threading import Lock, currentThread

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        

        self.queue_size_per_producer = queue_size_per_producer

        self.carts = {} 
        self.product_producer = {} 
        self.products = [] 
        self.products_number = [] 

        self.lock_new = Lock() 
        self.lock_print = Lock() 
        self.lock_product = Lock() 
        self.lock_id = Lock() 

        self.carts_nr = 0

    def register_producer(self):
        
        self.lock_id.acquire()
        producer_id = len(self.products_number)
        
        self.products_number.append(0)
        self.lock_id.release()

        return producer_id

    def publish(self, producer_id, product):
        
        prod_id = int(producer_id)

        
        if self.products_number[prod_id] >= self.queue_size_per_producer:
            return False

        self.products_number[prod_id] += 1

        self.products.append(product)
        self.product_producer[product] = prod_id 

        return True

    def new_cart(self):
        
        self.lock_new.acquire()
        self.carts_nr += 1
        cart_id = self.carts_nr
        self.lock_new.release()

        
        self.carts[cart_id] = []

        return cart_id

    def add_to_cart(self, cart_id, product):
        
        with self.lock_product:
            if product not in self.products:
                return False

            
            producer_id = self.product_producer[product]
            self.products_number[producer_id] -= 1
            self.products.remove(product)

        
        self.carts[cart_id].append(product)

        return True

    def remove_from_cart(self, cart_id, product):
        
        self.carts[cart_id].remove(product)
        self.products.append(product)

        self.lock_product.acquire()
        producer_id = self.product_producer[product]
        self.products_number[producer_id] += 1
        self.lock_product.release()


    def place_order(self, cart_id):
        

        list_all = []
        for product in self.carts[cart_id]:
            self.lock_print.acquire()


            print(str(currentThread().getName()) + " bought " + str(product))
            list_all.append(product)
            self.lock_print.release()

        return list_all


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):

        
        while True:
            for (product, quantity, product_wait_time) in self.products:
                i = 0

                while i < quantity:
                    pub = self.marketplace.publish(str(self.producer_id), product)

                    if pub is True:
                        time.sleep(product_wait_time)
                        i += 1
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
