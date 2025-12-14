


from threading import Thread, currentThread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        for cart in self.carts:


            id_cart = self.marketplace.new_cart()
            for operation in cart:
                op_count = 0
                while op_count < operation['quantity']:
                    if operation['type'] == 'add':
                        if self.marketplace.add_to_cart(id_cart, operation['product']) is False:


                            time.sleep(self.retry_wait_time)        
                        else:
                            op_count += 1
                    elif operation['type'] == 'remove':
                        self.marketplace.remove_from_cart(id_cart, operation['product'])
                        op_count += 1

            products_in_cart = self.marketplace.place_order(id_cart)
            for product in products_in_cart:                               
                print(currentThread().getName() + " bought " + str(product))

from threading import  Lock

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.all_carts = {}
        self.id_carts_lock = Lock()
        self.id_cart = -1
        self.id_producer = -1
        self.id_producer_lock = Lock()
        self.products_in_marketplace = []
        self.producers_queues = {}
        self.producers_products = {}
        self.add_remove_lock = Lock()

    def register_producer(self):
        

        self.id_producer_lock.acquire()


        self.id_producer += 1               
        self.id_producer_lock.release()

        self.producers_products[self.id_producer] = []     
        self.producers_queues[self.id_producer] = 0        

        return self.id_producer

    def publish(self, producer_id, product):
        

        if not self.producers_queues[int(producer_id)] < self.queue_size_per_producer:
            return False



        self.producers_queues[int(producer_id)] += 1
        self.products_in_marketplace.append(product)                   
        self.producers_products[int(producer_id)].append(product)

        return True

    def new_cart(self):
        



        self.id_carts_lock.acquire()
        self.id_cart += 1                       
        self.id_carts_lock.release()
        self.all_carts[self.id_cart] = []

        return self.id_cart

    def add_to_cart(self, cart_id, product):
        


        with self.add_remove_lock:
            if product not in self.products_in_marketplace:
                return False


            self.products_in_marketplace.remove(product)
            for producer in self.producers_products:
                if product in self.producers_products[producer]:            
                    self.producers_queues[producer] -= 1                    
                    self.producers_products[producer].remove(product)       
                    break

        self.all_carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        

        self.all_carts[cart_id].remove(product)

        with self.add_remove_lock:
            self.products_in_marketplace.append(product)
            for producer in self.producers_products:
                if product in self.producers_products[producer]:        
                    self.producers_queues[producer] += 1                
                    self.producers_products[producer].append(product)   
                    break


    def place_order(self, cart_id):
        

        return self.all_carts[cart_id]                             


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producerID = self.marketplace.register_producer()


    def run(self):
        while True:


            for product in self.products:
                quantity = 0
                while quantity < product[1]:
                    if self.marketplace.publish(str(self.producerID), product[0]):
                        time.sleep(product[2])                     
                        quantity += 1                              
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
