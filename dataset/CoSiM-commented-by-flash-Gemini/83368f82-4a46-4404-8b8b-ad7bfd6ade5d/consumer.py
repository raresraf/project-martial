


from threading import Thread, currentThread
from time import sleep

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        for cart in self.carts:


            cart_id = self.marketplace.new_cart()

            for action in cart:
                count = 0

                while count < action['quantity']:
                    if action['type'] == 'add':
                        


                        if self.marketplace.add_to_cart(cart_id, action['product']) is False:
                            sleep(self.retry_wait_time)

                        else:
                            count += 1

                    elif action['type'] == 'remove':
                        self.marketplace.remove_from_cart(cart_id, action['product'])
                        count += 1

            products_in_cart = self.marketplace.place_order(cart_id)
            for product in products_in_cart:
                
                print(currentThread().getName() + " bought " + str(product))

from threading import Lock

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.all_carts = {}
        self.carts_id_lock = Lock()
        self.cart_id = -1
        self.producer_id = -1
        self.producer_id_lock = Lock()
        self.products_in_marketplace = []
        self.producers_queues = {}
        self.producers_products = {}
        self.add_remove_lock = Lock()

    def register_producer(self):
        

        self.producer_id_lock.acquire()


        self.producer_id += 1   
        self.producer_id_lock.release()

        self.producers_products[self.producer_id] = []    
        self.producers_queues[self.producer_id] = 0    

        return self.producer_id

    def publish(self, producer_id, product):
        

        
        if self.producers_queues[int(producer_id)] < self.queue_size_per_producer:


            self.producers_queues[int(producer_id)] += 1
            self.products_in_marketplace.append(product)   
            self.producers_products[int(producer_id)].append(product)
            return True

        return False

    def new_cart(self):
        



        self.carts_id_lock.acquire()
        self.cart_id += 1   
        self.carts_id_lock.release()
        self.all_carts[self.cart_id] = []

        return self.cart_id

    def add_to_cart(self, cart_id, product):
        
        
        with self.add_remove_lock:

            if product in self.products_in_marketplace:


                self.products_in_marketplace.remove(product)

                for producer in self.producers_products:

                    if product in self.producers_products[producer]: 
                        self.producers_queues[producer] -= 1
                        self.producers_products[producer].remove(product)
                        break

            else: return False

        self.all_carts[cart_id].append(product)
        return True


    def remove_from_cart(self, cart_id, product):
        

        self.all_carts[cart_id].remove(product)
        self.products_in_marketplace.append(product)

    def place_order(self, cart_id):
        

        return self.all_carts[cart_id]  


from threading import Thread
from time import sleep

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs
        self.producer_id = self.marketplace.register_producer()


    def run(self):
        while True:



            for product in self.products:
                quantity = 0

                while quantity < product[1]:
                    
                    if self.marketplace.publish(str(self.producer_id), product[0]):
                        
                        sleep(product[2])
                        quantity += 1

                    else:
                        sleep(self.republish_wait_time)


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
