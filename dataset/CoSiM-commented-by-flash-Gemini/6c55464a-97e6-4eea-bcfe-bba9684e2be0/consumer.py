


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def get_name(self):
        
        return self.name

    def run(self):
        for cart in self.carts:
            


            cart_id = self.marketplace.new_cart()
            for operation in cart:
                quan_nr = 0
                
                while quan_nr < operation['quantity']:
                    
                    if operation['type'] == 'add':
                        res = self.marketplace.add_to_cart(cart_id,
                                                           operation
                                                           ['product'])
                    
                    if operation['type'] == 'remove':
                        res = self.marketplace.remove_from_cart(cart_id,
                                                                operation
                                                                ['product'])
                    
                    if res is None or res is True:
                        quan_nr = quan_nr + 1
                    
                    else:
                        
                        time.sleep(self.retry_wait_time)
            
            self.marketplace.place_order(cart_id)

from threading import currentThread, Lock

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        


        self.queue_size_per_producer = queue_size_per_producer
        self.nr_of_producers = 0 
        self.nr_of_carts = 0 
        self.nr_of_items = [] 

        self.carts = {} 
        self.producers = {} 

        self.lock = Lock()

    def register_producer(self):
        

        with self.lock:
            producer_id = self.nr_of_producers
            
            self.nr_of_producers = self.nr_of_producers + 1
            


            self.nr_of_items.insert(producer_id, 0)

        return producer_id

    def publish(self, producer_id, product):
        

        if self.nr_of_items[int(producer_id)] >= self.queue_size_per_producer:
            
            return False
        
        self.nr_of_items[int(producer_id)] += 1
        
        self.producers[product] = int(producer_id)
        return True

    def new_cart(self):
        

        with self.lock:
            
            self.nr_of_carts = self.nr_of_carts + 1
            cart_id = self.nr_of_carts
        
        self.carts[cart_id] = []
        return cart_id

    def add_to_cart(self, cart_id, product):
        

        with self.lock:
            if self.producers.get(product) is None:
                
                return False
            
            self.nr_of_items[self.producers[product]] -= 1
            
            producers_id = self.producers.pop(product)

        
        self.carts[cart_id].append(product)
        self.carts[cart_id].append(producers_id)
        return True

    def remove_from_cart(self, cart_id, product):
        

        
        if product in self.carts[cart_id]:
            index = self.carts[cart_id].index(product)
            
            self.carts[cart_id].remove(product)
            
            producers_id = self.carts[cart_id].pop(index)
            
            self.producers[product] = producers_id
            with self.lock:
                
                self.nr_of_items[int(producers_id)] += 1


    def place_order(self, cart_id):
        

        
        product_list = self.carts.pop(cart_id)
        
        for i in range(0, len(product_list), 2):
            
            with self.lock:
                print(currentThread().get_name() +" bought " +
                      str(product_list[i]))
                
                self.nr_of_items[product_list[i + 1]] -= 1
        return product_list


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = None



    def run(self):
        
        self.producer_id = self.marketplace.register_producer()
        while True:
            for product in self.products:
                produced = 0
                
                while produced < product[1]:
                    
                    
                    res = self.marketplace.publish(str(self.producer_id),
                                                   product[0])
                    if res:
                        
                        
                        time.sleep(product[2])
                        produced = produced + 1
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
