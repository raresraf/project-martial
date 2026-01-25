


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
            for action in cart:
                for _ in range(action["quantity"]):
                    if action["type"] == "add":
                        ret = self.marketplace.add_to_cart(cart_id, action["product"])
                        while ret is False:
                            time.sleep(self.retry_wait_time)
                            ret = self.marketplace.add_to_cart(cart_id, action["product"])
                    elif action["type"] == "remove":
                        self.marketplace.remove_from_cart(cart_id, action["product"])
            self.marketplace.place_order(cart_id)

from threading import Lock
from threading import currentThread

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.nr_of_producers = 0 
        self.nr_carts = 0 
        self.carts = [] 
        self.products = [] 
        self.map_between_product_and_id = {} 
        self.register = Lock() 
        self.publ = Lock() 
        self.for_cart = Lock() 
        self.for_action = Lock() 

    def register_producer(self):
        
        with self.register:
            self.nr_of_producers += 1

        
        self.products.append([])
        return self.nr_of_producers-1

    def publish(self, producer_id, product):

        
        with self.publ:

            if len(self.products[producer_id]) >= self.queue_size_per_producer:
                return False
        
        self.products[producer_id].append(product)
        self.map_between_product_and_id[product] = producer_id
        return True
    def new_cart(self):
        
        with self.for_cart:
            self.nr_carts += 1

        
        self.carts.append([])

        return self.nr_carts-1

    def add_to_cart(self, cart_id, product):
        
        
        with self.for_action:
            for lst in self.products:
                if product in lst:


                    if product in self.products[self.map_between_product_and_id[product]]:
                        self.products[self.map_between_product_and_id[product]].remove(product)
                        self.carts[cart_id].append(product)
                        return True
        return False

    def remove_from_cart(self, cart_id, product):
        
        


        if product in self.carts[cart_id]:
            self.carts[cart_id].remove(product)
            self.products[self.map_between_product_and_id[product]].append(product)

    def place_order(self, cart_id):
        
        
        lst = self.carts[cart_id]
        for prod in lst:
            print("{} bought {}".format(currentThread().getName(), prod))

        return lst


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.prod_id = self.marketplace.register_producer()
    
    
    
    def run(self):
        while 1:
            for prod in self.products:
                for _ in range(prod[1]):
                    if self.marketplace.publish(self.prod_id, prod[0]):
                        time.sleep(prod[2])
                    else:
                        while self.marketplace.publish(self.prod_id, prod[0]) is False:
                            time.sleep(self.republish_wait_time)
