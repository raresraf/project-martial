


import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        
        id_cart = self.marketplace.new_cart()

        for cart in self.carts:



            for command in cart:
                
                if command["type"] == "add":
                    for i in range(command["quantity"]):
                        available = self.marketplace.add_to_cart(id_cart, command["product"]) 
                        while not available:
                            time.sleep(self.retry_wait_time)
                            available = self.marketplace.add_to_cart(id_cart, command["product"])

                else:
                    for i in range(command["quantity"]):
                        
                        self.marketplace.remove_from_cart(id_cart, command["product"])

        
        self.marketplace.place_order(id_cart)

from threading import Lock


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size = queue_size_per_producer
        self.producer_id = 0 
        self.cart_id = 0 
        self.market = [[]] 
        self.cart = [[]] 
        self.lock_add = Lock() 
        self.lock_remove = Lock()
        self.lock_cart = Lock()
        self.lock_producer = Lock()
        self.lock_print = Lock()

    def register_producer(self):
        

        

        self.lock_producer.acquire()
        self.producer_id += 1
        self.market.append([]) 
        self.lock_producer.release()
        return self.producer_id

    def publish(self, producer_id, product):
        

        

        if len(self.market[producer_id - 1]) >= self.queue_size:
            return False
        self.market[producer_id - 1].append(product)
        return True

    def new_cart(self):
        

        

        self.lock_cart.acquire()
        self.cart_id += 1
        self.cart.append([])
        self.lock_cart.release()
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        

        



        self.lock_add.acquire()
        for i in range(len(self.market)):
            for j in range(len(self.market[i])):
                if self.market[i][j] == product: 
                    self.cart[cart_id - 1].append((product, i))
                    self.market[i].remove(product)
                    self.lock_add.release()
                    return True

        self.lock_add.release()
        return False 


    def remove_from_cart(self, cart_id, product):
        

        

        self.lock_remove.acquire()
        for i in range(len(self.cart[cart_id - 1])):
            if self.cart[cart_id - 1][i][0] == product: 
                self.market[self.cart[cart_id - 1][i][1]].append(product)
                prod_id = self.cart[cart_id - 1][i][1]
                self.cart[cart_id - 1].remove((product, prod_id))
                break

        self.lock_remove.release()

    def place_order(self, cart_id):
        

        

        self.lock_print.acquire()
        for i in range(len(self.cart[cart_id - 1])):
            print("cons"+ str(cart_id) + " bought " + str(self.cart[cart_id - 1][i][0]))
        self.lock_print.release()


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.marketplace = marketplace
        self.product_list = products


        self.wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        
        id_producer = self.marketplace.register_producer()
        while 1:
            for prod in self.product_list:
                
                for i in range(prod[1]):
                    
                    can_add = self.marketplace.publish(id_producer, prod[0])
                    if not can_add:
                        while not can_add:
                            time.sleep(self.wait_time)
                            can_add = self.marketplace.publish(id_producer, prod[0])
                    else:
                        time.sleep(prod[2])
