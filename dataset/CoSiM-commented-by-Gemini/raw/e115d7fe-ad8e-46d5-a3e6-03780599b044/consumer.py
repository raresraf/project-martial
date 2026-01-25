


from threading import Thread
import time


class Consumer(Thread):
    
    name = None

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()
            for prod in cart:
                
                if prod["type"] == "add":
                    while prod["quantity"] > 0:
                        check = self.marketplace.add_to_cart(cart_id, prod["product"])
                        
                        
                        if check:
                            prod["quantity"] -= 1
                        else:
                            time.sleep(self.retry_wait_time)
                else:
                    while prod["quantity"] > 0:
                        self.marketplace.remove_from_cart(cart_id, prod["product"])
                        prod["quantity"] -= 1
            cart_list = self.marketplace.place_order(cart_id)
            cart_list.reverse()
            for elem in cart_list:
                print(self.name + " bought " + str(elem))

from threading import Lock


class Marketplace:
    
    prod_id = 0
    cart_id = 0

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.prod_dict = {}
        self.cart_dict = {}
        self.lock = Lock()



    def register_producer(self):
        


        
        self.prod_id += 1
        self.prod_dict[self.prod_id] = []
        return self.prod_id

    def publish(self, producer_id, product):
        
        if len(self.prod_dict[producer_id]) < self.queue_size_per_producer:
            self.prod_dict[producer_id].append(product)
            return True
        return False



    def new_cart(self):
        


        
        self.cart_id += 1
        self.cart_dict[self.cart_id] = []
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        

        
        for key in self.prod_dict:
            self.lock.acquire()
            for prod in self.prod_dict[key]:
                if product == prod:
                    self.cart_dict[cart_id].append(product)
                    self.lock.release()
                    return True
            self.lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        
        self.cart_dict[cart_id].remove(product)   

    def place_order(self, cart_id):


        

        
        for prod in self.cart_dict[cart_id]:
            for key in self.prod_dict:
                for product in self.prod_dict[key]:
                    if product == prod:
                        self.prod_dict[key].remove(product)

        return self.cart_dict[cart_id]


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, daemon=kwargs["daemon"])
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        
        prod_id = self.marketplace.register_producer()
        while True:
            
            for prod in self.products:
                quantity = prod[1]
                while quantity > 0:
                    check = self.marketplace.publish(prod_id, prod[0])
                    
                    
                    


                    if check:
                        quantity -= 1
                        time.sleep(prod[2])
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
