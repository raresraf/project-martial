


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
            for products in cart:
                now_quantity = 0
                while now_quantity < products["quantity"]:
                    if products["type"] == "add":
                        check = self.marketplace.add_to_cart(cart_id, products["product"])
                    if products["type"] == "remove":
                        check = self.marketplace.remove_from_cart(cart_id, products["product"])
                    if check is False:
                        time.sleep(self.retry_wait_time)
                    else:
                        now_quantity += 1
            self.marketplace.place_order(cart_id)


from threading import Lock, currentThread

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.no_of_producers = 0
        self.producers = {} 
        self.no_of_carts = 0
        self.carts = {}
        self.producers_products = {} 
        self.available_products = [] 
        
        
        self.lock_reg_producers = Lock() 
        self.lock_carts = Lock() 
        self.lock_producers = Lock() 
        
        

    def register_producer(self):
        
        self.lock_reg_producers.acquire()
        self.no_of_producers += 1
        producer_id = self.no_of_producers
        self.producers[producer_id] = 0
        self.lock_reg_producers.release()
        return producer_id

    def publish(self, producer_id, product):
        
        if self.producers[int(producer_id)] >= self.queue_size_per_producer:
            return False

        self.producers[int(producer_id)] += 1
        self.producers_products[product] = int(producer_id)
        self.available_products.append(product)
        return True

    def new_cart(self):
        
        self.lock_carts.acquire()
        self.no_of_carts += 1
        cart_id = self.no_of_carts
        self.carts[cart_id] = []
        self.lock_carts.release()
        return cart_id

    def add_to_cart(self, cart_id, product):
        
        self.lock_producers.acquire()
        if product not in self.available_products:
            self.lock_producers.release()
            return False

        prod_id = self.producers_products[product]


        self.producers[prod_id] -= 1
        self.available_products.remove(product)
        self.carts[cart_id].append(product)
        self.lock_producers.release()
        return True


    def remove_from_cart(self, cart_id, product):
        
        self.carts[cart_id].remove(product)
        self.available_products.append(product)
        self.lock_producers.acquire()
        self.producers[self.producers_products[product]] += 1
        self.lock_producers.release()


    def place_order(self, cart_id):
        
        prod_list = self.carts.pop(cart_id)
        for product in prod_list:
            print("{} bought {}".format(currentThread().getName(), product))


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
            for sublist in self.products:
                count = 0
                while count < sublist[1]:
                    check = self.marketplace.publish(str(self.producer_id), sublist[0])
                    if check:
                        time.sleep(sublist[2])
                        count += 1
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
