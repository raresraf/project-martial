


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

            for operation in cart:
                quantity = operation["quantity"]
                if operation["type"] == "add":
                    for _ in range(quantity):
                        while self.marketplace.add_to_cart(cart_id, operation["product"]) is False:
                            time.sleep(self.retry_wait_time)

                if operation["type"] == "remove":
                    for _ in range(quantity):
                        self.marketplace.remove_from_cart(cart_id, operation["product"])

            self.marketplace.place_order(cart_id)


from threading import Lock, currentThread

class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.lock = Lock()
        self.cid = 0
        self.producer_items = [] 
        self.products = [] 
        self.carts = {} 
        self.producers = {} 

    def register_producer(self):
        
        self.lock.acquire()
        prod_id = len(self.producer_items)
        self.producer_items.append(0)
        self.lock.release()
        return prod_id

    def publish(self, producer_id, product):
        
        producer_id = int(producer_id)

        if self.producer_items[producer_id] >= self.queue_size_per_producer:
            return False

        self.producer_items[producer_id] += 1
        self.products.append(product)
        self.producers[product] = producer_id

        return True

    def new_cart(self):
        
        self.lock.acquire()
        self.cid += 1
        cart_id = self.cid
        self.lock.release()

        self.carts[cart_id] = []
        return cart_id

    def add_to_cart(self, cart_id, product):
        
        self.lock.acquire()
        if product not in self.products:
            self.lock.release()
            return False

        self.producer_items[self.producers[product]] -= 1


        self.products.remove(product)

        self.carts[cart_id].append(product)
        self.lock.release()

        return True

    def remove_from_cart(self, cart_id, product):
        
        self.carts[cart_id].remove(product)
        self.products.append(product)

        self.lock.acquire()
        self.producer_items[self.producers[product]] += 1
        self.lock.release()

    def place_order(self, cart_id):
        
        products_list = self.carts.get(cart_id)
        for product in products_list:
            self.lock.acquire()
            print("{} bought {}".format(currentThread().getName(), product))
            self.lock.release()

        return products_list


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
        while True:
            for product, quantity, wait_time in self.products:
                for _ in range(quantity):
                    while self.marketplace.publish(str(self.prod_id), product) is False:
                        time.sleep(self.republish_wait_time)

                    time.sleep(wait_time)


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
