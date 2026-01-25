


from threading import Thread
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)


        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']



    def add_product(self, cart_id, product, quantity):
        
        for _ in range(quantity):
            tmp = self.marketplace.add_to_cart(cart_id, product)

            
            while tmp is False:
                time.sleep(self.retry_wait_time)
                tmp = self.marketplace.add_to_cart(cart_id, product)


    def remove_product(self, cart_id, product, quantity):
        

        for _ in range(quantity):
            self.marketplace.remove_from_cart(cart_id, product)


    def run(self):
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()

            for request in cart:
                
                order = request["type"]
                product = request["product"]
                quantity = request["quantity"]

                if order == "add":
                    self.add_product(cart_id, product, quantity)
                elif order == "remove":
                    self.remove_product(cart_id, product, quantity)

            
            order = self.marketplace.place_order(cart_id)

            
            for product in order:
                print(self.name + " bought " + str(product))

from threading import Lock


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        

        self.queue_size_per_producer = queue_size_per_producer
        self.prod_count = 0
        self.cart_count = 0

        self.products = {}      
        self.carts = {}         
        self.lock = Lock()

    def register_producer(self):
        

        
        self.lock.acquire()
        self.prod_count = self.prod_count + 1
        self.lock.release()

        self.products[self.prod_count] = []
        return self.prod_count


    def publish(self, producer_id, product):
        

        lenght = len(self.products[producer_id])
        if lenght > self.queue_size_per_producer:
            
            return False

        
        self.products[producer_id].append(product)
        return True


    def new_cart(self):
        

        
        self.lock.acquire()
        self.cart_count = self.cart_count + 1
        self.lock.release()

        self.carts[self.cart_count] = []
        return self.cart_count


    def add_to_cart(self, cart_id, product):
        

        for producer_id in self.products:
            if product in self.products[producer_id]:
                
                tmp = (product, producer_id)

                
                self.carts[cart_id].append(tmp)
                self.products[producer_id].remove(product)
                return True

        
        return False


    def remove_from_cart(self, cart_id, product):
        

        
        for tmp in self.carts[cart_id]:
            current_prod = tmp[0]
            producer_id = tmp[1]

            if product == current_prod:
                
                self.products[producer_id].append(product)
                self.carts[cart_id].remove(tmp)
                return


    def place_order(self, cart_id):
        

        order = []
        
        for product in self.carts[cart_id]:
            order.append(product[0])

        
        self.carts.pop(cart_id)

        return order


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
            for task in self.products:
                
                product = task[0]
                quantity = task[1]
                making_time = task[2]

                
                for _ in range(quantity):
                    temp = self.marketplace.publish(self.producer_id, product)

                    
                    while not temp:
                        time.sleep(self.republish_wait_time)
                        temp = self.marketplace.publish(self.producer_id, product)

                    
                    time.sleep(making_time)


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
