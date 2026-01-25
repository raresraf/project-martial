


from threading import Thread, Lock
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace


        self.retry_wait_time = retry_wait_time
        self.cart_id = 0

    def run(self):
        for cart in self.carts:
            lock = Lock()
            lock.acquire()
            self.cart_id = self.marketplace.new_cart()
            lock.release()

            for ops in cart:
                type_operation = ops['type']
                product = ops['product']
                quantity = ops['quantity']
                i = 0

                if type_operation == "add":
                    while i < quantity:
                        status = self.marketplace.add_to_cart(self.cart_id, product)
                        if not status:
                            time.sleep(self.retry_wait_time)
                        else:
                            i += 1
                else:
                    while i < quantity:
                        self.marketplace.remove_from_cart(self.cart_id, product)
                        i += 1

            placed_order_cart = self.marketplace.place_order(self.cart_id)

            lock = Lock()
            for product_bought in placed_order_cart:
                lock.acquire()
                print("{} bought {}".format(self.name, product_bought))
                lock.release()



class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.count_producers = 0  
        self.carts = []  
        self.producer_products = []  
        self.reserved_products = []  

    def register_producer(self):
        
        self.producer_products.append([])
        self.reserved_products.append([])
        self.count_producers = self.count_producers + 1

        return self.count_producers - 1

    def publish(self, producer_id, product):
        
        if len(self.producer_products[producer_id]) < self.queue_size_per_producer:
            self.producer_products[producer_id].append(product)
            return True

        return False

    def new_cart(self):
        
        self.carts.append([])

        return len(self.carts) - 1

    def add_to_cart(self, cart_id, product):
        
        
        
        for i in range(self.count_producers):


            if product in self.producer_products[i]:
                self.carts[cart_id].append(product)
                self.reserved_products[i].append(product)
                self.producer_products[i].remove(product)
                return True

        return False

    def remove_from_cart(self, cart_id, product):
        


        self.carts[cart_id].remove(product)

        
        for i in range(self.count_producers):
            if product in self.reserved_products[i]:
                self.reserved_products[i].remove(product)
                self.producer_products[i].append(product)
                return True

        return False

    def place_order(self, cart_id):
        

        return self.carts[cart_id]


from threading import Thread, Lock
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)


        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = 0

    def run(self):
        lock = Lock()
        lock.acquire()
        self.producer_id = self.marketplace.register_producer()
        lock.release()

        while True:
            for product in self.products:
                product_id = product[0]
                quantity = product[1]
                waiting_time = product[2]
                i = 0

                while i < quantity:
                    status = self.marketplace.publish(self.producer_id, product_id)
                    if not status:
                        time.sleep(self.republish_wait_time)
                    else:
                        i += 1
                        time.sleep(waiting_time)


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
