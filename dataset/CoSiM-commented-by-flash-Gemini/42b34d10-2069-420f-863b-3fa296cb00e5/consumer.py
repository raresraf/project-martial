


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.consumer_name = kwargs["name"]

    def run(self):
        
        for cart in self.carts:

            cart_id = self.marketplace.new_cart()

            for data in cart:
                i = 0
                item = data["product"]
                operation = data["type"]

                while i < data["quantity"]:

                    if operation == "add":
                        available = self.marketplace.add_to_cart(cart_id, item)
                        if available:
                            i += 1
                        else:
                            time.sleep(self.retry_wait_time)

                    if operation == "remove":
                        self.marketplace.remove_from_cart(cart_id, item)
                        i += 1

            order = self.marketplace.place_order(cart_id)

            for item in order:
                print(self.consumer_name + " bought "+ str(item[0]))


from threading import Lock

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer



        self.num_prod = 0
        self.num_carts = 0
        
        self.prod_num_items = []
        
        self.items = {}
        
        self.carts = {}

        self.register_lock = Lock()
        self.new_cart_lock = Lock()
        self.cart_lock = Lock()



    def register_producer(self):
        


        with self.register_lock:
            prod_id = self.num_prod
            self.num_prod += 1

        self.prod_num_items.append(0)
        self.items[prod_id] = []
        return prod_id

    def publish(self, producer_id, product):
        
        if self.prod_num_items[producer_id] >= self.queue_size_per_producer:
            return False
        self.items[producer_id].append(product)
        self.prod_num_items[producer_id] += 1

        return True

    def new_cart(self):
        
        with self.new_cart_lock:
            cart_id = self.num_carts
            self.num_carts += 1

        self.carts[cart_id] = []

        return cart_id

    def add_to_cart(self, cart_id, product):
        
        found = False
        with self.cart_lock:
            for i in self.items:
                if product in self.items[i]:


                    self.items[i].remove(product)
                    self.prod_num_items[i] -= 1
                    prod_id = i
                    found = True
                    break

        if found:
            self.carts[cart_id].append((product, prod_id))

        return found

    def remove_from_cart(self, cart_id, product):
        



        for item, producer in self.carts[cart_id]:
            if item is product:
                prod_id = producer
                self.carts[cart_id].remove((item, producer))
                break

        self.items[prod_id].append(product)

        with self.cart_lock:
            self.prod_num_items[prod_id] += 1

    def place_order(self, cart_id):
        
        res = self.carts.pop(cart_id)
        return res


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products


        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        
        prod_id = self.marketplace.register_producer()
        while True:
            for (item, quantity, wait_time) in self.products:
                i = 0
                while i < quantity:
                    available = self.marketplace.publish(prod_id, item)

                    if available:
                        time.sleep(wait_time)
                        i += 1
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
