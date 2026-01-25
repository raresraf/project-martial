


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
            for order in cart:
                i = 0
                if order["type"] == "add":
                    while i < order["quantity"]:
                        while True:
                            out = self.marketplace.add_to_cart(cart_id, order["product"])
                            if out == False:
                                time.sleep(self.retry_wait_time)
                            else:
                                break
                        i += 1
                if order["type"] == "remove":
                    while i < order["quantity"]:
                        self.marketplace.remove_from_cart(cart_id, order["product"])
                        i += 1
            self.marketplace.place_order(cart_id)


from threading import currentThread, Lock

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        


        self.max_q_per_prod = queue_size_per_producer
        self.producer_id = 0
        self.products = {}
        self.cart_id = 0
        self.carts = {}
        self.marketplace = []

    def register_producer(self):
        
        lock = Lock()
        lock.acquire()
        self.producer_id += 1
        self.products[self.producer_id] = [] 
        return self.producer_id
        lock.release()

    def publish(self, producer_id, product):
        
        num_prod = self.products[producer_id]
        if len(num_prod) >= self.max_q_per_prod:
            return False

        self.marketplace.append((product, producer_id))
        num_prod.append(product)
        return True

    def new_cart(self):
        
        lock = Lock()
        lock.acquire()
        self.cart_id += 1
        cart_id = self.cart_id
        self.carts[cart_id] = [] 
        return cart_id
        lock.release()

    def add_to_cart(self, cart_id, product):
        
        for (product_type, producer_id) in self.marketplace:
            if product_type == product:
                if product in self.products[producer_id]:
                    self.carts[cart_id].append((product, producer_id))
                    self.marketplace.remove((product_type, producer_id))
                    self.products[producer_id].remove(product)
                    return True
        return False

    def remove_from_cart(self, cart_id, product):
        



        for (product_type, _producer_id) in self.carts[cart_id]:
            if product_type == product:
                self.carts[cart_id].remove((product, _producer_id))
                self.marketplace.append((product_type, _producer_id))
                self.products[_producer_id].append(product)
                break

    def place_order(self, cart_id):
        
        for (product, _producer_id) in self.carts[cart_id]:
            print("{} bought {}".format(currentThread().getName(), product))

        return self.carts.pop(cart_id, None)


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
            for (product, quantity, wait_time) in self.products:
                while quantity:
                    out = self.marketplace.publish(self.producer_id, product)
                    if out == False:
                        time.sleep(self.republish_wait_time)
                    else:
                        quantity -= 1
                        time.sleep(wait_time)
