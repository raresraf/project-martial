


from threading import Thread
from time import sleep

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.name = kwargs['name']
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        
        for cart in self.carts:


            cart_id = self.marketplace.new_cart()

            
            for operation in cart:
                for _ in range(operation['quantity']):
                    if operation['type'] == 'add':
                        while not self.marketplace.add_to_cart(cart_id, operation['product']):
                            sleep(self.retry_wait_time)
                    elif operation['type'] == 'remove':
                        self.marketplace.remove_from_cart(cart_id, operation['product'])

            
            order = self.marketplace.place_order(cart_id)
            for product in order:
                print("%s bought %s" % (self.name, product))


from threading import Lock

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.last_producer_id = 0
        self.last_cart_id = 0
        
        self.products_per_producer = {}
        
        self.carts = {}
        self.cart_lock = Lock()
        self.producer_id_lock = Lock()
        self.add_to_cart_lock = Lock()

    def register_producer(self):
        
        with self.producer_id_lock:
            self.last_producer_id += 1
            self.products_per_producer[self.last_producer_id] = []
            return self.last_producer_id

    def publish(self, producer_id, product):
        

        if len(self.products_per_producer[producer_id]) == self.queue_size_per_producer:
            return False

        self.products_per_producer[producer_id].append(product)
        return True

    def new_cart(self):
        
        with self.cart_lock:
            self.last_cart_id += 1
            self.carts[self.last_cart_id] = []
            return self.last_cart_id

    def add_to_cart(self, cart_id, product):
        
        
        for producer_id, products in self.products_per_producer.items():


            with self.add_to_cart_lock:
                if product in products:
                    products.remove(product)
                    self.carts[cart_id].append((producer_id, product))
                    return True

        return False

    def remove_from_cart(self, cart_id, product):
        
        
        producer_id = 0
        for cart_producer_id, cart_product in self.carts[cart_id]:
            if cart_product == product:
                producer_id = cart_producer_id

        
        self.carts[cart_id].remove((producer_id, product))
        self.products_per_producer[producer_id].append(product)

    def place_order(self, cart_id):
        
        return [product for _, product in self.carts[cart_id]]


from threading import Thread
from time import sleep

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = -1

    def run(self):
        
        self.producer_id = self.marketplace.register_producer()

        
        
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    while not self.marketplace.publish(self.producer_id, product[0]):
                        sleep(self.republish_wait_time)

                    sleep(product[2])
