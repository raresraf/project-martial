


from threading import Thread
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)



        self.carts = carts  
        self.marketplace = marketplace  
        self.retry_wait_time = retry_wait_time  

        
        self.actions = {
            'add': self.marketplace.add_to_cart,
            'remove': self.marketplace.remove_from_cart
        }

    def run(self):
        
        for cart in self.carts:
            


            cart_id = self.marketplace.new_cart()

            
            for operation in cart:
                iters = 0

                
                while iters < operation['quantity']:
                    ret = self.actions[operation['type']](
                        cart_id, operation['product'])

                    
                    if ret or ret is None:
                        iters += 1
                    else:
                        time.sleep(self.retry_wait_time)

            
            self.marketplace.place_order(cart_id)


from threading import currentThread, Lock


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer  
        self.producer_queues = []  
        self.all_products = []  
        self.producted_by = dict()  
        self.no_carts = 0  
        self.carts = dict()  

        self.producer_lock = Lock()  
        self.consumer_lock = Lock()  
        self.cart_lock = Lock()  

    def register_producer(self):
        
        
        with self.producer_lock:
            
            producer_id = len(self.producer_queues)
            self.producer_queues.append(0)

            return producer_id

    def publish(self, producer_id, product):
        
        
        if self.producer_queues[producer_id] >= self.queue_size_per_producer:
            return False

        
        self.producer_queues[producer_id] += 1
        self.producted_by[product] = producer_id

        
        self.all_products.append(product)

        return True

    def new_cart(self):
        
        
        with self.consumer_lock:
            
            cart_id = self.no_carts
            self.no_carts += 1

        
        self.carts.setdefault(cart_id, [])

        return cart_id

    def add_to_cart(self, cart_id, product):
        
        
        with self.cart_lock:


            if product not in self.all_products:
                return False

            
            self.producer_queues[self.producted_by[product]] -= 1

            
            self.all_products.remove(product)

        
        self.carts[cart_id].append(product)

        return True

    def remove_from_cart(self, cart_id, product):
        
        
        self.carts[cart_id].remove(product)

        
        self.all_products.append(product)

        
        self.producer_queues[self.producted_by[product]] += 1

    def place_order(self, cart_id):
        
        
        products = self.carts.pop(cart_id, None)

        
        for product in products:
            print(f'{currentThread().getName()} bought {product}')

        return products


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.products = products  
        self.marketplace = marketplace  
        self.republish_wait_time = republish_wait_time  

        self.own_id = marketplace.register_producer()  

    def run(self):
        while True:
            
            for (product, no_products, wait_time) in self.products:
                i = 0

                
                while i < no_products:
                    
                    if self.marketplace.publish(self.own_id, product):
                        time.sleep(wait_time)
                        i += 1
                    else:
                        
                        time.sleep(self.republish_wait_time)
