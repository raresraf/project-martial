


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
                cart_operations = 0
                quantity = operation["quantity"]

                
                while cart_operations < quantity:

                    
                    operation_name = operation["type"]
                    product = operation["product"]

                    
                    if operation_name == "add":
                        ret = self.marketplace.add_to_cart(cart_id, product)
                    elif operation_name == "remove":


                        ret = self.marketplace.remove_from_cart(cart_id, product)

                    
                    if ret is None or ret:
                        cart_operations += 1
                    else:
                        time.sleep(self.retry_wait_time)

            self.marketplace.place_order(cart_id)


from threading import Lock, currentThread


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer

        
        self.products = []
        
        self.carts = {}

        
        self.nr_prod_in_queue = []

        


        self.products_owners = {}

        self.nr_carts = 0

        
        self.lock_products_queue = Lock()

        
        self.lock_nr_carts = Lock()


    def register_producer(self):
        
        
        
        producer_id = len(self.nr_prod_in_queue)
        self.nr_prod_in_queue.append(0)

        return producer_id


    def publish(self, producer_id, product):
        

        if self.nr_prod_in_queue[producer_id] < self.queue_size_per_producer:
            self.products.append(product)
            self.nr_prod_in_queue[producer_id] += 1
            self.products_owners[product] = producer_id

            return True

        return False

    def new_cart(self):
        

        with self.lock_nr_carts:
            cart_id = self.nr_carts
            self.nr_carts += 1
            

        self.carts[cart_id] = []
        return cart_id

    def add_to_cart(self, cart_id, product):
        

        with self.lock_products_queue:
            
            if product in self.products:

                
                self.products.remove(product)

                
                producer_id = self.products_owners[product]
                self.nr_prod_in_queue[producer_id] -= 1

                
                self.carts[cart_id].append(product)

                return True

        return False

    def remove_from_cart(self, cart_id, product):
        

        


        self.carts[cart_id].remove(product)
        
        self.products.append(product)

        with self.lock_products_queue:
            
            producer_id = self.products_owners[product]
            self.nr_prod_in_queue[producer_id] += 1

    def place_order(self, cart_id):
        

        products_list = self.carts.pop(cart_id, None)
        order = self.carts.pop(cart_id, None)
        for product in products_list:
            print(currentThread().getName(), "bought", product)

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
            
            

            for (product, quantity, wait_time) in self.products:
                added_products = 0

                while added_products < quantity:
                    ret = self.marketplace.publish(self.producer_id, product)
                    if ret:
                        time.sleep(wait_time)
                        added_products += 1
                    else:
                        time.sleep(self.republish_wait_time)
