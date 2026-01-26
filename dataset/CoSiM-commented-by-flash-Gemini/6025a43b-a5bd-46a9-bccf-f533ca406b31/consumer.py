


from threading import Lock, Thread
from time import sleep

def print_products(consumer_name, products):
    

    for product in products:
        print("{} bought {}".format(consumer_name, product))

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        

        
        name = kwargs["name"]

        
        super().__init__()

        
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = name

        self.print_lock = Lock()

    def run(self):
        
        for cart_operations in self.carts:
            
            cart_id = self.marketplace.new_cart()

            
            for cart_operation in cart_operations:
                operation_type = cart_operation["type"]
                operation_product = cart_operation["product"]
                operation_cnt = cart_operation["quantity"]

                
                for _ in range(operation_cnt):
                    if operation_type == "add":
                        added = False

                        
                        while True:
                            added = self.marketplace.add_to_cart(cart_id, operation_product)

                            if not added:
                                sleep(self.retry_wait_time)
                            else:
                                break
                    elif operation_type == "remove":
                        self.marketplace.remove_from_cart(cart_id, operation_product)
                    else:
                        raise Exception("Unknown op: cart {}, cons {}".format(cart_id, self.name))

            ordered_products = self.marketplace.place_order(cart_id)

            
            with self.print_lock:
                print_products(self.name, ordered_products)


from threading import Lock

class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        

        
        self.queue_size_per_producer = queue_size_per_producer
        
        self.producer_queues = {}
        
        self.producer_queue_lock = Lock()
        


        self.producer_next_id = 0
        
        self.producer_id_generator_lock = Lock()

        
        self.carts = {}
        
        self.cart_next_id = 0
        
        self.cart_id_generator_lock = Lock()

    def register_producer(self):
        

        
        with self.producer_id_generator_lock:
            
            producer_id = self.producer_next_id

            
            self.producer_queues[producer_id] = []

            
            self.producer_next_id += 1

        return producer_id

    def publish(self, producer_id, product):
        

        
        producer_queue = self.producer_queues[producer_id]

        
        with self.producer_queue_lock:
            
            if len(producer_queue) < self.queue_size_per_producer:
                
                producer_queue.append(product)
                
                return True

        
        return False


    def new_cart(self):
        

        
        with self.cart_id_generator_lock:
            
            cart_id = self.cart_next_id

            
            self.carts[cart_id] = Cart()

            
            self.cart_next_id += 1

            return cart_id

    def add_to_cart(self, cart_id, product):
        

        
        no_producers = 0

        
        with self.producer_id_generator_lock:
            no_producers = self.producer_next_id

        
        for producer_id in range(no_producers):
            producer_stock = self.producer_queues[producer_id]

            
            if product in producer_stock:
                
                with self.producer_queue_lock:
                    
                    producer_stock.remove(product)

                
                self.carts[cart_id].add_product(product, producer_id)

                
                return True

        
        return False

    def remove_from_cart(self, cart_id, product):
        

        
        producer_id = self.carts[cart_id].remove_product(product)

        
        with self.producer_queue_lock:
            
            
            producer_queue = self.producer_queues[producer_id]

            if len(producer_queue) < self.queue_size_per_producer:
                
                producer_queue.append(product)


    def place_order(self, cart_id):
        
        return self.carts[cart_id].get_products()

class Cart:
    

    def __init__(self):
        

        
        self.products = []

    def add_product(self, product, producer_id):
        

        self.products.append({"product": product, "producer_id": producer_id})

    def remove_product(self, product):
        

        
        for prod in self.products:

            
            if prod["product"] == product:
                
                producer_id = prod["producer_id"]

                
                self.products.remove(prod)

                
                return producer_id

        return None

    def get_products(self):
        

        product_list = []

        for product_item in self.products:
            product_list.append(product_item["product"])

        return product_list


from threading import Thread
from time import sleep

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        

        
        Thread.__init__(self, daemon=kwargs["daemon"])

        
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.name = kwargs["name"]

    def run(self):
        
        producer_id = self.marketplace.register_producer()

        while True:
            for product in self.products:
                
                product_id = product[0]
                product_quantity = product[1]
                product_production_time = product[2]

                
                sleep(product_production_time)

                for _ in range(product_quantity):
                    produced = False

                    while True:
                        
                        produced = self.marketplace.publish(producer_id, product_id)

                        
                        if not produced:
                            sleep(self.republish_wait_time)
                        else:
                            break
