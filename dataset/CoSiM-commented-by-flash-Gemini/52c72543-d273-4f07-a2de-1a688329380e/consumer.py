


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
                no_ops = 0

                
                qty = operation["quantity"]
                op_type = operation["type"]
                prod = operation["product"]

                
                
                
                while no_ops < qty:


                    result = self.execute_operation(cart_id, op_type, prod)

                    if result is None or result:
                        no_ops += 1
                    else:
                        time.sleep(self.retry_wait_time)

            self.marketplace.place_order(cart_id)

    def execute_operation(self, cart_id, operation_type, product) -> bool:
        
        if operation_type == "add":
            return self.marketplace.add_to_cart(cart_id, product)

        if operation_type == "remove":
            return self.marketplace.remove_from_cart(cart_id, product)

        return False


from threading import Lock, currentThread


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        

        self.queue_size_per_producer = queue_size_per_producer
        self.products_mapping = {}  
        self.producers_queues = []  
        self.consumers_carts = {}  


        self.available_products = []  
        
        
        self.no_carts = 0

        self.consumer_cart_creation_lock = Lock()
        self.cart_operation_lock = Lock()

    def register_producer(self):
        
        new_producer_id = len(self.producers_queues)

        self.producers_queues.append(0)

        return new_producer_id

    def publish(self, producer_id, product):
        

        if self.producers_queues[producer_id] >= self.queue_size_per_producer:
            return False

        self.producers_queues[producer_id] += 1
        self.available_products.append(product)

        self.products_mapping[product] = producer_id

        return True

    def new_cart(self):
        
        with self.consumer_cart_creation_lock:
            self.no_carts += 1

            self.consumers_carts[self.no_carts] = []

            return self.no_carts

    def add_to_cart(self, cart_id, product):
        
        with self.cart_operation_lock:
            if product not in self.available_products:
                return False

            producer_id = self.products_mapping[product]
            self.producers_queues[producer_id] -= 1

            self.available_products.remove(product)

            self.consumers_carts[cart_id].append(product)

            return True

    def remove_from_cart(self, cart_id, product):
        



        self.consumers_carts[cart_id].remove(product)
        self.available_products.append(product)

        with self.cart_operation_lock:

            producer_id = self.products_mapping[product]
            self.producers_queues[producer_id] += 1

    def place_order(self, cart_id):
        
        products = self.consumers_carts.pop(cart_id, None)

        for product in products:
            print(currentThread().getName() + " bought " + str(product))

        return products


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
            for (product, no_products, publish_wait_time) in self.products:
                no_prod = 0

                while no_prod < no_products:
                    result = self.publish_product(product, publish_wait_time)

                    if result:
                        no_prod += 1

    def publish_product(self, product, publish_wait_time) -> bool:
        
        result = self.marketplace.publish(self.producer_id, product)

        
        
        if result:
            time.sleep(publish_wait_time)
            return True

        time.sleep(self.republish_wait_time)
        return False
