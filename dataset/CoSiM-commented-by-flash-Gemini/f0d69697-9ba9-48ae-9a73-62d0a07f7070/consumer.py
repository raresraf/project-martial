


from threading import Thread
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.carts = carts

        
        self.marketplace = marketplace

        
        self.retry_wait_time = retry_wait_time

        
        self.kwargs = kwargs

    def run(self):
        num_of_carts = len(self.carts)
        my_carts = []

        
        for _ in range(0, num_of_carts):
            new_cart_id = self.marketplace.new_cart()
            my_carts.append(new_cart_id)

        
        for current_cart in self.carts:
            
            current_cart_id = my_carts.pop(0)

            
            for current_operation in current_cart:
                desired_quantity = current_operation["quantity"]
                current_quantity = 0

                
                while current_quantity < desired_quantity:
                    current_operation_type = current_operation["type"]
                    current_operation_product = current_operation["product"]

                    
                    if current_operation_type == "add":
                        current_operation_status = self.marketplace\
                            .add_to_cart(current_cart_id, current_operation_product)
                    else:
                        current_operation_status = self.marketplace \
                            .remove_from_cart(current_cart_id, current_operation_product)

                    
                    if current_operation_status is True or current_operation_status is None:
                        current_quantity = current_quantity + 1
                    else:
                        time.sleep(self.retry_wait_time)

            
            bought_products = self.marketplace.place_order(current_cart_id)
            for bought_product in bought_products:
                print(self.kwargs["name"] + " bought " + str(bought_product))

from threading import Lock

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        
        self.producer_of_product = {}

        
        self.queue_size_of_producer = {}
        self.queue_size_of_producer_lock = Lock()

        
        self.queue_size_per_producer = queue_size_per_producer

        
        self.carts = {}
        self.carts_lock = Lock()

        
        self.all_products = []

    def register_producer(self):
        

        
        self.queue_size_of_producer_lock.acquire()

        
        current_producers_number = len(self.queue_size_of_producer)
        self.queue_size_of_producer[current_producers_number] = 0

        
        self.queue_size_of_producer_lock.release()
        return current_producers_number

    def publish(self, producer_id, product):
        

        
        if self.queue_size_of_producer[producer_id] >= self.queue_size_per_producer:
            return False

        
        self.queue_size_of_producer_lock.acquire()

        
        self.queue_size_of_producer[producer_id] = self.queue_size_of_producer[producer_id] + 1
        self.producer_of_product[product] = producer_id
        self.all_products.append(product)

        
        self.queue_size_of_producer_lock.release()
        return True

    def new_cart(self):
        
        
        self.carts_lock.acquire()

        
        current_carts_number = len(self.carts)
        self.carts[current_carts_number] = []

        
        self.carts_lock.release()

        return current_carts_number

    def add_to_cart(self, cart_id, product):
        

        
        self.queue_size_of_producer_lock.acquire()

        
        if product not in self.all_products:
            self.queue_size_of_producer_lock.release()
            return False

        
        producer_id = self.producer_of_product[product]
        self.queue_size_of_producer[producer_id] = self.queue_size_of_producer[producer_id] - 1
        self.all_products.remove(product)

        
        self.queue_size_of_producer_lock.release()

        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        

        
        self.carts[cart_id].remove(product)
        self.all_products.append(product)
        producer_id = self.producer_of_product[product]

        
        self.queue_size_of_producer_lock.acquire()

        
        self.queue_size_of_producer[producer_id] = self.queue_size_of_producer[producer_id] + 1

        
        self.queue_size_of_producer_lock.release()

    def place_order(self, cart_id):
        

        
        bought_products = self.carts[cart_id]
        self.carts[cart_id] = []

        return bought_products


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.products = products

        
        self.marketplace = marketplace

        
        self.republish_wait_time = republish_wait_time

        
        self.kwargs = kwargs

        
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        
        while True:
            
            for current_product in self.products:
                current_product_type = current_product[0]
                current_product_quantity_desired = current_product[1]
                current_product_quantity = 0
                current_product_time_to_create = current_product[2]

                
                while current_product_quantity < current_product_quantity_desired:
                    current_transaction_status = self.marketplace\
                        .publish(self.producer_id, current_product_type)

                    
                    if current_transaction_status is True:
                        time.sleep(current_product_time_to_create)
                        current_product_quantity = current_product_quantity + 1
                    else:
                        time.sleep(self.republish_wait_time)
