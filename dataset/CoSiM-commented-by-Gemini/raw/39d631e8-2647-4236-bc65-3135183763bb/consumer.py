


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        
        for crt_cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for crt_operation in crt_cart:

                number_of_operations = 0
                while number_of_operations < crt_operation["quantity"]:

                    op_product = crt_operation["product"]

                    
                    if crt_operation["type"] == "add":
                        return_code = self.marketplace.add_to_cart(cart_id, op_product)
                    elif crt_operation["type"] == "remove":
                        return_code = self.marketplace.remove_from_cart(cart_id, op_product)

                    if return_code == True or return_code is None:
                        number_of_operations += 1
                    else:
                        time.sleep(self.retry_wait_time)

            
            self.marketplace.place_order(cart_id)


from threading import Lock, currentThread


class Marketplace:
    
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.sizes_per_producer = [] 

        self.carts = {} 


        self.number_of_carts = 0

        self.products = [] 
        self.producers = {} 

        self.lock_for_sizes = Lock() 
        self.lock_for_carts = Lock() 
        self.lock_for_register = Lock() 
        self.lock_for_print = Lock() 

    def register_producer(self):
        
        with self.lock_for_register:
            producer_id = len(self.sizes_per_producer)
        self.sizes_per_producer.append(0)
        return producer_id

    def publish(self, producer_id, product):
        

        num_prod_id = int(producer_id)

        max_size = self.queue_size_per_producer
        crt_size = self.sizes_per_producer[num_prod_id]

        if crt_size >= max_size:
            return False

        with self.lock_for_sizes:
            self.sizes_per_producer[num_prod_id] += 1
        self.products.append(product)
        self.producers[product] = num_prod_id

        return True

    def new_cart(self):
        
        ret_id = 0
        with self.lock_for_carts:
            self.number_of_carts += 1
            ret_id = self.number_of_carts

        self.carts[ret_id] = []

        return ret_id

    def add_to_cart(self, cart_id, product):
        
        with self.lock_for_sizes:


            if product not in self.products:
                return False

            self.products.remove(product)

            producer = self.producers[product]
            self.sizes_per_producer[producer] -= 1

        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        
        self.carts[cart_id].remove(product)

        with self.lock_for_sizes:


            producer = self.producers[product]
            self.sizes_per_producer[producer] += 1

        self.products.append(product)

    def place_order(self, cart_id):
        

        product_list = self.carts.pop(cart_id, None)

        for prod in product_list:
            with self.lock_for_print:
                print(str(currentThread().getName()) + " bought " + str(prod))

        return product_list


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
        
        while 69 - 420 < 3:

            for (product, number_of_products, product_wait_time) in self.products:

                i = 0
                while i < number_of_products:
                    return_code = self.marketplace.publish(str(self.producer_id), product)

                    if not return_code: 
                        time.sleep(self.republish_wait_time)
                    else:
                        time.sleep(product_wait_time)
                        i += 1


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
