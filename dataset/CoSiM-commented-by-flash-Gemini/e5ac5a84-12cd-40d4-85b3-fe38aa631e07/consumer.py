


from threading import Thread
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.retry_wait_time = retry_wait_time
        self.carts = carts
        self.marketplace = marketplace

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            operations_number = 0

            for operation in cart:
                while operations_number < operation["quantity"]:
                    if operation["type"] == "add":
                        add_to_cart = self.marketplace.add_to_cart(cart_id, operation["product"])
                        if not add_to_cart:
                            time.sleep(self.retry_wait_time)
                        else:
                            operations_number = operations_number + 1
                    else:
                        self.marketplace.remove_from_cart(cart_id, operation["product"])
                        operations_number = operations_number + 1
                operations_number = 0

            self.marketplace.place_order(cart_id)

from threading import Lock, currentThread
import logging
from logging.handlers import RotatingFileHandler


class Marketplace:
    

    
    
    def __init__(self, queue_size_per_producer):
        
        self.producers_ids = []
        self.producers_sizes = []
        self.carts_number = 0
        self.carts = []
        self.print_lock = Lock()
        self.max_elements_for_producer = queue_size_per_producer
        self.num_carts_lock = Lock()
        self.register_lock = Lock()
        self.sizes_lock = Lock()
        self.product_to_producer = {}
        self.products = []
        self.logger = logging.getLogger('marketplace')
        self.logger.setLevel(logging.INFO)
        log_form = logging\
            .Formatter('%(asctime)s 
        rotating_file_handler = RotatingFileHandler('marketplace.log', 'a', 16384)
        rotating_file_handler.setFormatter(log_form)


        self.logger.addHandler(rotating_file_handler)

    
    def register_producer(self):
        
        with self.register_lock:
            prod_id = len(self.producers_ids)
            self.producers_ids.append(prod_id)
            self.producers_sizes.append(0)


        self.logger.info("prod_id = %s", str(prod_id))
        return prod_id

    
    
    def publish(self, producer_id, product):
        
        self.logger.info("producer_id = %s product = %s", str(producer_id), str(product))
        prod_id = int(producer_id)

        for i in range(0, len(self.producers_ids)):
            if self.producers_ids[i] == prod_id:
                if self.producers_sizes[i] >= self.max_elements_for_producer:
                    return False
                self.producers_sizes[i] = self.producers_sizes[i] + 1

        self.products.append(product)
        self.product_to_producer[product] = prod_id
        self.logger.info("return_value = %s", "True")

        return True

    
    
    def new_cart(self):
        
        with self.num_carts_lock:
            self.carts_number = self.carts_number + 1
            cart_id = self.carts_number

        self.carts.append({"id": cart_id, "list": []})


        self.logger.info("cart_id = %s", str(cart_id))

        return cart_id

    
    
    
    def add_to_cart(self, cart_id, product):
        
        self.logger.info("cart_id = %s product = %s", str(cart_id), str(product))
        with self.sizes_lock:
            if product in self.products:
                prod_id = self.product_to_producer[product]
                for i in range(0, len(self.producers_ids)):
                    if self.producers_ids[i] == prod_id:
                        self.producers_sizes[i] = self.producers_sizes[i] - 1
                self.products.remove(product)
                cart = [x for x in self.carts if x["id"] == cart_id][0]
                cart["list"].append(product)
                return True
        self.logger.info("return_value = %s", "False")
        return False

    
    
    def remove_from_cart(self, cart_id, product):
        
        self.logger.info("cart_id = %s product = %s", str(cart_id), str(product))
        cart = [x for x in self.carts if x["id"] == cart_id][0]
        cart["list"].remove(product)
        self.products.append(product)

        with self.sizes_lock:
            prod_id = self.product_to_producer[product]

            for i in range(0, len(self.producers_ids)):
                if self.producers_ids[i] == prod_id:
                    self.producers_sizes[i] = self.producers_sizes[i] - 1

    
    def place_order(self, cart_id):
        
        self.logger.info("cart_id = %s", str(cart_id))
        cart = [x for x in self.carts if x["id"] == cart_id][0]
        self.carts.remove(cart)
        for product in cart["list"]:
            with self.print_lock:
                print("{} bought {}".format(currentThread().getName(), product))
        self.logger.info("cart_items = %s", str(cart["list"]))

        return cart["list"]


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.republish_wait_time = republish_wait_time
        self.products = products
        self.marketplace = marketplace
        self.prod_id = self.marketplace.register_producer()

    def run(self):
        while True:
            for (product, number_products, time_sleep) in self.products:

                for i in range(number_products):
                    if self.marketplace.publish(str(self.prod_id), product):
                        time.sleep(time_sleep)
                    else:
                        time.sleep(self.republish_wait_time)
                        i -= 1


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
