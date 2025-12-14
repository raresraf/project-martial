


import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        


        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.operations = {"add": marketplace.add_to_cart,
                           "remove": marketplace.remove_from_cart}

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for operation in cart:
                quantity = operation["quantity"]

                while quantity > 0:
                    operation_type = operation["type"]
                    product = operation["product"]

                    if self.operations[operation_type](cart_id, product) is not False:
                        quantity -= 1
                    else:
                        time.sleep(self.retry_wait_time)

            self.marketplace.place_order(cart_id)

import sys
import logging
import time
from threading import Lock, currentThread
from logging.handlers import RotatingFileHandler


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.carts_lock = Lock()
        self.carts = []

        self.producers_lock = Lock()
        self.producers_capacity = queue_size_per_producer
        self.producers_sizes = []
        self.products = []

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s : %(message)s')
        formatter.converter = time.gmtime

        file_handler = RotatingFileHandler(
            "marketplace.log", maxBytes=4096, backupCount=0)
        file_handler.setFormatter(formatter)

        logger = logging.getLogger("marketplace")
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        self.logger = logger

    def register_producer(self):
        
        self.logger.info("enter register_producer()")

        with self.producers_lock:
            self.producers_sizes.append(0)
            self.logger.info("leave register_producer")
            return len(self.producers_sizes) - 1

    def publish(self, producer_id, product):
        
        self.logger.info(
            "enter publish(%d, %s)", producer_id, str(product))

        with self.producers_lock:
            if self.producers_sizes[producer_id] == self.producers_capacity:
                self.logger.info("leave publish")
                return False

            self.producers_sizes[producer_id] += 1
            self.products.append((product, producer_id))
            self.logger.info("leave publish")
            return True

    def new_cart(self):
        
        self.logger.info("enter new_cart()")
        with self.carts_lock:
            self.carts.append([])
            self.logger.info("leave new_cart")
            return len(self.carts) - 1

    def add_to_cart(self, cart_id, product):
        
        self.logger.info(
            "enter add_to_cart(%d, %s)", cart_id, str(product))



        self.producers_lock.acquire()
        for (prod, prod_id) in self.products:
            if prod == product:
                self.producers_sizes[prod_id] -= 1
                self.products.remove((prod, prod_id))
                self.producers_lock.release()
                self.carts[cart_id].append((prod, prod_id))
                self.logger.info("leave add_to_cart")
                return True

        self.producers_lock.release()
        self.logger.info("leave add_to_cart")
        return False

    def remove_from_cart(self, cart_id, product):
        
        self.logger.info("enter remove_from_cart(%d, %s)", cart_id, str(product))



        for (prod, prod_id) in self.carts[cart_id]:
            if prod == product:
                self.carts[cart_id].remove((prod, prod_id))
                self.producers_lock.acquire()
                self.products.append((prod, prod_id))
                self.producers_sizes[prod_id] += 1
                self.producers_lock.release()
                self.logger.info("leave remove_from_cart")
                return

    def place_order(self, cart_id):
        
        self.logger.info("enter place_order(%d)", cart_id)



        order = ""
        for (product, _) in self.carts[cart_id]:
            order += "{} bought {}\n".format(
                currentThread().getName(), product)
        sys.stdout.write(order)
        self.logger.info("leave place_order")
        return self.carts[cart_id]


import time
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time


        self.producer_id = marketplace.register_producer()

    def run(self):
        while True:
            for (product, quantity, wait_time) in self.products:
                while quantity > 0:
                    if self.marketplace.publish(self.producer_id, product):
                        quantity -= 1
                        time.sleep(wait_time / 20)
                    else:
                        time.sleep(self.republish_wait_time / 20)
