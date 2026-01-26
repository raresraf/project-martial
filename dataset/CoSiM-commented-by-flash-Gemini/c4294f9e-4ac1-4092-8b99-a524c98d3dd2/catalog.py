

from collections.abc import MutableMapping
from threading import RLock


class Catalog():
    
    def __init__(self, max_elems):
        
        self.lock = RLock()
        self.inventory = {}
        self.max_elems = max_elems
        self.size = 0

    def add_product(self, product):
        
        with self.lock:
            if self.size == self.max_elems:
                return False
            try:
                tup = self.inventory[product]
                (count, frozen) = tup
                self.inventory[product] = (count + 1, frozen)
            except KeyError:
                self.inventory[product] = (1, 0)
            self.size += 1
        return True

    def order_product(self, product):
        
        with self.lock:
            (count, frozen) = self.inventory[product]
            self.inventory[product] = (count, frozen - 1)
            self.size -= 1

    def free_product(self, product):
        
        with self.lock:
            if product not in self.inventory:
                return False
            (count, frozen) = self.inventory[product]
            self.inventory[product] = (count + 1, frozen - 1)
            return True

    def reserve_product(self, product):
        
        with self.lock:
            if product not in self.inventory:
                with open("inventory.txt", "w") as f:
                    f.write(str(product) + " not found in " +
                            str(self.inventory))
                return False
            (count, frozen) = self.inventory[product]
            if count == 0:
                return False
            self.inventory[product] = (count - 1, frozen + 1)
        return True


from threading import Thread
from time import sleep
from tema.marketplace import Marketplace


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        customer_id = self.marketplace.new_customers()
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                for _ in range(operation['quantity']):
                    sleep(self.retry_wait_time)
                    if operation['type'] == 'add':
                        finished = self.marketplace.add_to_cart(
                            cart_id, operation['product'])
                        while not finished:
                            sleep(self.retry_wait_time)
                            finished = self.marketplace.add_to_cart(
                                cart_id, operation['product'])
                    elif operation['type'] == 'remove':
                        self.marketplace.remove_from_cart(
                            cart_id, operation['product'])

            prods = self.marketplace.place_order(cart_id)
            for prod in prods:
                print("cons{} bought {}".format(customer_id, str(prod)))

import logging
import logging.handlers
from threading import RLock
from tema.catalog import Catalog


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        
        self.catalogs_lock = RLock()
        self.cartlock = RLock()
        self.customerslock = RLock()
        self.customeridlock = RLock()
        self.loglock = RLock()

        self.producers_catalogs = []
        self.carts = []
        self.queue_size_per_producer = queue_size_per_producer
        self.customers_active = 0
        self.customers_total = 0
        
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        handler = logging.handlers.RotatingFileHandler(
            "marketplace.log", backupCount=4, maxBytes=10000000)
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def customers_left(self):
        
        return self.customers_active > 0

    def new_customers(self):
        
        with self.customerslock:
            self.customers_total += 1
        return self.customers_total

    def register_producer(self):
        
        catalog = Catalog(self.queue_size_per_producer)
        with self.catalogs_lock:
            producer_id = len(self.producers_catalogs)
            self.producers_catalogs.append(catalog)



        self.log(f"{producer_id} = register_producer()")
        return producer_id

    def publish(self, producer_id, product):
        
        catalog = self.producers_catalogs[producer_id]
        ret = catalog.add_product(product)


        self.log(f"{ret} = publish({producer_id}, {product})")
        return ret

    def new_cart(self):
        

        with self.customerslock:
            self.customers_active += 1
        with self.cartlock:
            cart_id = len(self.carts)


            self.carts.append([])
        self.log(f"{cart_id} = new_cart()")
        return cart_id

    def add_to_cart(self, cart_id, product):
        
        cart = self.carts[cart_id]
        ret = False
        for catalog in self.producers_catalogs:
            ret = catalog.reserve_product(product)
            if ret:
                cart.append((product, catalog))
                break
        self.log(f"{ret} = add_to_cart({cart_id}, {product})")
        return ret

    def remove_from_cart(self, cart_id, product):
        

        cart = self.carts[cart_id]
        for (searched_product, catalog) in cart:
            if product == searched_product:
                self.log("{} == {}".format(product, searched_product))
                catalog.free_product(product)
                cart.remove((searched_product, catalog))
                break


        self.log(f"remove_from_cart({cart_id}, {product})")

    def place_order(self, cart_id):
        
        product_list = []
        cart = self.carts[cart_id]
        for (product, catalog) in cart:
            catalog.order_product(product)
            product_list.append(product)
        with self.customerslock:
            self.customers_active -= 1
        self.log(f"{product_list} = place_order({cart_id})")
        return product_list

    def log(self, message):
        
        pass
        
            


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        sleep(self.republish_wait_time)
        producer_id = self.marketplace.register_producer()
        while self.marketplace.customers_left():
            for bundle in self.products:
                product = bundle[0]
                quantity = bundle[1]
                wait_time = bundle[2]
                for _ in range(quantity):
                    finished = self.marketplace.publish(producer_id, product)
                    while not finished:
                        sleep(self.republish_wait_time)
                        finished = self.marketplace.publish(producer_id, product)
                    sleep(wait_time)


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
