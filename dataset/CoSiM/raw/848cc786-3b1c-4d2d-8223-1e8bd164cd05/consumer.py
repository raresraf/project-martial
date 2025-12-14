


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def print_purchased_products(self, purchased_products):
        
        for product in purchased_products:
            print(self.getName(), "bought", product)

    def add_to_cart(self, cart_id, product, quantity):
        
        for _ in range(quantity):
            is_added = self.marketplace.add_to_cart(cart_id, product)
            while not is_added:
                sleep(self.retry_wait_time)
                is_added = self.marketplace.add_to_cart(cart_id, product)

    def remove_to_cart(self, cart_id, product, quantity):
        
        for _ in range(quantity):
            self.marketplace.remove_from_cart(cart_id, product)

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                op_type = operation["type"]
                product = operation["product"]
                quantity = operation["quantity"]
                if op_type == "add":
                    self.add_to_cart(cart_id, product, quantity)
                elif op_type == "remove":
                    self.remove_to_cart(cart_id, product, quantity)
            purchased_products = self.marketplace.place_order(cart_id)
            self.print_purchased_products(purchased_products)

import unittest
import logging
import time
from threading import Lock
from logging import handlers
from tema.product import Coffee, Tea

class TestMarketplace(unittest.TestCase):
    def setUp(self):
        
        self.marketplace = Marketplace(5)
        self.coffee_1 = Coffee(name='Indonezia', price=1, acidity='5.05', roast_level='MEDIUM')
        self.coffee_2 = Coffee(name='Brasil', price=7, acidity='5.09', roast_level='MEDIUM')
        self.tea_1 = Tea(name='Wild Cherry', price=5, type='Black')
        self.tea_2 = Tea(name='Jasmine', price=6, type='Green')

    def test_register_producer(self):
        
        self.assertIsInstance(self.marketplace.register_producer(),
                              int, "Return value should be int")
        self.assertNotEqual(self.marketplace.register_producer(), 0, "id should be 0")
        self.assertNotEqual(self.marketplace.register_producer(), 1, "id should be 1")
        self.assertNotEqual(self.marketplace.register_producer(), 2, "id should be 2")
        id_1 = self.marketplace.register_producer()
        id_2 = self.marketplace.register_producer()
        id_3 = self.marketplace.register_producer()
        id_4 = self.marketplace.register_producer()
        ids = [id_2, id_3, id_4]
        self.assertNotIn(id_1, ids, "id should be unique")

    def test_new_cart(self):
        
        self.assertIsInstance(self.marketplace.new_cart(), int, "Return value should be int")
        self.assertNotEqual(self.marketplace.new_cart(), 0, "cart_id should be 0")
        self.assertNotEqual(self.marketplace.new_cart(), 1, "cart_id should be 1")
        self.assertNotEqual(self.marketplace.new_cart(), 2, "cart_id should be 2")
        cart_id_1 = self.marketplace.new_cart()
        cart_id_2 = self.marketplace.new_cart()
        cart_id_3 = self.marketplace.new_cart()
        cart_id_4 = self.marketplace.new_cart()
        cart_ids = [cart_id_2, cart_id_3, cart_id_4]
        self.assertNotIn(cart_id_1, cart_ids, "id should be unique")

    def test_publish(self):
        
        id_1 = self.marketplace.register_producer()
        id_2 = self.marketplace.register_producer()



        self.marketplace.publish(id_1, self.coffee_1)
        self.marketplace.publish(id_1, self.coffee_2)
        self.marketplace.publish(id_2, self.tea_1)
        self.assertEqual(len(self.marketplace.products[id_1]), 2, "Len should be 2")
        self.assertEqual(len(self.marketplace.products[id_2]), 1, "Len should be 1")

        ret = self.marketplace.publish(id_1, self.tea_1)
        self.assertTrue(ret, "Return should be True")
        ret = self.marketplace.publish(id_1, self.tea_1)
        self.assertTrue(ret, "Return should be True")
        self.marketplace.publish(id_1, self.tea_2)
        self.assertEqual(len(self.marketplace.products[id_1]), 5, "Len should be 5")

        ret = self.marketplace.publish(id_1, self.tea_2)
        self.assertEqual(len(self.marketplace.products[id_1]), 5, "Len should be 5")
        self.assertFalse(ret, "Return should be false")

        self.assertEqual(self.marketplace.products[id_1][0],
                         self.coffee_1,
                         "Product should be coffee_1")
        self.assertEqual(self.marketplace.products[id_1][1],
                         self.coffee_2,
                         "Product should be coffee_2")
        self.assertEqual(self.marketplace.products[id_2][0],
                         self.tea_1,
                         "Product should be tea_1")

        self.assertFalse(self.marketplace.publish(3, self.tea_1), "Return should be False")

    def test_add_to_cart(self):
        
        producer_id_1 = self.marketplace.register_producer()
        producer_id_2 = self.marketplace.register_producer()

        self.marketplace.publish(producer_id_1, self.coffee_1)
        self.marketplace.publish(producer_id_1, self.coffee_2)

        cart_id_1 = self.marketplace.new_cart()
        cart_id_2 = self.marketplace.new_cart()

        ret = self.marketplace.add_to_cart(cart_id_1, self.coffee_1)
        self.assertTrue(ret, "Return should be True")
        self.assertEqual(self.marketplace.carts[cart_id_1][0][0],
                         producer_id_1,
                         "Producer_id should be producer_id_1")
        self.assertEqual(self.marketplace.carts[cart_id_1][0][1],
                         self.coffee_1,
                         "Product should be coffee_1")

        ret = self.marketplace.add_to_cart(cart_id_1, self.coffee_1)
        self.assertFalse(ret, "Return should be False")

        ret = self.marketplace.add_to_cart(cart_id_2, self.coffee_2)
        self.assertTrue(ret, "Return should be True")
        self.assertEqual(self.marketplace.carts[cart_id_2][0][0],
                         producer_id_1, "Producer_id should be producer_id_1")
        self.assertEqual(self.marketplace.carts[cart_id_2][0][1],
                         self.coffee_2, "Product should be coffee_2")

        self.marketplace.publish(producer_id_1, self.coffee_1)
        self.marketplace.publish(producer_id_2, self.tea_1)
        self.marketplace.publish(producer_id_1, self.tea_2)



        self.marketplace.add_to_cart(cart_id_1, self.coffee_1)
        self.marketplace.add_to_cart(cart_id_1, self.tea_1)
        self.marketplace.add_to_cart(cart_id_1, self.tea_2)

        self.assertNotEqual(len(self.marketplace.carts[cart_id_1]), 5, "Len should be 5")

    def test_get_product_index(self):
        
        producer_id_1 = self.marketplace.register_producer()


        producer_id_2 = self.marketplace.register_producer()

        cart_id_1 = self.marketplace.new_cart()
        cart_id_2 = self.marketplace.new_cart()

        self.marketplace.publish(producer_id_1, self.coffee_1)
        self.marketplace.publish(producer_id_1, self.coffee_2)
        self.marketplace.publish(producer_id_1, self.tea_1)
        self.marketplace.publish(producer_id_1, self.tea_2)

        self.marketplace.publish(producer_id_2, self.coffee_1)
        self.marketplace.publish(producer_id_2, self.coffee_2)
        self.marketplace.publish(producer_id_2, self.tea_1)
        self.marketplace.publish(producer_id_2, self.tea_2)

        self.marketplace.add_to_cart(cart_id_2, self.coffee_1)
        self.marketplace.add_to_cart(cart_id_2, self.coffee_2)
        self.marketplace.add_to_cart(cart_id_2, self.tea_1)
        self.marketplace.add_to_cart(cart_id_2, self.tea_2)

        self.marketplace.add_to_cart(cart_id_1, self.tea_1)
        self.marketplace.add_to_cart(cart_id_1, self.coffee_1)
        self.marketplace.add_to_cart(cart_id_1, self.tea_2)
        self.marketplace.add_to_cart(cart_id_1, self.coffee_2)

        self.assertEqual(self.marketplace.get_product_index(cart_id_2, self.coffee_1),
                         0, "Index should be 1")
        self.assertEqual(self.marketplace.get_product_index(cart_id_2, self.tea_1),
                         2, "Index should be 2")
        self.assertEqual(self.marketplace.get_product_index(cart_id_2, self.tea_2),
                         3, "Index should be 3")

        self.assertEqual(self.marketplace.get_product_index(cart_id_1, self.coffee_1),
                         1, "Index should be 1")
        self.assertEqual(self.marketplace.get_product_index(cart_id_1, self.tea_1),
                         0, "Index should be 0")
        self.assertEqual(self.marketplace.get_product_index(cart_id_1, self.tea_2),
                         2, "Index should be 2")

    def test_remove_from_cart(self):
        
        producer_id_1 = self.marketplace.register_producer()
        producer_id_2 = self.marketplace.register_producer()

        cart_id_1 = self.marketplace.new_cart()
        cart_id_2 = self.marketplace.new_cart()

        self.marketplace.publish(producer_id_1, self.coffee_1)
        self.marketplace.publish(producer_id_1, self.coffee_2)
        self.marketplace.publish(producer_id_1, self.tea_1)
        self.marketplace.publish(producer_id_1, self.tea_2)

        self.marketplace.publish(producer_id_2, self.coffee_1)
        self.marketplace.publish(producer_id_2, self.coffee_2)
        self.marketplace.publish(producer_id_2, self.tea_1)
        self.marketplace.publish(producer_id_2, self.tea_2)

        self.marketplace.add_to_cart(cart_id_2, self.coffee_1)
        self.marketplace.add_to_cart(cart_id_2, self.coffee_2)
        self.marketplace.add_to_cart(cart_id_2, self.tea_1)
        self.marketplace.add_to_cart(cart_id_2, self.tea_2)

        self.marketplace.remove_from_cart(cart_id_2, self.coffee_1)
        self.assertEqual(self.marketplace.get_product_index(cart_id_2, self.coffee_1),
                         -1, "Index should be -1")
        self.marketplace.remove_from_cart(cart_id_2, self.tea_1)
        self.assertEqual(self.marketplace.get_product_index(cart_id_2, self.tea_1),
                         -1, "Index should be -1")
        self.assertEqual(self.marketplace.get_product_index(cart_id_2, self.coffee_2),
                         0, "Index should be 0")
        self.assertEqual(self.marketplace.get_product_index(cart_id_2, self.tea_2),
                         1, "Index should be 1")

        self.marketplace.add_to_cart(cart_id_1, self.tea_1)
        self.marketplace.add_to_cart(cart_id_1, self.coffee_1)
        self.marketplace.add_to_cart(cart_id_1, self.tea_2)
        self.marketplace.add_to_cart(cart_id_1, self.coffee_2)

        self.marketplace.remove_from_cart(cart_id_1, self.tea_1)
        self.marketplace.remove_from_cart(cart_id_1, self.tea_2)
        self.assertEqual(self.marketplace.get_product_index(cart_id_1, self.tea_1),
                         -1, "Index should be -1")
        self.assertEqual(self.marketplace.get_product_index(cart_id_1, self.tea_2),
                         -1, "Index should be -1")

    def test_place_order(self):
        
        producer_id_1 = self.marketplace.register_producer()

        cart_id_1 = self.marketplace.new_cart()
        cart_id_2 = self.marketplace.new_cart()

        for _ in range(2):
            self.marketplace.publish(producer_id_1, self.coffee_1)
            self.marketplace.publish(producer_id_1, self.coffee_2)
            self.marketplace.publish(producer_id_1, self.tea_1)
            self.marketplace.publish(producer_id_1, self.tea_2)

        self.marketplace.add_to_cart(cart_id_1, self.tea_1)
        self.marketplace.add_to_cart(cart_id_1, self.coffee_1)
        self.marketplace.add_to_cart(cart_id_1, self.tea_2)
        self.marketplace.add_to_cart(cart_id_1, self.coffee_2)

        self.marketplace.add_to_cart(cart_id_2, self.coffee_1)
        self.marketplace.add_to_cart(cart_id_2, self.coffee_1)


        self.marketplace.add_to_cart(cart_id_2, self.coffee_1)
        self.marketplace.add_to_cart(cart_id_2, self.coffee_2)

        purchased_products_cart_1 = self.marketplace.place_order(cart_id_1)
        purchased_products_cart_2 = self.marketplace.place_order(cart_id_2)
        purchased_products_cart_3 = self.marketplace.place_order(5)

        self.assertEqual(purchased_products_cart_3, [], "List should be []")
        self.assertEqual(len(purchased_products_cart_1), 4, "Len should be 4")
        self.assertEqual(len(purchased_products_cart_2), 1, "Len should be 1")

        purchased_products_cart_1 = self.marketplace.place_order(cart_id_1)
        self.assertEqual(len(purchased_products_cart_1), 0, "Len should be 0")


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.logger = logging.Logger('logger')
        self.logger.addHandler(handlers.RotatingFileHandler('marketplace.log',
                                                            maxBytes=50000,
                                                            backupCount=10))
        get_time = time.strftime('%m/%d/%Y %I:%M:%S %p', time.gmtime())
        self.logger.info(get_time + ": Marketplace init with queue_size_per_producer = " +
                         str(queue_size_per_producer))

        self.max_queue = queue_size_per_producer
        self.producers_lock = Lock()
        self.consumers_lock = Lock()
        self.products = {}
        self.carts = {}
        self.producers_id = 0
        self.cart_id = 0

    def register_producer(self):
        
        self.producers_lock.acquire()
        get_time = time.strftime('%m/%d/%Y %I:%M:%S %p', time.gmtime())
        self.logger.info(get_time + ": Marketplace register_producer() call")

        id = self.producers_id
        self.producers_id += 1
        self.products[id] = []

        get_time = time.strftime('%m/%d/%Y %I:%M:%S %p', time.gmtime())
        self.logger.info(get_time + ": Marketplace register_producer() call return " + str(id))
        self.producers_lock.release()
        return id

    def publish(self, producer_id, product):
        
        get_time = time.strftime('%m/%d/%Y %I:%M:%S %p', time.gmtime())
        self.logger.info(
            get_time + ": Marketplace publish() call with producer_id = " +
            str(producer_id) + " product = " + str(
                product))
        if producer_id > self.producers_id:
            get_time = time.strftime('%m/%d/%Y %I:%M:%S %p', time.gmtime())
            self.logger.info(get_time + ": Marketplace publish() call return False")
            return False
        self.producers_lock.acquire()
        if len(self.products[producer_id]) >= self.max_queue:
            ret = False
        else:
            self.products[producer_id].append(product)
            ret = True
        get_time = time.strftime('%m/%d/%Y %I:%M:%S %p', time.gmtime())
        self.logger.info(get_time + ": Marketplace publish() call return " + str(ret))
        self.producers_lock.release()
        return ret

    def new_cart(self):
        
        self.consumers_lock.acquire()
        get_time = time.strftime('%m/%d/%Y %I:%M:%S %p', time.gmtime())
        self.logger.info(get_time + ": Marketplace new_cart() call")

        id = self.cart_id
        self.cart_id += 1
        self.carts[id] = []

        get_time = time.strftime('%m/%d/%Y %I:%M:%S %p', time.gmtime())
        self.logger.info(get_time + ": Marketplace new_cart() call return " + str(id))
        self.consumers_lock.release()
        return id

    def add_to_cart(self, cart_id, product):
        

        self.consumers_lock.acquire()
        get_time = time.strftime('%m/%d/%Y %I:%M:%S %p', time.gmtime())
        self.logger.info(
            get_time + ": Marketplace add_to_cart() call with cart_id = " + str(cart_id) +
            " product = " + str(product))

        ret = False
        for producer in self.products:
            if product in self.products[producer]:
                index = self.products[producer].index(product)
                add_product = self.products[producer].pop(index)
                self.carts[cart_id].append((producer, add_product))
                ret = True
                break

        get_time = time.strftime('%m/%d/%Y %I:%M:%S %p', time.gmtime())
        self.logger.info(get_time + ": Marketplace add_to_cart() call return " + str(ret))

        self.consumers_lock.release()
        return ret

    def get_product_index(self, cart_id, product):
        index = 0
        for entry in self.carts[cart_id]:
            if entry[1] == product:
                return index
            index += 1
        return -1

    def remove_from_cart(self, cart_id, product):
        
        self.consumers_lock.acquire()
        self.producers_lock.acquire()
        get_time = time.strftime('%m/%d/%Y %I:%M:%S %p', time.gmtime())
        self.logger.info(
            get_time + ": Marketplace remove_from_cart() call with cart_id = " + str(cart_id) +
            " product = " + str(
                product))

        product_index = self.get_product_index(cart_id, product)
        if product_index != -1:
            entry = self.carts[cart_id].pop(product_index)
            self.products[entry[0]].append(entry[1])

        get_time = time.strftime('%m/%d/%Y %I:%M:%S %p', time.gmtime())
        self.logger.info(get_time + ": Marketplace remove_from_cart() call return")

        self.producers_lock.release()
        self.consumers_lock.release()

    def place_order(self, cart_id):
        
        get_time = time.strftime('%m/%d/%Y %I:%M:%S %p', time.gmtime())
        self.logger.info(get_time + ": Marketplace place_order() call with cart_id = " +
                         str(cart_id))

        if cart_id > self.cart_id:
            return []
        purchased_products = []
        for product in self.carts[cart_id]:
            purchased_products.append(product[1])
        self.carts[cart_id] = []

        get_time = time.strftime('%m/%d/%Y %I:%M:%S %p', time.gmtime())
        self.logger.info(get_time + ": Marketplace place_order() call return")
        return purchased_products


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        idx = 0
        list_size = len(self.products)
        while True:
            product = self.products[idx % list_size][0]
            quantity = self.products[idx % list_size][1]
            prod_time = self.products[idx % list_size][2]
            for _ in range(quantity):
                sleep(prod_time)
                is_published = self.marketplace.publish(self.producer_id, product)
                while not is_published:
                    sleep(self.wait_time)
                    is_published = self.marketplace.publish(self.producer_id, product)
            idx += 1


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
