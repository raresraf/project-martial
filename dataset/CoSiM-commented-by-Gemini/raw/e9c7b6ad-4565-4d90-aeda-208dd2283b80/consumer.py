"""
Module consumer.py
Provides core algorithm logic. Employs optimal time complexity routines.
"""



from threading import Thread, Lock
from typing import List, Dict
from time import sleep
from tema.marketplace import Marketplace


class Consumer(Thread):
    
    print_lock = Lock()

    def __init__(self, carts: List[List[Dict]], marketplace: Marketplace,
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
                 retry_wait_time: float, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
        for op_list in self.carts:
            cart_id = self.marketplace.new_cart()
            # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
            for operation in op_list:
                # Block Logic: Conditional state gate. Invariant: Execution paths are mutually exclusive.
                if operation['type'] == 'add':
                    # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
                    for _ in range(operation['quantity']):
                        retval = self.marketplace.add_to_cart(cart_id, operation['product'])
                        # Block Logic: State evaluation loop. Invariant: Loop boundaries remain safe.
                        while not retval:
                            sleep(self.retry_wait_time)
                            retval = self.marketplace.add_to_cart(cart_id, operation['product'])
                elif operation['type'] == 'remove':
                    # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
                    for _ in range(operation['quantity']):
                        self.marketplace.remove_from_cart(cart_id, operation['product'])

            msg = "\n".join([f'{self.name} bought {str(prod)}'
                             # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
                             for prod in self.marketplace.place_order(cart_id)])
            Consumer.print_lock.acquire()
            print(msg)
            Consumer.print_lock.release()

from logging import Logger, Formatter
from logging.handlers import RotatingFileHandler
import time
from threading import Lock
from typing import Dict, List, Tuple
import unittest
from tema.product import Product, Tea, Coffee


class Marketplace:
    

    def __init__(self, queue_size_per_producer: int):
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        
        
        self.queue_size_per_producer: int
        
        self.producer_id_generator_lock: Lock
        
        self.producer_count: int
        
        self.cart_id_generator_lock: Lock
        
        self.cart_count: int
        
        self.carts: Dict[int, List[Tuple[Product, str]]]
        
        
        self.products: Dict[str, Tuple[Lock, List[Tuple[bool, Product]]]]
        
        self.logger = Logger("marketplace logger", level="INFO")
        fmt = Formatter(fmt='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        fmt.converter = time.gmtime
        rfh = RotatingFileHandler("marketplace.log", delay=True)
        rfh.formatter = fmt
        self.logger.addHandler(rfh)
        
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id_generator_lock = Lock()
        self.producer_count = 0
        self.cart_id_generator_lock = Lock()
        self.cart_count = 0
        self.carts = {}
        self.products = {}

    def register_producer(self):
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        
        self.logger.info('entry: register_producer')
        
        self.producer_id_generator_lock.acquire()
        producer_id = str(self.producer_count)
        self.producer_count += 1
        self.producer_id_generator_lock.release()
        
        self.products[producer_id] = (Lock(), [])
        self.logger.info('exit: register_producer')
        return producer_id

    def publish(self, producer_id: str, product: Product):
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        
        self.logger.info('enter: publish %s %s', producer_id, product)
        lock, plist = self.products[producer_id]
        lock.acquire()  
        # Block Logic: Conditional state gate. Invariant: Execution paths are mutually exclusive.
        if len(plist) == self.queue_size_per_producer:
            lock.release()
            self.logger.info('exit_fail: publish %s %s - queue full', producer_id, product)
            return False
        plist.append((False, product))


        lock.release()
        self.logger.info('exit_success: publish %s %s', producer_id, product)
        return True

    def new_cart(self):
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        
        self.logger.info('enter: new_cart')
        
        self.cart_id_generator_lock.acquire()
        cart_id = self.cart_count
        self.cart_count += 1
        self.cart_id_generator_lock.release()
        
        self.carts[cart_id] = []
        self.logger.info('enter: new_cart')
        return cart_id

    def add_to_cart(self, cart_id: int, product: Product):
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        
        self.logger.info('enter: add_to_cart %d %s', cart_id, product)
        # Block Logic: Conditional state gate. Invariant: Execution paths are mutually exclusive.
        if cart_id not in self.carts:
            self.logger.info('exit_fail: add_to_cart %d %s - cart not found', cart_id, product)
            return False

        # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
        for producer_id, (lock, plist) in self.products.items():
            lock.acquire()  
            # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
            for idx, (reserved, prod) in enumerate(plist):
                # Block Logic: Conditional state gate. Invariant: Execution paths are mutually exclusive.
                if not reserved and prod == product:
                    plist[idx] = (True, prod)  
                    self.carts[cart_id].append((prod, producer_id))  
                    lock.release()  
                    self.logger.info('exit_success: add_to_cart %d %s', cart_id, product)
                    return True
            lock.release()  
        self.logger.info('exit_fail: add_to_cart %d %s', cart_id, product)
        return False

    def remove_from_cart(self, cart_id: int, product: Product):
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        
        self.logger.info('enter: remove_from_cart %d %s', cart_id, product)
        # Block Logic: Conditional state gate. Invariant: Execution paths are mutually exclusive.
        if cart_id not in self.carts:
            self.logger.info('exit_fail: remove_from_cart %d %s - cart not found', cart_id, product)
            return
        
        # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
        for prod, prod_id in self.carts[cart_id]:
            # Block Logic: Conditional state gate. Invariant: Execution paths are mutually exclusive.
            if prod == product:
                lock, lst = self.products[prod_id]
                
                lock.acquire()
                # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
                for idx, (reserved, listed_product) in enumerate(lst):
                    # Block Logic: Conditional state gate. Invariant: Execution paths are mutually exclusive.
                    if reserved and listed_product == product:
                        self.products[prod_id][1][idx] = False, listed_product
                        self.carts[cart_id].remove((product, prod_id))
                        lock.release()
                        self.logger.info('exit_success: remove_from_cart %d %s', cart_id, product)
                        return
                lock.release()
                self.logger.info('exit_fail: remove_from_cart %d %s'
                                 ' - product in cart but not in producer list', cart_id, product)
                return
        self.logger.info('exit_fail: remove_from_cart %d %s - product not found', cart_id, product)

    def place_order(self, cart_id: int):
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        
        self.logger.info('enter: place_order %d', cart_id)
        # Block Logic: Conditional state gate. Invariant: Execution paths are mutually exclusive.
        if cart_id not in self.carts:
            self.logger.info('exit_fail: place_order %d - cart not found', cart_id)
            return None
        products = []
        
        # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
        for product, producer in self.carts[cart_id]:
            lock, lst = self.products[producer]
            lock.acquire()
            lst.remove((True, product))
            lock.release()
            products.append(product)
        self.carts.pop(cart_id)
        self.logger.info('exit: place_order %d', cart_id)
        return products


class TestMarketplace(unittest.TestCase):
    

    def setUp(self) -> None:
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        self.marketplace = Marketplace(10)

    def tearDown(self):
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        
        
        self.marketplace.logger.handlers[0].close()

    def test_register_producer(self):
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        
        previous_ids = []
        # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
        for _ in range(1000):
            new_id = self.marketplace.register_producer()
            self.assertNotIn(new_id, previous_ids)
            previous_ids.append(new_id)

    def test_publish(self):
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        
        
        marketplace1 = Marketplace(5)
        marketplace2 = self.marketplace
        marketplace3 = Marketplace(100)
        product = Tea(name='Yorkshire Tea', price=14, type='Black')
        
        producers1 = [marketplace1.register_producer() for _ in range(10)]
        producers2 = [marketplace2.register_producer() for _ in range(50)]
        producers3 = [marketplace3.register_producer() for _ in range(1000)]
        
        # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
        for _ in range(5):
            # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
            for producer in producers1:
                self.assertTrue(marketplace1.publish(producer, product))
        # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
        for producer in producers1:
            self.assertFalse(marketplace1.publish(producer, product))
        
        # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
        for _ in range(10):
            # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
            for producer in producers2:
                self.assertTrue(marketplace2.publish(producer, product))
        # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
        for producer in producers2:
            self.assertFalse(marketplace2.publish(producer, product))
        
        # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
        for _ in range(100):
            # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
            for producer in producers3:
                self.assertTrue(marketplace3.publish(producer, product))
        # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
        for producer in producers1:
            self.assertFalse(marketplace1.publish(producer, product))
        
        marketplace1.logger.handlers[0].close()
        marketplace3.logger.handlers[0].close()

    def test_new_cart(self):
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        
        previous_ids = []
        # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
        for _ in range(1000):
            new_id = self.marketplace.new_cart()
            self.assertNotIn(new_id, previous_ids)
            previous_ids.append(new_id)

    def test_add_to_cart(self):
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        
        product1 = Tea(name='Yorkshire Tea', price=14, type='Black')
        product2 = Tea(name='Sencha', price=22, type='Green')
        product3 = Coffee(name='Indonesia', price=1, acidity="5.05", roast_level="MEDIUM")
        producer1 = self.marketplace.register_producer()
        producer2 = self.marketplace.register_producer()
        cart1 = self.marketplace.new_cart()
        cart2 = self.marketplace.new_cart()

        def fail_add(cart, product):
            """
            Executes core routine.
            @pre: Valid inputs provided.
            @post: Desired state transformation complete.
            """
            self.assertFalse(self.marketplace.add_to_cart(cart, product))

        def succeed_add(cart, product):
            """
            Executes core routine.
            @pre: Valid inputs provided.
            @post: Desired state transformation complete.
            """
            self.assertTrue(self.marketplace.add_to_cart(cart, product))

        
        fail_add(cart1, product1)
        fail_add(cart1, product2)
        fail_add(cart2, product3)
        
        self.marketplace.publish(producer1, product1)
        
        fail_add(cart1, product2)
        fail_add(cart2, product3)
        
        succeed_add(cart1, product1)
        fail_add(cart1, product1)
        
        fail_add(cart2, product1)
        
        self.marketplace.publish(producer1, product1)
        self.marketplace.publish(producer2, product1)
        
        succeed_add(cart1, product1)
        succeed_add(cart2, product1)
        
        fail_add(cart1, product1)
        fail_add(cart2, product1)
        
        self.marketplace.publish(producer1, product1)
        self.marketplace.publish(producer2, product1)
        self.marketplace.publish(producer1, product2)
        self.marketplace.publish(producer2, product2)
        self.marketplace.publish(producer1, product3)
        self.marketplace.publish(producer2, product3)
        
        succeed_add(cart1, product1)
        succeed_add(cart1, product1)
        succeed_add(cart1, product2)
        succeed_add(cart1, product2)
        succeed_add(cart1, product3)
        succeed_add(cart1, product3)
        
        fail_add(cart1, product1)
        fail_add(cart2, product1)
        fail_add(cart1, product2)
        fail_add(cart2, product2)
        fail_add(cart1, product3)
        fail_add(cart2, product3)

    def test_remove_from_cart(self):
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        
        producer = self.marketplace.register_producer()
        product1 = Tea(name='Yorkshire Tea', price=14, type='Black')
        product2 = Tea(name='Sencha', price=22, type='Green')
        self.marketplace.publish(producer, product1)
        cart1 = self.marketplace.new_cart()
        cart2 = self.marketplace.new_cart()

        def fail_add(cart, product):
            """
            Executes core routine.
            @pre: Valid inputs provided.
            @post: Desired state transformation complete.
            """
            self.assertFalse(self.marketplace.add_to_cart(cart, product))

        def succeed_add(cart, product):
            """
            Executes core routine.
            @pre: Valid inputs provided.
            @post: Desired state transformation complete.
            """
            self.assertTrue(self.marketplace.add_to_cart(cart, product))

        
        succeed_add(cart1, product1)
        fail_add(cart2, product1)
        self.marketplace.remove_from_cart(cart1, product1)
        succeed_add(cart2, product1)
        
        

        
        self.marketplace.publish(producer, product1)
        self.marketplace.publish(producer, product1)
        self.marketplace.publish(producer, product2)
        self.marketplace.publish(producer, product2)
        
        succeed_add(cart1, product1)
        succeed_add(cart2, product1)
        succeed_add(cart1, product2)
        succeed_add(cart2, product2)
        
        fail_add(cart1, product1)
        fail_add(cart2, product1)
        fail_add(cart1, product2)
        fail_add(cart2, product2)
        
        
        self.marketplace.remove_from_cart(cart1, product1)
        self.marketplace.remove_from_cart(cart1, product1)
        
        succeed_add(cart2, product1)
        fail_add(cart1, product1)
        fail_add(cart2, product1)
        fail_add(cart1, product2)
        fail_add(cart2, product2)

    def test_place_order(self):
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        
        producer = self.marketplace.register_producer()

        product1 = Tea(name='Yorkshire Tea', price=14, type='Black')
        product2 = Tea(name='Sencha', price=22, type='Green')
        product3 = Coffee(name='Indonesia', price=1, acidity="5.05", roast_level="MEDIUM")

        
        # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
        for _ in range(3):
            self.marketplace.publish(producer, product1)
            self.marketplace.publish(producer, product2)
            self.marketplace.publish(producer, product3)

        cart1 = self.marketplace.new_cart()
        cart2 = self.marketplace.new_cart()

        # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
        for _ in range(2):
            self.marketplace.add_to_cart(cart1, product1)
            self.marketplace.add_to_cart(cart1, product2)
            self.marketplace.add_to_cart(cart1, product3)

        

        # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
        for _ in range(2):
            self.marketplace.add_to_cart(cart2, product1)
            self.marketplace.add_to_cart(cart2, product2)
            self.marketplace.add_to_cart(cart2, product3)

        

        # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
        for _ in range(3):
            self.marketplace.remove_from_cart(cart1, product1)

        self.marketplace.remove_from_cart(cart1, product2)

        

        self.marketplace.add_to_cart(cart2, product1)

        

        

        cart1prod = self.marketplace.place_order(cart1)
        cart2prod = self.marketplace.place_order(cart2)

        cart1counts = {product1: 0, product2: 0, product3: 0}
        cart2counts = {product1: 0, product2: 0, product3: 0}

        ref1 = {product1: 0, product2: 1, product3: 2}
        ref2 = {product1: 2, product2: 1, product3: 1}

        # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
        for prod in cart1prod:
            cart1counts[prod] += 1

        # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
        for prod in cart2prod:
            cart2counts[prod] += 1

        self.assertDictEqual(cart1counts, ref1)
        self.assertDictEqual(cart2counts, ref2)


from threading import Thread
from typing import List
from time import sleep
from tema.marketplace import Marketplace
from tema.product import Product


class Producer(Thread):
    

    def __init__(self, products: List[Product], marketplace: Marketplace,
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
                 republish_wait_time: float, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.marketplace = marketplace
        self.products = products
        self.republish_wait_time = republish_wait_time
        self.producer_id = marketplace.register_producer()

    def run(self):
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        # Block Logic: State evaluation loop. Invariant: Loop boundaries remain safe.
        while True:
            # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
            for prod, quant, time in self.products:
                # Block Logic: Iterates over collection. Invariant: Processes elements sequentially.
                for _ in range(quant):
                    ret_val = self.marketplace.publish(self.producer_id, prod)
                    # Block Logic: State evaluation loop. Invariant: Loop boundaries remain safe.
                    while not ret_val:
                        sleep(self.republish_wait_time)
                        ret_val = self.marketplace.publish(self.producer_id, prod)
                    sleep(time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    
    type: str

    def __eq__(self, other):
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        return isinstance(other, Tea) \
            and self.name == other.name \
            and self.price == other.price \
            and self.type == other.type


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    
    acidity: str
    roast_level: str

    def __eq__(self, other):
        """
        Executes core routine.
        @pre: Valid inputs provided.
        @post: Desired state transformation complete.
        """
        return isinstance(other, Coffee) \
            and self.name == other.name \
            and self.price == other.price \
            and self.acidity == other.acidity \
            and self.roast_level == other.roast_level
