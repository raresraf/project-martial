


from threading import Thread


from time import sleep

from tema.marketplace import Marketplace


class Consumer(Thread):
    

    def __init__(self, carts: list, marketplace: Marketplace, retry_wait_time: int, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for operation in cart:
                function = self.marketplace.add_to_cart if operation['type'] == 'add' \
                    else self.marketplace.remove_from_cart

                for _ in range(operation['quantity']):
                    while function(cart_id, operation['product']) is False:
                        sleep(self.retry_wait_time)

            product_list = self.marketplace.place_order(cart_id)

            if len(product_list) > 0:
                print("\n".join([f"{self.name} bought {product}" for product in product_list]))


from logging.handlers import RotatingFileHandler
from multiprocessing import Lock
import unittest
import logging
import time
from tema.product import Product, Coffee, Tea


logging.Formatter.converter = time.gmtime
ROTATING_FILE = RotatingFileHandler(filename='marketplace.log', maxBytes=1048576,
                                    backupCount=5)
ROTATING_FILE.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
LOGGER = logging.getLogger('marketplace')
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(ROTATING_FILE)


class Marketplace:
    
    def __init__(self, queue_size_per_producer: int):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.producers = 0
        self.producers_queue = {}
        self.lock = Lock()
        self.carts = {}
        self.cart_ids = 0
        self.products = []


    def register_producer(self):
        
        LOGGER.info('Registering a producer')

        with self.lock:
            self.producers += 1
            producer_id = self.producers
            self.producers_queue[producer_id] = 0

        LOGGER.info('Producer registered with id %s', producer_id)

        return producer_id


    def publish(self, producer_id: str, product: Product):
        
        LOGGER.info('Producer %s is publishing a %s product', producer_id, product)

        acquired = self.lock.acquire(timeout=0.5)

        if not acquired or self.producers_queue[producer_id] >= self.queue_size_per_producer:
            self.lock.release()
            LOGGER.info('Producer %s was not able to publish a %s product', producer_id, product)
            return False

        self.products.append((product, producer_id))
        self.producers_queue[producer_id] += 1
        self.lock.release()

        LOGGER.info('Producer %s published a %s product', producer_id, product)

        return True


    def new_cart(self):
        

        LOGGER.info('Creating a new cart')

        with self.lock:
            self.cart_ids += 1
            cart_id = self.cart_ids
            self.carts[cart_id] = []

        LOGGER.info('Cart was created with id %s', cart_id)

        return cart_id


    def add_to_cart(self, cart_id: int, product: Product):
        

        LOGGER.info('Adding a %s product to cart %s', product, cart_id)

        with self.lock:
            try:
                product_index = list(
                    map(lambda product_tuple: product_tuple[0], self.products)
                ).index(product)
            except ValueError:
                LOGGER.info('Product %s was not found in the marketplace', product)
                return False


            if cart_id not in self.carts:
                LOGGER.info('Cart %s was not found in the marketplace', cart_id)
                return False

            self.producers_queue[self.products[product_index][1]] -= 1
            self.carts[cart_id].append((product, self.products[product_index][1]))
            del self.products[product_index]

        LOGGER.info('Product %s was added to cart %s', product, cart_id)

        return True


    def remove_from_cart(self, cart_id: int, product: Product):
        

        LOGGER.info('Removing a %s product from cart %s', product, cart_id)

        with self.lock:
            cart_product_list = list(
                map(lambda product_tuple: product_tuple[0], self.carts[cart_id])
            )
            if cart_id not in self.carts or product not in cart_product_list:
                LOGGER.info('Product %s was not found in cart %s', product, cart_id)
                return False

            try:
                product_index = cart_product_list.index(product)
            except ValueError:
                LOGGER.info('Product %s was not found in cart %s', product, cart_id)
                return False

            self.products.append(self.carts[cart_id][product_index])
            self.producers_queue[self.carts[cart_id][product_index][1]] += 1
            del self.carts[cart_id][product_index]

        LOGGER.info('Product %s was removed from cart %s', product, cart_id)

        return True




    def place_order(self, cart_id: int):
        

        LOGGER.info('Placing order for cart %s', cart_id)

        with self.lock:
            if cart_id not in self.carts:
                LOGGER.info('Cart %s was not found in the marketplace', cart_id)
                return None

            products = self.carts[cart_id]
            del self.carts[cart_id]

        LOGGER.info('Order was placed for cart %s', cart_id)

        return list(map(lambda x: x[0], products))


class TestMarketplace(unittest.TestCase):
    

    def setUp(self) -> None:
        super().setUp()
        self.marketplace = Marketplace(queue_size_per_producer=5)

    def test_register_producer(self):
        

        
        self.assertEqual(self.marketplace.register_producer(), 1)
        self.assertEqual(self.marketplace.register_producer(), 2)
        self.assertEqual(self.marketplace.register_producer(), 3)


    def test_publish(self):
        

        
        producer_id = self.marketplace.register_producer()

        
        self.assertEqual(self.marketplace.publish(producer_id, Product("product1", 10)), True)
        self.assertEqual(self.marketplace.publish(producer_id, Product("product2", 10)), True)
        self.assertEqual(self.marketplace.publish(producer_id, Product("product3", 10)), True)
        self.assertEqual(self.marketplace.publish(producer_id, Product("product4", 10)), True)
        self.assertEqual(self.marketplace.publish(producer_id, Product("product5", 10)), True)

        
        self.assertEqual(self.marketplace.publish(producer_id, Product("product6", 10)), False)
        self.assertEqual(self.marketplace.publish(producer_id, Product("product7", 10)), False)
        self.assertEqual(self.marketplace.publish(producer_id, Product("product8", 10)), False)


    def test_new_cart(self):
        

        
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)
        self.assertEqual(self.marketplace.new_cart(), 3)


    def test_add_to_cart(self):
        

        
        producer_id = self.marketplace.register_producer()

        self.marketplace.publish(producer_id, Coffee("Espresso", 10, 0.2, "MEDIUM"))
        self.marketplace.publish(producer_id, Tea("Green Tea", 10, "Indian"))

        
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, Coffee("Espresso", 10, 0.2, "MEDIUM"))
        self.assertEqual(len(self.marketplace.products), 1)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 1)

        
        self.assertEqual(self.marketplace.add_to_cart(1000, Tea("Green Tea", 10, "Indian")), False)

        
        self.assertEqual(self.marketplace.add_to_cart(
            cart_id,
            Tea("Black Tea", 10, "Indian")), False)


    def test_remove_from_cart(self):
        

        
        producer_id = self.marketplace.register_producer()

        self.marketplace.publish(producer_id, Coffee("Espresso", 10, 0.2, "MEDIUM"))
        self.marketplace.publish(producer_id, Tea("Green Tea", 10, "Indian"))

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, Coffee("Espresso", 10, 0.2, "MEDIUM"))
        self.marketplace.add_to_cart(cart_id, Tea("Green Tea", 10, "Indian"))

        
        self.assertEqual(len(self.marketplace.carts[cart_id]), 2)
        self.assertEqual(self.marketplace.remove_from_cart(
            cart_id,
            Coffee("Espresso", 10, 0.2, "MEDIUM")), True)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 1)

        
        self.assertEqual(self.marketplace.remove_from_cart(
            1000,
            Tea("Green Tea", 10, "Indian")), False)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 1)

        
        self.assertEqual(self.marketplace.remove_from_cart(
            cart_id,
            Coffee("Espresso", 10, 0.2, "MEDIUM")), False)
        self.assertEqual(len(self.marketplace.carts[cart_id]), 1)


    def test_place_order(self):
        
        producer_id = self.marketplace.register_producer()

        self.marketplace.publish(producer_id, Coffee("Espresso", 10, 0.2, "MEDIUM"))
        self.marketplace.publish(producer_id, Tea("Green Tea", 10, "Indian"))
        self.marketplace.publish(producer_id, Tea("Black Tea", 15, "Indian"))

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, Coffee("Espresso", 10, 0.2, "MEDIUM"))
        self.marketplace.add_to_cart(cart_id, Tea("Green Tea", 10, "Indian"))

        self.assertEqual(
            self.marketplace.place_order(cart_id),
            [Coffee("Espresso", 10, 0.2, "MEDIUM"), Tea("Green Tea", 10, "Indian")]
        )

        self.assertEqual(self.marketplace.place_order(cart_id), None)

        self.assertEqual(len(self.marketplace.carts), 0)


from threading import Thread


from time import sleep

from tema.marketplace import Marketplace


class Producer(Thread):
    

    def __init__(self, products: list, marketplace: Marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        producer_id = self.marketplace.register_producer()

        while True:
            for (product, quantity, produce_time) in self.products:
                for _ in range(quantity):
                    sleep(produce_time)
                    while self.marketplace.publish(producer_id, product) is False:
                        sleep(self.republish_wait_time)


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
