


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def print_order(self, products):
        
        output = ""
        for product in products:
            output += self.name + ' bought ' + str(product) + '\n'
        output = output[:-1]
        print(output)

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for item in cart:
                action = item['type']
                product = item['product']
                quantity = item['quantity']


                for _ in range(quantity):
                    if action == 'add':
                        ret_value = self.marketplace.add_to_cart(cart_id, product)
                        while not ret_value:
                            time.sleep(self.retry_wait_time)
                            ret_value = self.marketplace.add_to_cart(cart_id, product)
                    elif action == 'remove':
                        self.marketplace.remove_from_cart(cart_id, product)
            products = self.marketplace.place_order(cart_id)
            self.print_order(products)

import logging.handlers
import time
import unittest
from threading import Semaphore
from tema.product import Tea, Coffee


class Booth:
    
    def __init__(self, producer):
        self.producer = producer
        self.num_products = 0
        self.num_products_mutex = Semaphore(1)

    def __eq__(self, other):
        if not isinstance(other, Booth):
            return False
        return self.producer == other.producer


class Cart:
    
    def __init__(self, cart_id):
        self.cart_id = cart_id
        self.products = []

    def __eq__(self, other):
        if not isinstance(other, Cart):
            return False
        return self.cart_id == other.cart_id


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        handler = logging.handlers.RotatingFileHandler(filename='marketplace.log',
                                                       mode='a',
                                                       maxBytes=10000,
                                                       backupCount=1)
        logging.Formatter.converter = time.gmtime
        logging.basicConfig(
            handlers=[handler],
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%d-%m-%Y %H:%M:%S')
        logging.basicConfig(
            handlers=[handler],
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.ERROR,
            datefmt='%d-%m-%Y %H:%M:%S')
        self.queue_size_per_producer = queue_size_per_producer
        self.producers = {}
        self.num_producers = 0
        self.carts = {}
        self.num_carts = 0
        self.products = []
        self.shopping_mutex = Semaphore(1)
        self.carts_mutex = Semaphore(1)
        self.register_mutex = Semaphore(1)

    def register_producer(self):
        
        logging.info('A producer wants to register')

        
        self.register_mutex.acquire()
        
        producer_id = self.num_producers
        self.producers[producer_id] = Booth(producer_id)
        self.num_producers += 1
        self.register_mutex.release()

        logging.info('A producer registered with id = ' + str(producer_id))
        return str(producer_id)

    def publish(self, producer_id, product):
        
        logging.info('Producer ' + producer_id + ' wants to publish product: ' + str(product))

        
        pid = int(producer_id)
        booth = self.producers[pid]
        booth.num_products_mutex.acquire()
        if booth.num_products < self.queue_size_per_producer:
            
            self.shopping_mutex.acquire()
            
            self.products.append((product, pid))
            booth.num_products += 1
            self.shopping_mutex.release()
            booth.num_products_mutex.release()
            logging.info('Producer ' + producer_id
                         + ' published product ' + str(product) + ' successfully')
            return True

        
        logging.error('Producer ' + producer_id
                      + ' could not publish product, because its queue is full')
        booth.num_products_mutex.release()
        return False

    def new_cart(self):
        
        logging.info('A consumer wants a new cart')

        
        self.carts_mutex.acquire()
        
        cart_id = self.num_carts
        self.carts[cart_id] = Cart(cart_id)
        self.num_carts += 1
        self.carts_mutex.release()

        logging.info('The consumer got a new cart')
        return cart_id

    def add_to_cart(self, cart_id, product):
        
        logging.info('Consumer with cart id ' + str(cart_id)
                     + ' wants to add to cart the product ' + str(product))

        
        self.shopping_mutex.acquire()
        for i in range(len(self.products)):
            if product == self.products[i][0]:
                
                self.carts[cart_id].products.append(self.products[i])
                self.products = self.products[:i] + self.products[i+1:]
                self.shopping_mutex.release()
                logging.info('Consumer with cart id ' + str(cart_id)
                             + ' added to cart the product ' + str(product) + ' successfully')
                return True

        
        self.shopping_mutex.release()
        logging.error('Consumer with cart id ' + str(cart_id)
                      + ' could not add to cart the product ' + str(product))
        return False

    def remove_from_cart(self, cart_id, product):
        
        logging.info('Consumer with cart id ' + str(cart_id)
                     + ' wants to remove the product ' + str(product) + ' from the cart')

        
        self.shopping_mutex.acquire()
        for i in range(len(self.carts[cart_id].products)):
            if product == self.carts[cart_id].products[i][0]:
                
                self.products.append(self.carts[cart_id].products[i])
                self.carts[cart_id].products = self.carts[cart_id].products[:i] \
                                               + self.carts[cart_id].products[i+1:]
                self.shopping_mutex.release()

                logging.info('Consumer with cart id ' + str(cart_id)
                             + ' removed product ' + str(product) + ' from the cart')
                break

    def place_order(self, cart_id):
        
        logging.info('Consumer with cart id ' + str(cart_id) + ' wants to place the order')

        cart = self.carts[cart_id].products
        products = []
        for (product, producer_id) in cart:
            
            products.append(product)
            
            
            booth = self.producers[producer_id]
            booth.num_products_mutex.acquire()
            booth.num_products -= 1
            booth.num_products_mutex.release()

        
        del self.carts[cart_id]

        logging.info('Consumer with cart id ' + str(cart_id) + ' placed the order successfully')
        return products


class TestMarketplace(unittest.TestCase):
    
    def setUp(self) -> None:
        self.marketplace = Marketplace(3)

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), '0')
        self.assertEqual(self.marketplace.register_producer(), '1')
        self.assertEqual(self.marketplace.register_producer(), '2')
        self.assertEqual(self.marketplace.num_producers, 3)
        self.assertEqual(self.marketplace.producers, {0: Booth(0), 1: Booth(1), 2: Booth(2)})

    def test_publish(self):
        
        p_1 = Tea(name='Linden', type='Herbal', price=9)
        pid = self.marketplace.register_producer()
        products = [(p_1, int(pid)), (p_1, int(pid)), (p_1, int(pid))]

        self.assertEqual(self.marketplace.publish(pid, p_1), True)
        self.assertEqual(self.marketplace.publish(pid, p_1), True)
        self.assertEqual(self.marketplace.publish(pid, p_1), True)
        self.assertEqual(self.marketplace.publish(pid, p_1), False)
        self.assertEqual(self.marketplace.producers[int(pid)].num_products, 3)
        self.assertEqual(self.marketplace.products, products)

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)
        self.assertEqual(self.marketplace.num_carts, 3)
        self.assertEqual(self.marketplace.carts, {0: Cart(0), 1: Cart(1), 2: Cart(2)})

    def test_add_to_cart(self):
        
        p_1 = Tea(name='Linden', type='Herbal', price=9)
        p_2 = Coffee(name='Indonezia', acidity='5.05', roast_level='MEDIUM', price=1)

        pid = self.marketplace.register_producer()
        self.marketplace.publish(pid, p_1)

        cart_id = self.marketplace.new_cart()
        self.assertEqual(self.marketplace.add_to_cart(cart_id, p_1), True)
        self.assertEqual(self.marketplace.add_to_cart(cart_id, p_1), False)
        self.assertEqual(self.marketplace.add_to_cart(cart_id, p_2), False)

        products = [(p_1, int(pid))]
        self.assertEqual(self.marketplace.carts[cart_id].products, products)
        self.assertEqual(self.marketplace.products, [])

    def test_remove_from_cart(self):
        
        p_1 = Tea(name='Linden', type='Herbal', price=9)
        p_2 = Coffee(name='Indonezia', acidity='5.05', roast_level='MEDIUM', price=1)

        pid = self.marketplace.register_producer()
        self.marketplace.publish(pid, p_1)


        self.marketplace.publish(pid, p_1)
        self.marketplace.publish(pid, p_2)

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, p_1)
        self.marketplace.add_to_cart(cart_id, p_1)
        self.marketplace.add_to_cart(cart_id, p_2)

        products = [(p_1, int(pid)), (p_2, int(pid))]
        self.marketplace.remove_from_cart(cart_id, p_1)
        self.assertEqual(self.marketplace.carts[cart_id].products, products)
        self.assertEqual(self.marketplace.products, [(p_1, int(pid))])

    def test_place_order(self):
        
        p_1 = Tea(name='Linden', type='Herbal', price=9)
        p_2 = Coffee(name='Indonezia', acidity='5.05', roast_level='MEDIUM', price=1)

        pid = self.marketplace.register_producer()
        self.marketplace.publish(pid, p_1)
        self.marketplace.publish(pid, p_1)
        self.marketplace.publish(pid, p_2)

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, p_1)
        self.marketplace.add_to_cart(cart_id, p_1)
        self.marketplace.add_to_cart(cart_id, p_2)

        self.assertEqual(self.marketplace.place_order(cart_id), [p_1, p_1, p_2])
        self.assertEqual(self.marketplace.producers[int(pid)].num_products, 0)
        self.assertEqual(self.marketplace.carts, {})


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        pid = self.marketplace.register_producer()
        while True:
            for item in self.products:
                product = item[0]
                quantity = item[1]
                waiting = item[2]
                for _ in range(quantity):


                    ret_value = self.marketplace.publish(pid, product)
                    while not ret_value:
                        time.sleep(self.republish_wait_time)
                        ret_value = self.marketplace.publish(pid, product)
                    time.sleep(waiting)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    
    name: str
    price: int

    def __eq__(self, other):
        pass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    
    type: str

    def __eq__(self, other):
        if not isinstance(other, Tea):
            return False
        return other.name == self.name and \
                    other.price == self.price and \
                    other.type == self.type


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    
    acidity: str
    roast_level: str

    def __eq__(self, other):
        if not isinstance(other, Coffee):
            return False
        return other.name == self.name and \
                    other.price == self.price and \
                    other.acidity == self.acidity and \
                    other.roast_level == self.roast_level
