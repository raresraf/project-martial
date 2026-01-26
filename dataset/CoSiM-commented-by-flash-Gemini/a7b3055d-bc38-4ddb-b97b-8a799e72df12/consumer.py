


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
                op_type = operation['type']
                wanted_product = operation['product']
                wanted_quantity = operation['quantity']

                current_quantity = 0

                
                while current_quantity < wanted_quantity:

                    can_do_op = None

                    
                    if op_type == "add":
                        can_do_op = self.marketplace.add_to_cart(cart_id, wanted_product)
                    elif op_type == "remove":
                        self.marketplace.remove_from_cart(cart_id, wanted_product)

                    if can_do_op is False:
                        
                        time.sleep(self.retry_wait_time)
                    else:
                        current_quantity += 1

            
            self.marketplace.place_order(cart_id)


from threading import currentThread, Lock
import time
import unittest
import logging
from logging.handlers import RotatingFileHandler

logging.basicConfig(
        handlers=[RotatingFileHandler('./marketplace.log', maxBytes=100000, backupCount=10,
                mode='a')],
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s")
logging.Formatter.converter = time.gmtime
logger = logging.getLogger()

class TestMarketplace(unittest.TestCase):

    

    @classmethod
    def setUpClass(cls):
        print('SetUpClass')

    def setUp(self):
        self.marketplace = Marketplace(15)
        self.prod1 = {  "product_type": "Coffee",
                        "name": "Indonezia",
                        "acidity": 5.05,
                        "roast_level": "MEDIUM",
                        "price": 1
                    }

        self.prod2 = {  "product_type": "Tea",
                        "name": "Linden",
                        "type": "Herbal",
                        "price": 9
                    }

    def test_register_producer(self):
        
        print('\nTest Register Producer\n')
        num_producers = 3
        res = -1

        for new_id in range(num_producers):
            res = self.marketplace.register_producer()
            self.assertEqual(res, new_id)

    def test_publish(self):
        

        print('\nTest Publish\n')
        pid = self.marketplace.register_producer()

        for _ in range(self.marketplace.queue_size_per_producer):
            res = self.marketplace.publish(pid, self.prod1)
            self.assertEqual(res, True)

        res = self.marketplace.publish(pid, self.prod2)
        self.assertEqual(res, False)


    def test_new_cart(self):
        
        print('\nTest New Cart\n')
        num_carts = 3
        tmp = -1

        for i in range(num_carts):
            tmp = self.marketplace.new_cart()
            self.assertEqual(tmp, i)

        self.assertEqual(tmp + 1, num_carts)


    def test_add_to_cart(self):
        
        print('\nTest Add\n')
        pid = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()

        for _ in range(2):
            self.marketplace.publish(pid, self.prod2)

        self.assertTrue(self.marketplace.add_to_cart(cart_id, self.prod2))
        self.assertFalse(self.marketplace.add_to_cart(cart_id, self.prod1))


    def test_remove_from_cart(self):
        
        print('\nTest Remove\n')
        pid = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        self.marketplace.producers[pid].extend([self.prod1, self.prod2, self.prod2])
        self.marketplace.carts[cart_id].extend([(self.prod1, pid), (self.prod2, pid)])

        prod1_occurences_prod = self.marketplace.producers[pid].count(self.prod1)
        prod1_occurences_cart = self.marketplace.carts[cart_id].count((self.prod1, pid))

        self.marketplace.remove_from_cart(cart_id, self.prod1)

        new_prod1_occurences_prod = self.marketplace.producers[pid].count(self.prod1)
        new_prod1_occurences_cart = self.marketplace.carts[cart_id].count((self.prod1, pid))

        self.assertGreater(new_prod1_occurences_prod, prod1_occurences_prod)
        self.assertLess(new_prod1_occurences_cart, prod1_occurences_cart)


    def test_place_order(self):
        
        print('\nTest Place Order\n')
        pid = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        expected_cart = [self.prod1, self.prod1, self.prod2]

        for _ in range(0, self.marketplace.queue_size_per_producer, 3):
            self.marketplace.publish(pid, self.prod2)
            self.marketplace.publish(pid, self.prod2)
            self.marketplace.publish(pid, self.prod1)

        self.marketplace.add_to_cart(cart_id, self.prod2)

        for _ in  range(3):
            self.marketplace.add_to_cart(cart_id, self.prod1)

        self.marketplace.remove_from_cart(cart_id, self.prod1)

        res = self.marketplace.place_order(cart_id)

        count_expected = expected_cart.count(self.prod1)
        count_res = res.count(self.prod1)

        self.assertEqual(count_expected, count_res)

        count_expected = expected_cart.count(self.prod2)
        count_res = res.count(self.prod2)

        self.assertEqual(count_expected, count_res)


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        

        self.queue_size_per_producer = queue_size_per_producer

        self.producers = {} 
        self.num_producers = 0 

        
        self.carts = {}
        self.num_carts = 0 

        
        self.producer_reg = Lock()
        self.cart_reg = Lock()
        self.print_res = Lock()
        self.add_prod = Lock()
        self.remove_prod = Lock()

        logging.info("Set up Marketplace with queue_size_per_producer = %s",
                    queue_size_per_producer)


    def register_producer(self):
        

        
        
        

        curr_producer_id = -1

        with self.producer_reg:
            curr_producer_id = self.num_producers
            self.producers[self.num_producers] = []
            self.num_producers += 1

            logging.info('Producer registered with id = %s', curr_producer_id)

        return curr_producer_id

    def publish(self, producer_id, product):
        

        
        

        logging.info("Producer with id = %s wants to publish product = %s", producer_id, product)

        if len(self.producers[producer_id]) == self.queue_size_per_producer:
            logging.info("Producer with id = %s can't publish %s", producer_id, product)
            return False

        self.producers[producer_id].append(product)
        logging.info("Producer with id = %s published product = %s", producer_id, product)

        return True


    def new_cart(self):
        

        
        
        
        

        curr_cart_id = -1

        with self.cart_reg:
            curr_cart_id = self.num_carts
            self.carts[self.num_carts] = []
            self.num_carts += 1

        logging.info("Consumer has cart registered cart with id = %s ", curr_cart_id)

        return curr_cart_id

    def add_to_cart(self, cart_id, product):
        

        
        
        

        with self.add_prod:
            logging.info("Consumer with cart_id = %s wants to add product = %s",
                        cart_id, product)

            for p_id in range(self.num_producers):

                if product in self.producers[p_id]:
                    logging.info(, cart_id, product, p_id)

                    self.producers[p_id].remove(product)
                    self.carts[cart_id].append((product, p_id))

                    return True

            return False

    def remove_from_cart(self, cart_id, product):
        

        
        
        
        

        with self.remove_prod:

            logging.info("Consumer  with cart_id = %s wants to remove product = %s",
                        cart_id, product)

            p_id = -1
            for elem in self.carts[cart_id]:
                if product == elem[0]:
                    self.producers[elem[1]].append(product)
                    p_id = elem[1]
                    logging.info(, cart_id, product, p_id)
                    break
            self.carts[cart_id].remove((product, p_id))


    def place_order(self, cart_id):
        

        
        

        bought_products = []

        with self.print_res:
            logging.info("Consumer with cart_id = %s place order", cart_id)

            for elem in self.carts[cart_id]:
                thread_name = currentThread().getName()
                product = elem[0]
                print(f"{thread_name} bought {product}")
                bought_products.append(elem[0])
            logging.info("Consumer with cart_id = %s has products %s",
                        cart_id, str(bought_products))

        return bought_products


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = -1

    def run(self):
        
        self.producer_id = self.marketplace.register_producer()



        while True:
            for (product, quantity, wait_time) in self.products:
                for _ in range(quantity):
                    can_publish = self.marketplace.publish(self.producer_id, product)

                    
                    if can_publish is True:
                        time.sleep(wait_time)
                    else:
                        time.sleep(self.republish_wait_time)


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
