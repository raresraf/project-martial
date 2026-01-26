


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.name = kwargs["name"]
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()

            
            for cart_op in cart:
                quantity = cart_op.get("quantity")

                
                if cart_op.get("type") == "add":
                    
                    while quantity > 0:
                        
                        while not self.marketplace.add_to_cart(cart_id, cart_op.get("product")):
                            time.sleep(self.retry_wait_time)
                        quantity -= 1
                elif cart_op.get("type") == "remove":
                    
                    while quantity > 0:
                        self.marketplace.remove_from_cart(cart_id, cart_op.get("product"))
                        quantity -= 1

            
            for product in self.marketplace.place_order(cart_id):
                print(f"{self.name} bought {product}")


from threading import Lock
import unittest
import logging
from tema.product import Tea, Coffee

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        
        self.nb_producers = 0
        
        self.nb_consumers = 0
        
        self.producers = {}
        
        self.consumers = {}
        
        self.producer_lock = Lock()
        
        self.consumer_lock = Lock()

        
        logging.basicConfig(filename="marketplace.log", filemode='w',
                            level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            datefmt='%d/%m/%Y %H:%M:%S')

    def register_producer(self):
        
        logging.info("producer registered with id %s", self.nb_producers)
        
        self.producers[self.nb_producers] = []
        self.nb_producers += 1

        
        return self.nb_producers - 1

    def publish(self, producer_id, product):
        
        logging.info("producer %s published product %s", producer_id, product)
        
        if len(self.producers[producer_id]) == self.queue_size_per_producer:
            logging.info("publish returned False")
            return False

        
        with self.producer_lock:
            self.producers[producer_id].append(product)
        logging.info("publish returned True")
        return True

    def new_cart(self):
        
        logging.info("cart registered with id %s", self.nb_consumers)
        
        self.consumers[self.nb_consumers] = []
        self.nb_consumers += 1

        
        return self.nb_consumers - 1

    def add_to_cart(self, cart_id, product):
        
        logging.info("cart %s added to cart %s", cart_id, product)
        
        for producer_id in range(self.nb_producers):
            
            for prd in self.producers[producer_id]:
                
                if prd == product:
                    with self.consumer_lock:
                        
                        self.producers[producer_id].remove(product)
                        self.consumers[cart_id].append([product, producer_id])
                    logging.info("add_to_cart returned True")
                    return True
        logging.info("add_to_cart returned False")
        return False


    def remove_from_cart(self, cart_id, product):
        
        logging.info("cart %s removed from cart %s", cart_id, product)
        
        for [prd, producer_id] in self.consumers[cart_id]:
            
            if prd == product:
                with self.consumer_lock:
                    
                    self.consumers[cart_id].remove([product, producer_id])
                    self.producers[producer_id].append(product)
                break

    def place_order(self, cart_id):
        
        
        products = [product for [product, _] in self.consumers[cart_id]]
        logging.info("cart %s placed order: %s", cart_id, products)

        return products


class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        
        self.marketplace = Marketplace(3)
        self.producers = []
        self.consumers = []
        self.products = []

        
        self.products.append(Tea(name='Linden', price=9, type='Herbal'))
        self.products.append(Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM'))

        
        for _ in range(10):
            self.producers.append(self.marketplace.register_producer())

        
        for _ in range(5):
            self.consumers.append(self.marketplace.new_cart())

    def test_register_producer(self):
        
        
        self.assertEqual(self.marketplace.register_producer(), 10)

    def test_publish(self):
        
        
        for _ in range(3):


            self.assertEqual(self.marketplace.publish(0, self.products[0]), True)
        self.assertEqual(self.marketplace.publish(0, self.products[0]), False)

    def test_new_cart(self):
        
        
        self.assertEqual(self.marketplace.new_cart(), 5)

    def test_add_to_cart1(self):
        
        
        for i in range(2):
            self.marketplace.publish(0, self.products[i])

        self.assertEqual(self.marketplace.add_to_cart(0, self.products[0]), True)


        self.assertEqual(self.marketplace.add_to_cart(0, self.products[1]), True)
        
        self.assertEqual(self.marketplace.add_to_cart(0, self.products[0]), False)

    def test_add_to_cart2(self):
        
        
        for i in range(2):
            for j in range(2):
                self.marketplace.publish(i, self.products[j])

        


        self.assertEqual(self.marketplace.add_to_cart(0, self.products[0]), True)
        self.assertEqual(self.marketplace.add_to_cart(0, self.products[1]), True)
        self.assertEqual(self.marketplace.add_to_cart(0, self.products[0]), True)

    def test_add_to_cart3(self):
        
        
        for i in range(2):
            self.marketplace.publish(0, self.products[i])

        self.assertEqual(self.marketplace.add_to_cart(0, self.products[0]), True)
        self.assertEqual(self.marketplace.add_to_cart(1, self.products[1]), True)
        
        self.assertEqual(self.marketplace.add_to_cart(1, self.products[0]), False)

    def test_remove_from_cart1(self):
        
        
        for i in range(2):
            self.marketplace.publish(0, self.products[i])



        self.marketplace.add_to_cart(0, self.products[0])
        self.marketplace.remove_from_cart(0, self.products[0])
        
        self.assertEqual(self.marketplace.add_to_cart(0, self.products[0]), True)

    def test_remove_from_cart2(self):
        
        
        for i in range(2):
            self.marketplace.publish(0, self.products[i])
            self.marketplace.add_to_cart(0, self.products[i])
            self.marketplace.remove_from_cart(0, self.products[i])
            
            self.assertEqual(self.marketplace.add_to_cart(0, self.products[i]), True)

    def test_place_order(self):
        
        for i in range(2):
            self.marketplace.publish(0, self.products[i])
            self.marketplace.add_to_cart(0, self.products[i])

        
        self.assertEqual(self.marketplace.place_order(0), self.products)


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
        
        while True:
            for product in self.products:
                
                [product_id, quantity, wait_time] = product
                time.sleep(wait_time)

                
                while quantity > 0:
                    
                    while not self.marketplace.publish(self.producer_id, product_id):
                        time.sleep(self.republish_wait_time)
                    quantity -= 1


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
