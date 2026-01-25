


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def get_name(self):
        return self.name

    def run(self):
        for cart in self.carts:


            cart_id = self.marketplace.new_cart()
            for operation in cart:
                quantity = 0
                while quantity < operation['quantity']:
                    if operation['type'] == 'add':
                        result = self.marketplace.add_to_cart(cart_id, operation['product'])
                    if operation['type'] == 'remove':
                        result = self.marketplace.remove_from_cart(cart_id, operation['product'])

                    if result is None or result is True:
                        quantity += 1
                    else:
                        time.sleep(self.retry_wait_time)

            self.marketplace.place_order(cart_id)


from threading import currentThread, Lock
import unittest

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        


        self.queue_size_per_producer = queue_size_per_producer
        
        self.num_producers = 0
        self.num_carts = 0
        self.num_items_producer = []

        self.producers = {}
        self.carts = {}

        self.lock = Lock()

    def register_producer(self):
        

        with self.lock:
            producer_id = self.num_producers
            self.num_producers += 1


            self.num_items_producer.insert(producer_id, 0)

        return producer_id

    def publish(self, producer_id, product):
        

        casted_producer_int = int(producer_id)

        
        if self.num_items_producer[casted_producer_int] >= self.queue_size_per_producer:
            return False

        self.num_items_producer[casted_producer_int] += 1
        self.producers[product] = casted_producer_int

        return True

    def new_cart(self):
        
        with self.lock:
            self.num_carts = self.num_carts + 1
            cart_id = self.num_carts

        self.carts[cart_id] = []
        return cart_id

    def add_to_cart(self, cart_id, product):
        

        with self.lock:
            
            if self.producers.get(product) is None:
                return False

            self.num_items_producer[self.producers[product]] -= 1
            producers_id = self.producers.pop(product)

        self.carts[cart_id].append(product)
        self.carts[cart_id].append(producers_id)

        return True

    def remove_from_cart(self, cart_id, product):
        

        if product in self.carts[cart_id]:
            index = self.carts[cart_id].index(product)
            self.carts[cart_id].remove(product)
            producers_id = self.carts[cart_id].pop(index)
            self.producers[product] = producers_id

            with self.lock:
                self.num_items_producer[int(producers_id)] += 1


    def place_order(self, cart_id):
        

        product_list = self.carts.pop(cart_id)

        for i in range(0, len(product_list), 2):
            with self.lock:
                print(currentThread().get_name() + " bought " + str(product_list[i]))
                self.num_items_producer[product_list[i + 1]] -= 1

        return product_list

class TestMarketplace(unittest.TestCase):
    product = "Tea(name='Linden', price=9, type='Herbal')"
    product2 = "Tea(name='Linden', price=10, type='Herbal')"
    def setUp(self):
        self.marketplace = Marketplace(13)

    def test_publish_limit_fail(self):
        producer_id = self.marketplace.queue_size_per_producer - 1
        self.marketplace.num_items_producer = [0] * self.marketplace.queue_size_per_producer
        self.marketplace.num_items_producer[producer_id] = self.marketplace.queue_size_per_producer

        self.assertFalse(self.marketplace.publish(str(producer_id), self.product),
            "Queue size per producer limit!")

    def test_publish_limit_success(self):
        producer_id = self.marketplace.queue_size_per_producer - 1
        self.marketplace.num_items_producer = [0] * self.marketplace.queue_size_per_producer
        self.marketplace.num_items_producer[producer_id] \
            = self.marketplace.queue_size_per_producer - 1

        self.assertTrue(self.marketplace.publish(str(producer_id), self.product), "Cannot publish!")

    def test_add_to_cart_fail(self):
        cart_id = self.marketplace.new_cart()

        self.assertFalse(self.marketplace.add_to_cart(cart_id, self.product),
            "Product shouldn't be found!")

    def test_add_to_cart_success(self):
        product = "Tea(name='Linden', price=9, type='Herbal')"
        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()
        self.marketplace.publish(str(producer_id), product)

        self.assertTrue(self.marketplace.add_to_cart(cart_id, product),
            "Product should be found!")

    def test_new_cart(self):
        cart_id = self.marketplace.new_cart()
        self.assertEqual(cart_id, self.marketplace.num_carts, "Number of carts not increased!")
        self.assertIsInstance(self.marketplace.carts[cart_id], type([]), "List not initialized!")

    def test_remove_from_cart(self):
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(str(producer_id), self.product)
        self.marketplace.publish(str(producer_id), self.product2)
        self.marketplace.publish(str(producer_id), self.product)

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, self.product)
        self.marketplace.add_to_cart(cart_id, self.product)
        self.marketplace.add_to_cart(cart_id, self.product2)

        self.marketplace.remove_from_cart(cart_id, self.product2)

        self.assertNotIn(self.product2, self.marketplace.carts[cart_id], "Product not removed!")>>>> file: producer.py


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = 0



    def run(self):
        self.producer_id = self.marketplace.register_producer()

        while True:
            for product in self.products:
                produced = 0
                while produced < product[1]:
                    result = self.marketplace.publish(str(self.producer_id), product[0])

                    if result:
                        time.sleep(product[2])
                        produced += 1
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
