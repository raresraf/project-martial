


from threading import Thread
from time import sleep

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for oper in cart:
                type_of_operation = oper["type"]
                prod = oper["product"]
                quantity = oper["quantity"]
                if type_of_operation == "add":
                    self.add_cart(cart_id, prod, quantity)
                elif type_of_operation == "remove":
                    self.remove_cart(cart_id, prod, quantity)
            p_purchased = self.marketplace.place_order(cart_id)
            for prod in p_purchased:
                print(f"{self.getName()} bought {prod}")

    def add_cart(self, cart_id, product_id, quantity):
        
        for _ in range(quantity):
            while True:
                added = self.marketplace.add_to_cart(cart_id, product_id)
                if added:
                    break
                sleep(self.retry_wait_time)

    def remove_cart(self, cart_id, product_id, quantity):
        
        for _ in range(quantity):
            while True:
                removed = self.marketplace.remove_from_cart(cart_id, product_id)
                if removed:
                    break
                sleep(self.retry_wait_time)


from threading import Lock
import unittest
import sys
sys.path.insert(1, './tema')
import product as produs

class Marketplace:
    
    def __init__(self, queue_size_per_producer):


        

        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0
        self.cart_id = 0
        self.queues = []
        self.carts = []
        self.mutex = Lock()
        self.products_dict = {}

    def register_producer(self):
        

        self.mutex.acquire()
        producer_id = self.producer_id
        self.producer_id += 1
        self.queues.append([])
        self.mutex.release()
        return str(producer_id)

    def publish(self, producer_id, product):
        

        index_prod = int(producer_id)
        if len(self.queues[index_prod]) == self.queue_size_per_producer:
            return False
        self.queues[index_prod].append(product)
        self.products_dict[product] = index_prod
        return True

    def new_cart(self):
        

        self.mutex.acquire()
        cart_id = self.cart_id
        self.cart_id += 1
        self.mutex.release()
        self.carts.append([])
        return cart_id

    def add_to_cart(self, cart_id, product):
        

        prod_in_queue = False
        for queue in self.queues:
            if product in queue:
                prod_in_queue = True
                queue.remove(product)
                break
        if not prod_in_queue:
            return False
        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        

        if product not in self.carts[cart_id]:
            return False
        index_producer = self.products_dict[product]
        if len(self.queues[index_producer]) == self.queue_size_per_producer:
            return False


        self.carts[cart_id].remove(product)
        self.queues[index_producer].append(product)
        return True

    def place_order(self, cart_id):
        

        cart_product_list = self.carts[cart_id]
        self.carts[cart_id] = []
        return cart_product_list

class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        
        self.marketplace = Marketplace(4)

    def test_register_producer(self):
        
        self.assertEqual(self.marketplace.register_producer(), str(0))
        self.assertNotEqual(self.marketplace.register_producer(), str(3))
        self.assertEqual(self.marketplace.register_producer(), str(2))
        self.assertNotEqual(self.marketplace.register_producer(), str(0))
        self.assertNotEqual(self.marketplace.register_producer(), str(3))
        self.assertNotEqual(self.marketplace.register_producer(), str(2))

    def test_publish(self):
        
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.assertTrue(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))
        self.assertFalse(self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(1), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(1), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(1), produs.Tea("Linden", 9, "Herbal")))
        self.assertTrue(self.marketplace.publish(str(1), produs.Tea("Linden", 9, "Herbal")))
        self.assertFalse(self.marketplace.publish(str(1), produs.Tea("Linden", 9, "Herbal")))
        self.assertFalse(self.marketplace.publish(str(1), produs.Tea("Linden", 9, "Herbal")))

    def test_new_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertNotEqual(self.marketplace.new_cart(), 3)
        self.assertEqual(self.marketplace.new_cart(), 2)
        self.assertNotEqual(self.marketplace.new_cart(), 0)
        self.assertNotEqual(self.marketplace.new_cart(), 3)
        self.assertNotEqual(self.marketplace.new_cart(), 2)

    def test_add_to_cart(self):
        
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal"))
        self.assertTrue(self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal")))
        self.assertFalse(self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal")))

    def test_remove_from_cart(self):
        
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal"))
        self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal"))
        self.assertTrue(self.marketplace.remove_from_cart(0, produs.Tea("Linden", 9, "Herbal")))
        self.assertFalse(self.marketplace.remove_from_cart(0, produs.Tea("Linden", 9, "Herbal")))

    def test_place_order(self):
        
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal"))
        self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal"))
        self.assertEqual([produs.Tea("Linden", 9, "Herbal")], self.marketplace.place_order(0))


from threading import Thread
from time import sleep

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
                quantity = product[1]
                for _ in range(0, quantity):
                    self.publish_product(product[0], product[2])

    def publish_product(self, product, production_time):
        
        while True:
            published = self.marketplace.publish(self.producer_id, product)
            if published:
                sleep(production_time)
                break
            sleep(self.republish_wait_time)
