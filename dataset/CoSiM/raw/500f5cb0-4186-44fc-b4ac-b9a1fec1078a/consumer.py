

import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)



        self.retry_wait_time = retry_wait_time
        self.carts = carts
        self.marketplace = marketplace


    def add(self, cart_id, product_id, qty):
        

        
        count = 0
        while count < qty:
            
            must_wait = not (self.marketplace.add_to_cart(cart_id, product_id))
            if must_wait:
                
                time.sleep(self.retry_wait_time)
            else:
                
                count += 1

    def rm(self, cart_id, product_id, qty):
        

        
        count = 0
        while count < qty:
            
            self.marketplace.remove_from_cart(cart_id, product_id)
            count += 1

    def run(self):
        
        cart_id = self.marketplace.new_cart()

        for cart in self.carts:
            for product in cart:
                


                product_id = product.get("product")
                qty = product.get("quantity")
                op = product.get("type")

                
                if op == "add":
                    self.add(cart_id, product_id, qty)
                elif op == "remove":
                    self.rm(cart_id, product_id, qty)

        
        order = self.marketplace.place_order(cart_id)
        for product in order:
            
            print(self.name, "bought", product)


from threading import Lock
import unittest


class TestMarketplace(unittest.TestCase):
    def setUp(self):
        
        self.marketplace = Marketplace(10)

        
        self.producer_id = self.marketplace.register_producer()

        
        self.product = {"product_type": "Coffee"}
        self.product["name"] = "Indonezia"
        self.product["acidity"] = 5.05
        self.product["roast_level"] = "MEDIUM"
        self.product["price"] = 1

        
        self.cart_id = self.marketplace.new_cart()

    def tearDown(self):
        self.product = None
        self.producer_id = -1
        self.cart_id = -1
        self.marketplace.producers = {}
        self.marketplace.carts = {}
        self.marketplace = None

    def test_register_producer(self):
        
        self.assertGreater(self.producer_id, 0)
        
        self.assertListEqual(self.marketplace.producers.get(self.producer_id), [])

    def test_publish(self):
        
        ret = self.marketplace.publish(self.producer_id, self.product)

        
        self.assertTrue(ret)

    def test_new_cart(self):
        
        self.assertGreater(self.cart_id, 0)
        
        self.assertListEqual(self.marketplace.carts.get(self.cart_id), [])

    def test_add_to_cart(self):
        
        ret = self.marketplace.publish(self.producer_id, self.product)
        
        self.assertTrue(ret)
        
        ret = self.marketplace.add_to_cart(self.cart_id, self.product)
        
        self.assertTrue(ret)

    def test_remove_from_cart(self):
        
        ret = self.marketplace.publish(self.producer_id, self.product)
        
        self.assertTrue(ret)
        
        ret = self.marketplace.add_to_cart(self.cart_id, self.product)
        
        self.assertTrue(ret)
        
        self.marketplace.remove_from_cart(self.cart_id, self.product)
        
        self.assertListEqual(self.marketplace.carts.get(self.cart_id), [])

    def test_place_order(self):
        order = self.marketplace.place_order(self.cart_id)
        
        self.assertIsNotNone(order)
        
        self.assertListEqual(order, self.marketplace.carts.get(self.cart_id))


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        

        self.queue_size_per_producer = queue_size_per_producer

        
        self.producer_count = 0

        
        self.producers = {}

        
        self.publishing_lock = Lock()

        
        self.consumer_count = 0

        
        self.carts = {}

        
        self.cart_lock = Lock()


    def register_producer(self):
        

        
        self.producer_count += 1
        self.producers[self.producer_count] = []

        return self.producer_count

    def publish(self, producer_id, product):
        

        successful_publish = False

        
        self.publishing_lock.acquire()

        
        products = self.producers.get(producer_id)
        queue_size = len(products)

        
        if queue_size < self.queue_size_per_producer:
            
            successful_publish = True
            self.producers.get(producer_id).append(product)

        
        self.publishing_lock.release()

        return successful_publish

    def new_cart(self):
        

        
        self.consumer_count += 1
        self.carts[self.consumer_count] = []

        return self.consumer_count

    def add_to_cart(self, cart_id, product):
        

        
        product_owner = None

        
        self.cart_lock.acquire()

        


        for curr_producer in list(self.producers.keys()):
            for curr_product in self.producers.get(curr_producer):
                if product == curr_product:
                    
                    product_owner = curr_producer
                    break

        if product_owner is not None:
            


            self.producers.get(product_owner).remove(product)
            
            self.carts.get(cart_id).append([product, product_owner])

        
        self.cart_lock.release()

        
        ret = product_owner is not None
        return ret

    def remove_from_cart(self, cart_id, product):
        

        
        product_owner = None

        
        cart_products = self.carts.get(cart_id)

        
        self.cart_lock.acquire()

        
        for curr_product in cart_products:
            if product == curr_product[0]:
                
                product_owner = curr_product[1]
                break

        if product_owner is not None:
            
            self.carts[cart_id].remove([product, product_owner])
            self.producers.get(product_owner).append(product)

        
        self.cart_lock.release()

    def place_order(self, cart_id):
        

        
        order = []

        
        cart_products = self.carts.get(cart_id)
        for product in cart_products:
            
            order.append(product[0])

        return order

import time
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)

        self.republish_wait_time = republish_wait_time
        self.products = products
        self.marketplace = marketplace

    def run(self):
        while True:
            
            producer_id = self.marketplace.register_producer()

            
            for product in self.products:
                
                count = 0

                
                product_id = product[0]
                qty = product[1]
                waiting_time = product[2]

                
                while count < qty:
                    
                    must_wait = not (self.marketplace.publish(producer_id, product_id))

                    if must_wait:
                        
                        time.sleep(self.republish_wait_time)
                    else:
                        
                        count += 1
                        time.sleep(waiting_time)
