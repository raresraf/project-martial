"""
This module implements a multi-threaded producer-consumer simulation of a marketplace.

It defines the following main classes:
- Marketplace: A thread-safe marketplace where producers can publish products and
  consumers can purchase them. It manages inventory, producers, and consumer carts.
- Producer: A thread that generates and publishes products to the marketplace.
- Consumer: A thread that adds products to a cart and places orders from the marketplace.

The simulation uses threading primitives like Locks to ensure data consistency
in a concurrent environment. It also includes a suite of unit tests to verify the
marketplace's functionality.
"""
from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    Represents a consumer thread that simulates purchasing products from the marketplace.

    This thread processes a list of shopping carts, where each cart contains a series
    of 'add' or 'remove' operations. It interacts with the `Marketplace` to perform
    these actions.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of cart operations to be performed.
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (float): The time in seconds to wait before retrying to
                                     add a product if it is currently unavailable.
            **kwargs: Keyword arguments for the `Thread` constructor, including the
                      consumer's 'name'.
        """

        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

        self.name = self.kwargs['name']

    def run(self):
        """
        The main execution logic for the consumer thread.

        Iterates through the assigned carts, creating a new cart in the marketplace
        for each one. It then processes all 'add' and 'remove' operations for that
        cart. For 'add' operations, it will busy-wait and retry if the product is
        not immediately available. Finally, it places an order for the contents of
        the cart.
        """
        for cart in self.carts:
            new_cart_id = self.marketplace.new_cart()
            self.marketplace.add_empty_cart(self.name, new_cart_id)

            for operation in cart:
                if operation['type'] == 'add':
                    for _ in range(operation['quantity']):
                        is_available = self.marketplace 
                            .add_to_cart(new_cart_id, operation['product'])

                        while is_available is False:
                            sleep(self.retry_wait_time)
                            is_available = self.marketplace 
                                .add_to_cart(new_cart_id, operation['product'])
                elif operation['type'] == 'remove':
                    for _ in range(operation['quantity']):
                        self.marketplace.remove_from_cart(new_cart_id, operation['product'])

            _ = self.marketplace.place_order(new_cart_id)


from threading import Lock
import unittest
import time
import logging
from logging.handlers import RotatingFileHandler
from tema.product import Coffee, Tea

class Marketplace:
    """
    Manages the interactions between producers and consumers in a thread-safe manner.

    This class handles the inventory of products, registration of producers, and the
    entire lifecycle of a consumer's shopping cart, from creation to placing an order.
    It uses locks to protect shared data structures like producer and cart dictionaries
    from concurrent access issues.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products that a single
                                           producer can list in the marketplace at any
                                           given time.
        """
        self.queue_size_per_producer = queue_size_per_producer

        self.producer_ids = 0
        self.cart_ids = 0
        self.producer_ids_lock = Lock()
        self.cart_ids_lock = Lock()
        self.cart_lock = Lock()
        self.carts_dict = {}
        self.producers_dict = {}
        self.prod_id = 0

        logging.basicConfig(
            handlers=[RotatingFileHandler('marketplace.log', maxBytes=200000, backupCount=10)],
                        level=logging.INFO)

    @staticmethod
    def gmt_time():
        """
        A static utility method to get the current time formatted as a GMT string.

        Returns:
            str: The current time in HH:MM:SS format (GMT).
        """
        gmt = time.gmtime(time.time())
        gmt_time = str(gmt.tm_hour) + ':' + str(gmt.tm_min) + ':' + str(gmt.tm_sec)
        return gmt_time

    def register_producer(self):
        """
        Registers a new producer, assigning them a unique ID and an inventory slot.

        Returns:
            int: The unique ID for the newly registered producer.
        """

        logging.info('GMT %s --> New producer register', self.gmt_time())

        with self.producer_ids_lock:
            self.producer_ids += 1

        
        self.producers_dict[self.producer_ids] = {'products' : [], 'products_count' : 0}

        logging.info('GMT %s --> Producer was registered', self.gmt_time())
        return self.producer_ids

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        The product will not be published if the producer's inventory queue is full.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product: The product to be published.

        Returns:
            bool: True if the product was successfully published, False otherwise.
        """
        logging.info('GMT %s --> Producer %s wants to publish the product: %s',
                     self.gmt_time(), producer_id, product)

        
        if self.producers_dict[producer_id]['products_count'] == self.queue_size_per_producer:
            logging.info('GMT %s --> Product couldn't be published. 
                        The producer's queue is full.', self.gmt_time())
            return False

        
        self.producers_dict[producer_id]['products_count'] += 1
        self.producers_dict[producer_id]['products'] 
            .append({'available': True, 'product': product})

        logging.info('GMT %s --> Product was published successfully', self.gmt_time())
        return True

    def new_cart(self):
        """
        Creates a new, unique shopping cart ID for a consumer.

        Returns:
            int: The unique ID for the new cart.
        """
        logging.info('GMT %s --> New cart to register.', self.gmt_time())

        with self.cart_ids_lock:
            self.cart_ids += 1

        logging.info('GMT %s --> The cart was registered successfully.', self.gmt_time())
        return self.cart_ids

    def add_empty_cart(self, consumer_name, new_cart_id):
        """
        Associates a new, empty cart with a consumer's name.

        Args:
            consumer_name (str): The name of the consumer.
            new_cart_id (int): The ID of the new cart to associate.
        """
        logging.info('GMT %s --> Consumer %s wants to receive an empty cart with id %s.',
                     self.gmt_time(), consumer_name, new_cart_id)

        self.carts_dict[new_cart_id] = {'consumer_name' : consumer_name, 'products' : []}

        logging.info('GMT %s --> Cart was assigned successfully.', self.gmt_time())

    def add_to_cart(self, cart_id, product):
        """
        Adds a specified product to a consumer's shopping cart.

        This method searches for an available instance of the product from any
        producer and, if found, marks it as unavailable and adds it to the cart.
        The operation is protected by a lock to ensure atomicity.

        Args:
            cart_id (int): The ID of the target shopping cart.
            product: The product to add.

        Returns:
            bool: True if the product was successfully added, False if no available
                  instance of the product was found.
        """

        logging.info('GMT %s --> Someone wants to add %s in the cart with id %s.',
                     self.gmt_time(), product, cart_id)

        with self.cart_lock:
            found_product = 0

            for producer_id, producer_info in self.producers_dict.items():
                if found_product == 1:
                    break

                
                products_list = producer_info['products']
                for product_info in products_list:
                    if product_info['product'] == product and product_info['available'] is True:
                        product_info['available'] = False
                        self.prod_id = producer_id
                        self.carts_dict[cart_id]['products'] 
                            .append({'producer_id': self.prod_id, 'product':product})
                        found_product = 1
                        break

        if found_product == 1:
            logging.info('GMT %s --> The product was successfully added to cart.',
                         self.gmt_time())
            return True

        logging.info('GMT %s --> The product couldn' be added to cart.', self.gmt_time())
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a specified product from a consumer's shopping cart.

        This method finds the product in the consumer's cart, removes it, and then
        marks the corresponding product in the producer's inventory as available again.
        The operation is protected by a lock.

        Args:
            cart_id (int): The ID of the target shopping cart.
            product: The product to remove.
        """

        logging.info('GMT %s --> Someone wants to remove %s from the cart with id %s.',
                     self.gmt_time(), product, cart_id)

        with self.cart_lock:
            found_producer_id = 0
            consumer_products_list = self.carts_dict[cart_id]['products']

            
            for product_info in consumer_products_list:
                if product_info['product'] == product:
                    consumer_products_list.remove(product_info)
                    found_producer_id = product_info['producer_id']
                    break

            
            producer_products_list = self.producers_dict[found_producer_id]['products']
            for product_info in producer_products_list:
                if product_info['product'] == product and product_info['available'] is False:
                    product_info['available'] = True
                    break



            logging.info('GMT %s --> The product was successfully removed from the cart.',
                         self.gmt_time())

    def place_order(self, cart_id):
        """
        Finalizes the purchase of all items in a shopping cart.

        For each item in the cart, this method removes it permanently from the
        corresponding producer's inventory. It also logs the purchased items.
        The operation is protected by a lock.

        Args:
            cart_id (int): The ID of the cart for which to place the order.

        Returns:
            list: A list of the products that were in the cart.
        """

        logging.info('GMT %s --> Someone wants to place an order for cart with id %s.',
                     self.gmt_time(), cart_id)
        with self.cart_lock:
            
            for cart_product_info in self.carts_dict[cart_id]['products']:
                producer_id = cart_product_info['producer_id']
                producer_list = self.producers_dict[producer_id]['products']

                for producer_product_info in producer_list:
                    if producer_product_info['product'] == cart_product_info['product'] 
                    and producer_product_info['available'] is False:
                        producer_list.remove(producer_product_info)
                        self.producers_dict[producer_id]['products_count'] -= 1
                        break

            for product in self.carts_dict[cart_id]['products']:
                print(f"{self.carts_dict[cart_id]['consumer_name']} bought {product['product']}")

            logging.info('GMT %s --> The order was successfully placed.', self.gmt_time())
            return self.carts_dict[cart_id]['products']

class MarketplaceTestCase(unittest.TestCase):
    """
    A suite of unit tests for the Marketplace class to ensure its core
    functionality operates as expected under various conditions.
    """
    product1 = Tea(name='Linden', price=9, type='Herbal')
    product2 = Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM')
    product1_info = (product1, 2, 0.18)
    product2_info = (product2, 1, 0.23)
    republish_wait_time = 0.15
    kwargs = {'name': 'prod1', 'daemon': True}

    def setUp(self):
        
        self.marketplace = Marketplace(queue_size_per_producer=3)

    def test_register_producer(self):
        
        id_producer = self.marketplace.register_producer()
        self.assertEqual(id_producer, 1)
        self.assertEqual(len(self.marketplace.producers_dict), 1)

        id_producer = self.marketplace.register_producer()
        self.assertEqual(id_producer, 2)
        self.assertEqual(len(self.marketplace.producers_dict), 2)

        id_producer = self.marketplace.register_producer()
        self.assertEqual(id_producer, 3)
        self.assertEqual(len(self.marketplace.producers_dict), 3)

    def test_publish(self):
        
        
        _ = self.marketplace.register_producer()
        _ = self.marketplace.register_producer()
        _ = self.marketplace.register_producer()

        
        mock_mp = Marketplace(queue_size_per_producer=3)
        mock_mp.producers_dict = 
            {1: {'products': [], 'products_count': 0}, 
            2: {'products': [], 'products_count': 0}, 
            3: {'products': [], 'products_count': 0}}

        
        mock_mp.producers_dict[1]['products'].append({'available': True, 'product':self.product1})
        mock_mp.producers_dict[1]['products_count'] = 1
        published = self.marketplace.publish(1, self.product1)
        self.assertTrue(published)
        self.assertEqual(self.marketplace.producers_dict, mock_mp.producers_dict)

        
        mock_mp.producers_dict[1]['products'].append({'available': True, 'product':self.product1})
        mock_mp.producers_dict[1]['products_count'] = 2
        published = self.marketplace.publish(1, self.product1)
        self.assertTrue(published)
        self.assertEqual(self.marketplace.producers_dict, mock_mp.producers_dict)

        
        mock_mp.producers_dict[1]['products'].append({'available': True, 'product':self.product2})
        mock_mp.producers_dict[1]['products_count'] = 3
        published = self.marketplace.publish(1, self.product2)
        self.assertTrue(published)
        self.assertEqual(self.marketplace.producers_dict, mock_mp.producers_dict)

        
        published = self.marketplace.publish(1, self.product2)
        self.assertFalse(published)
        self.assertEqual(self.marketplace.producers_dict, mock_mp.producers_dict)

        
        mock_mp.producers_dict[3]['products'].append({'available': True, 'product':self.product2})
        mock_mp.producers_dict[3]['products_count'] = 1
        published = self.marketplace.publish(3, self.product2)
        self.assertTrue(published)
        self.assertEqual(self.marketplace.producers_dict, mock_mp.producers_dict)

    def test_new_cart(self):
        
        id_cart = self.marketplace.new_cart()
        self.assertEqual(id_cart, 1)

        id_cart = self.marketplace.new_cart()
        self.assertEqual(id_cart, 2)

        id_cart = self.marketplace.new_cart()
        self.assertEqual(id_cart, 3)

    def test_add_empty_cart(self):
        
        
        mock_mp = Marketplace(queue_size_per_producer=3)
        mock_mp.carts_dict = {}

        
        mock_mp.carts_dict[1] = {'consumer_name':'cons1', 'products':[]}
        self.marketplace.add_empty_cart('cons1', 1)
        self.assertEqual(self.marketplace.carts_dict, mock_mp.carts_dict)

         
        mock_mp.carts_dict[2] = {'consumer_name':'cons1', 'products':[]}
        self.marketplace.add_empty_cart('cons1', 2)
        self.assertEqual(self.marketplace.carts_dict, mock_mp.carts_dict)

         
        mock_mp.carts_dict[3] = {'consumer_name':'cons2', 'products':[]}
        self.marketplace.add_empty_cart('cons2', 3)
        self.assertEqual(self.marketplace.carts_dict, mock_mp.carts_dict)

    def test_add_to_cart(self):
        
        
        mock_mp = Marketplace(queue_size_per_producer=3)
        mock_producers_dict = 
            {1: {'products': [{'available': True, 'product':self.product1}, 
                            {'available': True, 'product':self.product2}], 
                'products_count': 2}, 
            2: {'products': [{'available': True, 'product':self.product1}], 'products_count': 1}, 
            3: {'products': [{'available': True, 'product':self.product2}], 'products_count': 1}}
        mock_carts_dict = 
            {1: {'consumer_name': 'cons1', 'products': []}, 
            2: {'consumer_name': 'cons1', 'products': []}, 
            3: {'consumer_name': 'cons2', 'products': []}}
        mock_mp.producers_dict = mock_producers_dict
        mock_mp.carts_dict = mock_carts_dict
        self.marketplace.producers_dict = mock_producers_dict
        self.marketplace.carts_dict = mock_carts_dict

        
        mock_mp.producers_dict[1]['products'] =  
            [{'available': False, 'product':self.product1}, 
            {'available': True, 'product':self.product2}]
        mock_mp.carts_dict[1]['products'].append(self.product1)
        added_to_cart = self.marketplace.add_to_cart(1, self.product1)
        self.assertTrue(added_to_cart)
        self.assertEqual(self.marketplace.producers_dict, mock_mp.producers_dict)
        self.assertEqual(self.marketplace.carts_dict, mock_mp.carts_dict)

        
        mock_mp.producers_dict[1]['products'] = 
            [{'available': False, 'product':self.product1}, 
            {'available': False, 'product':self.product2}]
        mock_mp.carts_dict[2]['products'].append(self.product2)
        added_to_cart = self.marketplace.add_to_cart(2, self.product2)
        self.assertTrue(added_to_cart)
        self.assertEqual(self.marketplace.producers_dict, mock_mp.producers_dict)
        self.assertEqual(self.marketplace.carts_dict, mock_mp.carts_dict)

        
        mock_mp.producers_dict[2]['products'] = [{'available': True, 'product':self.product1}]
        mock_mp.carts_dict[2]['products'].append(self.product1)
        added_to_cart = self.marketplace.add_to_cart(2, self.product1)
        self.assertTrue(added_to_cart)
        self.assertEqual(self.marketplace.producers_dict, mock_mp.producers_dict)
        self.assertEqual(self.marketplace.carts_dict, mock_mp.carts_dict)

        
        added_to_cart = self.marketplace.add_to_cart(3, self.product1)
        self.assertFalse(added_to_cart)

    def test_place_order(self):
        
        mock_mp = Marketplace(queue_size_per_producer=3)
        mock_producers_dict = 
            {1: {'products': [{'available': False, 'product':self.product1}, 
                            {'available': False, 'product':self.product2}], 
                'products_count': 2}}
        mock_carts_dict = 
            {1: {'consumer_name': 'cons1', 
                'products': [{'producer_id' : 1, 'product' : self.product1}, 
                            {'producer_id' : 1, 'product' : self.product2}]}}
        mock_mp.producers_dict = mock_producers_dict
        mock_mp.carts_dict = mock_carts_dict
        self.marketplace.producers_dict = mock_producers_dict
        self.marketplace.carts_dict = mock_carts_dict


        order = self.marketplace.place_order(1)
        order_result = []
        for prod in order:
            order_result.append(prod['product'])
        expected_order = [self.product1, self.product2]
        self.assertEqual(order_result, expected_order)


from threading import Thread
from time import sleep

class Producer(Thread):
    """
    Represents a producer thread that continuously creates and publishes products.

    This thread registers itself with the marketplace and then enters an infinite loop
    to produce a predefined list of products. It respects the production time for
    each item and will wait and retry if the marketplace's queue is full.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of tuples, where each tuple contains the product,
                             quantity to produce, and production time.
            marketplace (Marketplace): The marketplace instance to interact with.
            republish_wait_time (float): The time in seconds to wait before retrying
                                         to publish a product if the marketplace is full.
            **kwargs: Keyword arguments for the `Thread` constructor.
        """

        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

        self.name = self.kwargs['name']
        self.id_producer = self.marketplace.register_producer()

    def run(self):
        """
        The main execution logic for the producer thread.

        Continuously loops through its list of products, producing and publishing
        each one according to its specified quantity and production time. If
        publishing fails (e.g., because the queue is full), it waits before retrying.
        """
        while True:
            for (product_info, quantity, production_time) in self.products:
                for _ in range(quantity):
                    while True:
                        can_publish = self.marketplace.publish(self.id_producer, product_info)
                        if can_publish:
                            sleep(production_time)
                            break

                        sleep(self.republish_wait_time)
