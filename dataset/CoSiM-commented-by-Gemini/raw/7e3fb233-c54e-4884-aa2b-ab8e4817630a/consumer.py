"""
This module simulates a multi-threaded producer-consumer marketplace.

It contains the core components for the simulation:
- Marketplace: A thread-safe central hub where products are published and purchased.
- Producer: A thread that publishes products to the marketplace.
- Consumer: A thread that adds products to a cart and places orders.
- TestMarketplace: A suite of unit tests to validate the marketplace functionality.
"""
from time import sleep
from threading import Thread


class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace.

    Each consumer is initialized with a predefined set of shopping actions
    (adding and removing items from a cart) and executes them sequentially.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        Args:
            carts (list): A list of shopping carts, where each cart is a list
                          of add/remove operations.
            marketplace (Marketplace): The central marketplace object.
            retry_wait_time (float): Time in seconds to wait before retrying to
                                     add a product if it's not available.
            **kwargs: Keyword arguments for the parent Thread class.
        """
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        """
        The main execution method for the consumer thread.

        Iterates through its assigned carts, processes all add/remove commands
        for each cart, and finally places the order.
        """
        i = 0
        
        # Invariant: Loop through all shopping lists assigned to this consumer.
        for i in range(len(self.carts)):
            listAddRem = self.carts[i]
            j = 0
            new_cart = self.marketplace.new_cart()
            
            # Invariant: Process each action (add/remove) in the current shopping list.
            for j in range(len(listAddRem)):
                
                command = listAddRem[j]
                AddRem = command["type"]
                prod = command["product"]
                qty = command["quantity"]
                k = 0
                
                # Pre-condition: The 'qty' must be a positive integer.
                # Invariant: Execute the add or remove operation 'qty' times.
                while k < qty:
                    if AddRem == "add":
                        # Attempt to add the product to the cart.
                        res = self.marketplace.add_to_cart(new_cart, prod)
                        if res:
                            k += 1
                        else:
                            # If adding fails (product unavailable), wait and retry.
                            sleep(self.retry_wait_time)
                    elif AddRem == "remove":
                        # Remove the product from the cart.
                        self.marketplace.remove_from_cart(new_cart, prod)
                        k += 1
            
            # After processing all actions, place the final order.
            self.marketplace.place_order(new_cart)

import sys
import logging
sys.path.append('/tema/product')
import unittest
from threading import Lock, currentThread


class TestLogging:
    """
    A utility class intended to provide a singleton logger.

    Note: This class has implementation issues, such as using 'cls' as an
    instance variable and referencing an undefined 'logger' object. It is
    documented as-is without correction.
    """
    _myLogg = None

    def __init__(cls):
        """Initializes the logger if it hasn't been already."""
        if cls._myLogg is None:
            cls._myLogg = logging.getLogger("logg")
            file = logging.handlers.RotatingFileHandler('marketplace.log',
														mode='a', maxBytes=4096, backupCount=0,
														encoding=None, delay=False, errors=None)

            formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
            file.setFormatter(formatter)
            logger.addHandler(file)

class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""

    @classmethod
    def setUp(self):
        """
        Set up the test environment before each test.

        Initializes a marketplace, registers producers, creates carts, and
        defines sample products for testing.
        """
        self.Marketplace = Marketplace(3)
        self.prod1 = self.Marketplace.register_producer()
        self.prod2 = self.Marketplace.register_producer()
        self.cart1 = self.Marketplace.new_cart()
        self.cart2 = self.Marketplace.new_cart()
        obj = product.Coffee("Indonezia", 5.05, 1, "Medium")
        obj2 = product.Tea("Linden", "Herbal", 9)
        obj3 = product.Coffee("ElDuMari", 6.05, 2, "Large")
        self.produs1 = obj
        self.produs2 = obj2
        self.produs3 = obj3

    def test_register(self):
        """Tests that producer registration returns sequential IDs."""
        self.assertEqual(self.prod1, 0)
        self.assertEqual(self.prod2, 1)

    def test_publish(self):
        """
        Tests the product publishing logic, including capacity limits.
        
        A producer should be able to publish products up to their queue size
        and should fail afterwards.
        """
        i = 0
        for i in range(5):
            if i < 3:
                self.assertTrue(self.Marketplace.publish(
                    self.prod1, self.produs1))
                self.assertTrue(self.Marketplace.publish(
                    self.prod2, self.produs2))
            else:
                self.assertFalse(self.Marketplace.publish(
                    self.prod1, self.produs1))
                self.assertFalse(self.Marketplace.publish(
                    self.prod2, self.produs2))

    def test_new_cart(self):
        """Tests that new cart creation returns sequential IDs."""
        self.assertEqual(self.cart1, 0)
        self.assertEqual(self.cart2, 1)

    def test_add_to_cart(self):
        """
        Tests adding products to a cart.

        Ensures that a product can only be added if it has been published to
        the marketplace first.
        """
        i = 0
        j = 0
        for j in range(3):
            self.Marketplace.publish(self.prod1, self.produs1)
            self.Marketplace.publish(self.prod2, self.produs2)
        for i in range(6):
            if i < 3:
                self.assertTrue(self.Marketplace.add_to_cart(
                    self.cart1, self.produs1))
                self.assertTrue(self.Marketplace.add_to_cart(
                    self.cart2, self.produs2))
            else:
                self.assertFalse(self.Marketplace.add_to_cart(
                    self.cart1, self.produs1))
                self.assertFalse(self.Marketplace.add_to_cart(
                    self.cart2, self.produs2))

    def test_remove_from_cart(self):
        """Tests removing products from a cart."""
        for j in range(3):


            self.Marketplace.publish(self.prod1, self.produs1)
            self.Marketplace.publish(self.prod2, self.produs2)
            self.Marketplace.add_to_cart(self.cart1, self.produs1)



        self.Marketplace.add_to_cart(self.cart1, self.produs2)

        self.Marketplace.remove_from_cart(self.cart1, self.produs1)
        self.assertEqual(self.Marketplace.cartList[self.cart1],
                         [self.produs1, self.produs1, self.produs2])

        self.Marketplace.remove_from_cart(self.cart1, self.produs2)
        self.assertEqual(self.Marketplace.cartList[self.cart1],
                         [self.produs1, self.produs1])

        self.Marketplace.remove_from_cart(self.cart1, self.produs1)
        self.Marketplace.remove_from_cart(self.cart1, self.produs1)
        self.assertEqual(self.Marketplace.cartList[self.cart1], [])

    def test_place_order(self):
        """
        Tests the final 'place order' logic.

        Verifies that the correct list of products is returned after a series
        of add and remove operations.
        """
        self.Marketplace.publish(self.prod1, self.produs1)
        self.Marketplace.publish(self.prod2, self.produs2)
        self.Marketplace.publish(self.prod1, self.produs2)

        self.Marketplace.add_to_cart(self.cart1, self.produs1)
        self.Marketplace.add_to_cart(self.cart1, self.produs2)
        self.Marketplace.add_to_cart(self.cart1, self.produs2)

        self.Marketplace.publish(self.prod1, self.produs3)
        self.Marketplace.add_to_cart(self.cart1, self.produs3)

        self.Marketplace.publish(self.prod1, self.produs3)
        self.Marketplace.add_to_cart(self.cart1, self.produs3)

        self.Marketplace.remove_from_cart(self.cart1, self.produs1)


        self.Marketplace.remove_from_cart(self.cart1, self.produs2)
        self.Marketplace.remove_from_cart(self.cart1, self.produs3)

        self.assertEqual(self.Marketplace.place_order(self.cart1),
                         [self.produs2, self.produs3])


class Marketplace:
    """
    A thread-safe marketplace for producers to publish and consumers to buy products.

    This class acts as the central shared resource, using locks to manage
    concurrent access from multiple producer and consumer threads.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in the marketplace at
                                           any given time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = -1
        self.cart_id = -1
        self.listofproducts = [] 
        self.productsgotid = {} 
        self.producersNrQueueSize = [] 
        self.cartList = {} 
        # Locks to ensure thread-safe operations
        self.lock_operations = Lock()
        self.CartsLocks = Lock()
        self.LockAddRemCart = Lock()
        self.ListSubmit = Lock()

    def register_producer(self):
        """
        Registers a new producer, assigning it a unique ID.

        This operation is thread-safe.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        with self.lock_operations:
            self.id_producer += 1
            
            self.producersNrQueueSize.insert(self.id_producer, self.queue_size_per_producer)
        return self.id_producer

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace from a specific producer.

        This operation is thread-safe. It will fail if the producer has reached
        its publication limit (queue size).

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (any): The product to be published.

        Returns:
            bool: True if the product was published successfully, False otherwise.
        """
        with self.lock_operations:
            
            # Pre-condition: Check if the producer has capacity to publish.
            if self.producersNrQueueSize[producer_id] - 1 > 0:
                
            	self.producersNrQueueSize[producer_id] -= 1
            	
            	self.productsgotid[product] = producer_id
            	self.listofproducts.append(product)
            	return True
            return False

    def new_cart(self):
        """
        Creates a new, empty shopping cart and returns its ID.

        This operation is thread-safe.

        Returns:
            int: The unique ID for the newly created cart.
        """
        with self.CartsLocks:
            self.cart_id += 1
            
            
            self.cartList[self.cart_id] = []
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product from the marketplace to a specific shopping cart.

        This operation is thread-safe. It atomically moves a product from the
        global product list to the cart's specific list.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (any): The product to add.

        Returns:
            bool: True if the product was available and added, False otherwise.
        """
        with self.LockAddRemCart:
            
            # Pre-condition: Check if the product exists in the marketplace.
            if product in self.listofproducts:
            	
                self.cartList[cart_id].append(product)
                
                # Restore the producer's capacity since the item is now in a cart.
                self.producersNrQueueSize[self.productsgotid[product]] += 1
                
                self.listofproducts.remove(product)
                return True
            return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart and returns it to the marketplace.

        This operation is thread-safe.

        Args:
            cart_id (int): The ID of the cart from which to remove the product.
            product (any): The product to remove.
        """
        with self.LockAddRemCart:
        	
            self.cartList[cart_id].remove(product)
            # The producer's capacity is reduced again as the product is returned.
            self.producersNrQueueSize[self.productsgotid[product]] -= 1
            self.listofproducts.append(product)

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.

        This thread-safe method prints the items being bought and returns the
        final list of products in the cart.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: The final list of products in the cart.
        """
        with self.ListSubmit:
            
            for i in self.cartList[cart_id]:
                
                print(currentThread().getName() + " bought " + str(i))
        return self.cartList[cart_id]

from time import sleep
from threading import Thread


class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.

    The producer runs in an infinite loop, continuously attempting to publish
    a predefined list of products according to specified quantities and timings.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer instance.

        Args:
            products (list): A list of products to be published. Each element is
                             a tuple containing (product, quantity, publish_interval).
            marketplace (Marketplace): The central marketplace object.
            republish_wait_time (float): Time in seconds to wait before retrying
                                         to publish if the marketplace is full.
            **kwargs: Keyword arguments for the parent Thread class.
        """
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        """
        The main execution method for the producer thread.

        Registers with the marketplace and enters an infinite loop to publish
        its products.
        """
        id_producer = self.marketplace.register_producer()
        # Invariant: The producer will continuously try to publish products.
        while True:
            
            # Invariant: Loop through all product types this producer can create.
            for type_prod in self.products:
                
                id_produs = type_prod[0]
                
                qty = type_prod[1]
                
                time_to_wait = type_prod[2]
                i = 0
                
                # Pre-condition: The 'qty' must be a positive integer.
                # Invariant: Publish the product 'qty' times.
                while i < qty:
                    ret = self.marketplace.publish(id_producer, id_produs)
                    if ret:
                        # If publish is successful, wait before publishing the next unit.
                        i += 1
                        sleep(time_to_wait)
                    else:
                        # If publish fails, wait and retry.
                        sleep(self.republish_wait_time)
