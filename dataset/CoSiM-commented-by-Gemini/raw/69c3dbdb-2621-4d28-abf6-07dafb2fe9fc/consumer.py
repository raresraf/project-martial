"""
This module simulates a simple e-commerce marketplace using a producer-consumer model.

It defines:
- A `Marketplace` class that acts as the central, thread-safe inventory and cart manager.
- A `Consumer` class that simulates customer shopping behavior in a separate thread.
- A `TestMarketplace` class containing unit tests to verify the marketplace logic.
"""

import time
from threading import Thread
from multiprocessing import Lock
from itertools import product
import logging
from logging.handlers import RotatingFileHandler
import unittest


class Consumer(Thread):
    """
    Represents a consumer that processes a list of shopping carts.

    Each consumer runs in its own thread, simulating a user adding and removing
    items from carts and finally placing an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of cart action lists. Each inner list contains
                          dictionaries representing 'add' or 'remove' actions.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Seconds to wait before retrying to add a
                                     product if it's out of stock.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        Thread.__init__(self, **kwargs)


    def run(self):
        """
        The main execution loop for the consumer thread.

        Iterates through each assigned cart, performs the specified add/remove
        operations, and places the order.
        """
        # Block Logic: This loop processes each shopping cart assigned to the consumer.
        for c in self.carts:
            c_id = self.marketplace.new_cart()
            # Block Logic: This loop executes the sequence of actions for a single cart.
            for req in c:
                type = req['type']
                prod = req['product']
                qty = req['quantity']
                if type == 'add':
                    # Block Logic: Attempts to add a product `qty` times.
                    for _ in range(0, qty):
                        prod_added_flag = self.marketplace.add_to_cart(c_id, prod)

                        # Invariant: Loop until the product is successfully added.
                        # This simulates waiting for a product to be restocked.
                        while prod_added_flag is False:
                            time.sleep(self.retry_wait_time)
                            prod_added_flag = self.marketplace.add_to_cart(c_id, prod)

                elif type == 'remove':
                    for _ in range(0, qty):
                       self.marketplace.remove_from_cart(c_id, prod)

            shopping_list = self.marketplace.place_order(c_id)

            # The consumer lock ensures that print statements from different
            # threads are not interleaved, making the output readable.
            self.marketplace.cons_lock.acquire()
            for elem in shopping_list:
                print("{} bought {}".format(self.getName(), elem))
            self.marketplace.cons_lock.release()


class Marketplace:
    """
    A thread-safe marketplace that manages producers, inventory, and customer carts.

    This class is the central shared resource, coordinating all operations between
    producers and consumers using locks to ensure data integrity.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products any
                                           single producer can have in stock.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.lock_prod = Lock()
        self.lock_cart = Lock()
        self.prod_id = 0
        self.lock = Lock()

        
        self.prod_list = {}

        
        self.carts_list = {}

        self.cart_id = 0
        self.cons_lock = Lock()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        RotatingFileHandler(filename='marketplace.log', maxBytes=50000, backupCount=20)
        self.logger.addHandler(RotatingFileHandler("marketplace.log"))

    def register_producer(self):
        """
        Registers a new producer, providing a unique producer ID.

        This operation is thread-safe.

        Returns:
            int: The newly assigned producer ID.
        """

        self.logger.info("Register producer was called")

        self.lock_prod.acquire()
        self.prod_id += 1
        self.prod_list[self.prod_id] = []
        self.logger.info("Register producer completed")
        self.lock_prod.release()

        return self.prod_id


    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace inventory.

        The operation will fail if the producer's inventory is full.

        Args:
            producer_id (int): The ID of the producer.
            product: The product to be added to inventory.

        Returns:
            bool: True if the product was published successfully, False otherwise.
        """

        self.logger.info("Publish was called")

        if len(self.prod_list[producer_id]) == self.queue_size_per_producer:
            self.logger.info("Publish returned False")
            return False

        self.prod_list[producer_id].append(product)
        return True


    def new_cart(self):
        """
        Creates a new, empty shopping cart and returns its unique ID.

        This operation is thread-safe.

        Returns:
            int: The newly assigned cart ID.
        """

        self.logger.info("New cart was called")

        self.lock_cart.acquire() 
        self.cart_id += 1
        self.carts_list[self.cart_id] = []
        self.logger.info("New cart was created successfully")
        self.lock_cart.release()

        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified shopping cart.

        Functional Intent: This method simulates moving stock. It searches all
        producer inventories for the requested product. If found, it is removed
        from the producer's stock and added to the consumer's cart.

        Args:
            cart_id (int): The ID of the cart to add to.
            product: The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        self.logger.info("Add to cart was called")

        # Searches for the product across all producer inventories.
        for i in self.prod_list:
            if product in self.prod_list[i]:

                self.prod_list[i].remove(product)
                self.carts_list[cart_id].append(tuple((i, product)))
                self.logger.info("Add to cart was made successfully")
                return True
        self.logger.info("Add to cart could not be completed")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart, returning it to the producer's inventory.

        This operation is thread-safe.

        Args:
            cart_id (int): The ID of the cart to remove from.
            product: The product to remove.
        """
        self.logger.info("Remove from cart was called")

        self.lock.acquire()
        # Block Logic: Find the product in the cart and return it to its original producer.
        for i in self.carts_list[cart_id]:

            if i[1] == product:
                self.prod_list[i[0]].append(product)
                self.carts_list[cart_id].remove(i)
                self.logger.info("Remove from cart was successful")
                break

        self.lock.release()
        return


    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.

        This action 'consumes' the items in the cart by returning them as a final
        list and clearing the cart.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: A list of the products that were in the cart.
        """
        self.logger.info("Place order was called")

        cart = self.carts_list[cart_id].copy()
        self.carts_list[cart_id] = []

        final_list = []
        for i in cart:
            final_list.append(i[1])
        self.logger.info("Place order created the list")
        return final_list


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self):
        """Sets up a new Marketplace instance and test data before each test."""
        self.marketplace = Marketplace(8)
        self.prod_list = {"name1":"Tea", "name2":"Coffee", "name3":"Wild berries Tea"}
        self.name_list = ["Tea", "Coffee", "Wild berries Tea"]
        self.list = ["Coffee", "Wild berries Tea"]

    def test_register_prod(self):
        """Tests that producer registration correctly increments the producer ID."""
        rez = self.marketplace.register_producer()
        self.assertEqual(rez, 1, 'Producer not updated')

    def test_publish(self):
        """Tests that a registered producer can successfully publish a product."""
        self.marketplace.register_producer()
        rez = self.marketplace.publish(1, self.prod_list["name3"])
        self.assertTrue(rez, 'Product is not published')

    def test_new_cart(self):
        """Tests that new cart creation correctly increments the cart ID."""
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        rez = self.marketplace.new_cart()
        self.assertEqual(rez, 4, 'Cart id not updated')

    def test_add_to_cart(self):
        """
        Tests adding products to a cart.

        Verifies that available products are added successfully and unavailable
        products are not, and that the final cart contents are correct.
        """
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.prod_list["name3"])
        self.marketplace.publish(1, self.prod_list["name2"])
        id_1 = self.marketplace.new_cart()
        id_2 = self.marketplace.new_cart()
        rez1 = self.marketplace.add_to_cart(id_1, self.prod_list["name1"])
        rez2 = self.marketplace.add_to_cart(id_1, self.prod_list["name2"])
        rez3 = self.marketplace.add_to_cart(id_1, self.prod_list["name3"])          
        self.assertEqual(id_2, 2, 'Cart id not set right')
        self.assertTrue(rez3, 'Product not added')
        self.assertFalse(rez1, 'Product not added')

        list_check = []
        for i in self.marketplace.carts_list[1]:
            list_check.append(i[1])

        self.assertListEqual(self.list, list_check, 'Not all products added are in list')

    def test_remove_from_cart(self):
        """Tests that removing a product from a cart works correctly."""
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.prod_list["name1"])
        self.marketplace.publish(1, self.prod_list["name2"])
        self.marketplace.publish(1, self.prod_list["name3"])
        id_1 = self.marketplace.new_cart()
        self.marketplace.add_to_cart(id_1, self.prod_list["name1"])
        self.marketplace.add_to_cart(id_1, self.prod_list["name2"])
        self.marketplace.remove_from_cart(id_1, self.prod_list["name1"])

        list_check = []
        for i in self.marketplace.carts_list[1]:
            list_check.append(i[1])

        self.assertListEqual(['Coffee'], list_check, 'List mismatch after remove')


    def test_place_order(self):
        """Tests the full lifecycle of adding, removing, and placing an order."""
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.prod_list["name1"])
        self.marketplace.publish(1, self.prod_list["name2"])
        self.marketplace.publish(1, self.prod_list["name3"])
        id_1 = self.marketplace.new_cart()
        self.marketplace.add_to_cart(id_1, self.prod_list["name1"])
        self.marketplace.add_to_cart(id_1, self.prod_list["name2"])
        self.marketplace.remove_from_cart(id_1, self.prod_list["name2"])
        self.marketplace.add_to_cart(id_1, self.prod_list["name3"])
        self.marketplace.add_to_cart(id_1, self.prod_list["name2"])
        self.marketplace.remove_from_cart(id_1, self.prod_list["name1"])

        list_check = []
        for i in self.marketplace.carts_list[1]:
            list_check.append(i[1])

        self.assertListEqual(['Wild berries Tea','Coffee'], list_check, 'List mismatch after placing order')
