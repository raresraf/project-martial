"""
This module implements a multi-threaded simulation of an e-commerce marketplace.

It defines three main components:
- `Marketplace`: A central class that manages shared resources, including product
  inventories from different producers and customer shopping carts.
- `Producer`: A thread that simulates a manufacturer creating products and
  publishing them to the marketplace.
- `Consumer`: A thread that simulates a customer adding and removing products
  from a cart and eventually placing an order.

The simulation uses locks for some atomic operations but relies heavily on
polling with `sleep` for others, and contains potential race conditions in
its handling of shared product queues and carts. The file also includes a
`unittest` suite for the `Marketplace` class.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    Represents a consumer thread that simulates a shopping process.

    Each consumer is given a list of shopping lists ('carts'), and for each one,
    it interacts with the marketplace to add/remove products and place an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping operations to perform.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying an operation.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution loop for the consumer.

        Iterates through each assigned shopping cart, performs the add/remove
        operations, places the order, and prints the purchased products.
        """
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
        """
        Repeatedly tries to add a specified quantity of a product to a cart.

        This method uses a polling mechanism, retrying the operation until it
        succeeds.
        """
        for _ in range(quantity):
            while True:
                added = self.marketplace.add_to_cart(cart_id, product_id)
                if added:
                    break
                sleep(self.retry_wait_time)

    def remove_cart(self, cart_id, product_id, quantity):
        """
        Repeatedly tries to remove a specified quantity of a product from a cart.

        This method uses a polling mechanism, retrying the operation until it
        succeeds.
        """
        for _ in range(quantity):
            while True:
                removed = self.marketplace.remove_from_cart(cart_id, product_id)
                if removed:
                    break
                sleep(self.retry_wait_time)


from threading import Lock
import unittest
import sys
# This suggests the test depends on a 'tema' directory at the same level.
sys.path.insert(1, './tema')
import product as produs

class Marketplace:
    """
    Manages producers, products, and carts as a central shared resource.

    This class is intended to be thread-safe, but its current implementation has
    potential race conditions. For example, `add_to_cart` checks for a product's
    existence and then removes it without locking the sequence, making it
    vulnerable to race conditions between multiple consumer threads.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products each
                                           producer can have in their queue.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0


        self.cart_id = 0
        self.queues = []
        self.carts = []
        self.mutex = Lock()
        self.products_dict = {}

    def register_producer(self):
        """
        Registers a new producer, giving them a unique ID and a product queue.

        Returns:
            str: The unique ID assigned to the new producer.
        """
        self.mutex.acquire()
        producer_id = self.producer_id
        self.producer_id += 1
        self.queues.append([])
        self.mutex.release()
        return str(producer_id)

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to their inventory queue.

        Args:
            producer_id (str): The ID of the publishing producer.
            product: The product to be published.

        Returns:
            bool: True if publishing was successful, False if the queue was full.
        """
        index_prod = int(producer_id)
        if len(self.queues[index_prod]) == self.queue_size_per_producer:
            return False
        self.queues[index_prod].append(product)
        self.products_dict[product] = index_prod
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.

        Returns:
            int: The unique ID assigned to the new cart.
        """
        self.mutex.acquire()


        cart_id = self.cart_id
        self.cart_id += 1
        self.mutex.release()
        self.carts.append([])
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's cart if it is available in any producer's
        inventory.

        Args:
            cart_id (int): The ID of the cart to add to.
            product: The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        prod_in_queue = False
        # Atomicity hazard: This iteration is not protected by a lock. Two
        # consumers could find the same product and both attempt to remove it.
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
        """
        Removes a product from a cart and returns it to the original producer's
        inventory queue.

        Args:
            cart_id (int): The ID of the cart to remove from.
            product: The product to remove.

        Returns:
            bool: True if the product was in the cart and successfully returned
                  to the producer's queue, False otherwise.
        """
        if product not in self.carts[cart_id]:
            return False
        index_producer = self.products_dict[product]
        if len(self.queues[index_producer]) == self.queue_size_per_producer:
            return False
        self.carts[cart_id].remove(product)
        self.queues[index_producer].append(product)
        return True

    def place_order(self, cart_id):
        """
        Finalizes a purchase by taking all products from a cart.

        Args:
            cart_id (int): The ID of the cart to place an order for.

        Returns:
            list: The list of products that were in the cart.
        """
        cart_product_list = self.carts[cart_id]
        self.carts[cart_id] = []
        return cart_product_list

class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace class.

    These tests validate the behavior of the marketplace in a single-threaded
    context, ensuring that the core logic for registration, publishing, and
    cart manipulation works as expected under ideal conditions.
    """
    def setUp(self):
        """Initializes a new Marketplace instance before each test."""
        self.marketplace = Marketplace(4)

    def test_register_producer(self):
        """Tests the unique ID generation for producer registration."""
        self.assertEqual(self.marketplace.register_producer(), str(0))
        self.assertNotEqual(self.marketplace.register_producer(), str(3))
        self.assertEqual(self.marketplace.register_producer(), str(2))
        self.assertNotEqual(self.marketplace.register_producer(), str(0))
        self.assertNotEqual(self.marketplace.register_producer(), str(3))
        self.assertNotEqual(self.marketplace.register_producer(), str(2))

    def test_publish(self):
        """Tests the product publishing logic, including queue size limits."""
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
        """Tests the unique ID generation for new carts."""
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertNotEqual(self.marketplace.new_cart(), 3)
        self.assertEqual(self.marketplace.new_cart(), 2)
        self.assertNotEqual(self.marketplace.new_cart(), 0)
        self.assertNotEqual(self.marketplace.new_cart(), 3)
        self.assertNotEqual(self.marketplace.new_cart(), 2)

    def test_add_to_cart(self):
        """Tests adding a product to a cart."""
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal"))
        self.assertTrue(self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal")))
        self.assertFalse(self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal")))

    def test_remove_from_cart(self):
        """Tests removing a product from a cart."""
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal"))
        self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal"))
        self.assertTrue(self.marketplace.remove_from_cart(0, produs.Tea("Linden", 9, "Herbal")))
        self.assertFalse(self.marketplace.remove_from_cart(0, produs.Tea("Linden", 9, "Herbal")))

    def test_place_order(self):
        """Tests the final order placement logic."""
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        self.marketplace.publish(str(0), produs.Tea("Linden", 9, "Herbal"))
        self.marketplace.add_to_cart(0, produs.Tea("Linden", 9, "Herbal"))
        self.assertEqual([produs.Tea("Linden", 9, "Herbal")], self.marketplace.place_order(0))


from threading import Thread
from time import sleep

class Producer(Thread):
    """
    Represents a producer thread that continuously creates and publishes products.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products to be produced.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        
        
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution loop for the producer.

        Continuously loops through its product list and attempts to publish each one.
        """
        while True:
            for product in self.products:
                quantity = product[1]
                for _ in range(0, quantity):
                    self.publish_product(product[0], product[2])

    def publish_product(self, product, production_time):
        """
        Handles the publishing of a single product with a retry mechanism.

        After successfully publishing, it simulates production time by sleeping.
        """
        while True:
            published = self.marketplace.publish(self.producer_id, product)
            if published:
                sleep(production_time)
                break
            sleep(self.republish_wait_time)
