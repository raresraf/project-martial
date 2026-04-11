"""
This module implements a simulation of a marketplace with producers and consumers.

It defines the core components of the simulation:
- Product: Dataclasses representing products available in the marketplace.
- Marketplace: The central class that manages producers, products, and customer carts
  using thread-safe operations.
- Producer: A thread that publishes products to the marketplace.
- Consumer: A thread that adds products to a cart and places an order.
- TestMarketplace: A suite of unittest cases to verify the functionality of the Marketplace.
"""

import time
from threading import Thread
import logging
from logging.handlers import RotatingFileHandler
from threading import Lock
import unittest
from dataclasses import dataclass

# Assuming product definitions are in a 'tema' package, which is not present.
# The following are placeholder classes to make the code runnable.
@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base class for a product with a name and price."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """Represents a Tea product, inheriting from Product and adding a type."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """Represents a Coffee product, inheriting from Product with acidity and roast level."""
    acidity: str
    roast_level: str


class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace.

    A consumer thread processes a list of carts, where each cart contains
    a series of operations (add/remove products). It simulates the behavior

    of a customer shopping in the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of cart operations for the consumer to process.
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (float): Time to wait before retrying to add a product
                                     if it's not immediately available.
            **kwargs: Keyword arguments for the Thread class, including 'name'.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution logic for the consumer thread.

        Iterates through its assigned carts, executes the add/remove operations
        for each product, and finally places the order.
        """
        # Pre-condition: Iterate through each shopping cart assigned to this consumer.
        for cart in self.carts:
            # Each consumer gets a new unique cart ID from the marketplace.
            cart_id = self.marketplace.new_cart()
            # Invariant: Process all operations within a single cart.
            for operation in cart:
                if operation['type'] == 'add':
                    # Block Logic: Add a specified quantity of a product to the cart.
                    for _ in range(operation['quantity']):
                        success = self.marketplace.add_to_cart(cart_id, operation['product'])
                        # Invariant: If a product is not available, retry until it is.
                        # This models a persistent consumer waiting for a product to be restocked.
                        while not success:
                            time.sleep(self.retry_wait_time)
                            success = self.marketplace.add_to_cart(cart_id, operation['product'])
                else:
                    # Block Logic: Remove a specified quantity of a product from the cart.
                    for _ in range(operation['quantity']):
                        self.marketplace.remove_from_cart(cart_id, operation['product'])

            # After processing all operations, place the final order.
            orders = self.marketplace.place_order(cart_id)
            for order in orders:
                print(self.kwargs['name'], 'bought', order)

class Marketplace:
    """
    Manages the inventory and transactions between producers and consumers.

    This class provides a thread-safe environment for producers to publish
    products and for consumers to shop. It uses locks to protect shared
    data structures related to producers, products, and carts.

    Attributes:
        queue_size_per_producer (int): The maximum number of products a single
                                       producer can have in the marketplace at one time.
        producers_slots (dict): Maps producer_id to the number of available slots.
        producers_queues (dict): Maps producer_id to a list of their published products.
        carts (dict): Maps cart_id to the contents of a shopping cart.
        next_producer_id (int): A counter for generating unique producer IDs.
        next_cart_id (int): A counter for generating unique cart IDs.
        slots_locks (dict): Locks for protecting access to `producers_slots`.
        queues_locks (dict): Locks for protecting access to `producers_queues`.
        producer_id_lock (Lock): Lock for atomically generating new producer IDs.
        cart_id_lock (Lock): Lock for atomically generating new cart IDs.
        logger (Logger): A logger for recording marketplace events.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): Max products per producer.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producers_slots = {}
        self.producers_queues = {}
        self.carts = {}
        self.next_producer_id = 0
        self.next_cart_id = 0

        # Architectural Pattern: Fine-grained locking. Different locks are used for
        # different data structures to reduce contention between threads.
        self.slots_locks = {}
        self.queues_locks = {}
        self.producer_id_lock = Lock()
        self.cart_id_lock = Lock()

        # Setup logging for marketplace events.
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('marketplace.log', maxBytes=20000, backupCount=5)
        self.logger.addHandler(handler)

    def register_producer(self):
        """
        Registers a new producer, providing them with a unique ID and inventory space.

        Returns:
            str: The unique ID assigned to the new producer.
        """
        # Synchronization: The producer_id_lock ensures that the generation of a new
        # producer ID is an atomic operation, preventing race conditions.
        self.producer_id_lock.acquire()
        producer_id = str(self.next_producer_id)
        self.next_producer_id += 1
        self.producer_id_lock.release()

        # Initialize data structures for the new producer.
        self.producers_slots[producer_id] = self.queue_size_per_producer
        self.producers_queues[producer_id] = []
        self.slots_locks[producer_id] = Lock()
        self.queues_locks[producer_id] = Lock()

        self.logger.info('Register producer: producer_id = %s', producer_id)
        return producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        Args:
            producer_id (str): The ID of the producer publishing the product.
            product (Product): The product to be published.

        Returns:
            bool: True if the product was published successfully, False if the
                  producer's queue is full.
        """
        # Error Handling: Check if the producer has available slots before publishing.
        # This is a non-blocking check on the number of available slots.
        self.slots_locks[producer_id].acquire()
        if self.producers_slots[producer_id] == 0:
            self.logger.info('Publish product: producer_id = %s, product = %s, 
                            return = False', producer_id, product)
            self.slots_locks[producer_id].release()
            return False
        self.slots_locks[producer_id].release()

        # Add the product to the producer's queue.
        self.queues_locks[producer_id].acquire()
        self.producers_queues[producer_id].append(product)
        self.queues_locks[producer_id].release()

        # Decrement the number of available slots for the producer.
        self.slots_locks[producer_id].acquire()
        self.producers_slots[producer_id] -= 1
        self.slots_locks[producer_id].release()

        self.logger.info('Publish product: producer_id = %s, product = %s, 
                        return = True ', producer_id, product)
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.

        Returns:
            int: The unique ID for the newly created cart.
        """
        # Synchronization: Atomically generate a new cart ID.
        self.cart_id_lock.acquire()
        cart_id = self.next_cart_id
        self.next_cart_id = self.next_cart_id + 1
        self.cart_id_lock.release()

        self.carts[cart_id] = {}

        self.logger.info('New cart: cart_id = %d', cart_id)
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified shopping cart.

        This function searches through all producer queues to find the requested
        product. If found, it moves the product from the producer's queue to
        the consumer's cart.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (Product): The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        # Block Logic: Iterate through all producers to find the desired product.
        for producer in self.producers_queues:
            # Synchronization: Lock the producer's queue while searching and removing.
            self.queues_locks[producer].acquire()
            if product in self.producers_queues[producer]:
                self.producers_queues[producer].remove(product)
                self.queues_locks[producer].release()

                # If this is the first product from this producer, initialize their list in the cart.
                if not producer in self.carts[cart_id].keys():
                    self.carts[cart_id][producer] = []
                self.carts[cart_id][producer].append(product)

                self.logger.info('Add to cart: cart_id = %d, product = %s, 
                                return = True', cart_id, product)
                return True
            self.queues_locks[producer].release()

        # If the loop completes without finding the product.
        self.logger.info('Add to cart: cart_id = %d, product = %s, 
                        return = False', cart_id, product)
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart and returns it to the producer's queue.

        Args:
            cart_id (int): The ID of the cart.
            product (Product): The product to remove.
        """
        for producer in self.carts[cart_id]:
            if product in self.carts[cart_id][producer]:
                # Remove the product from the cart.
                self.carts[cart_id][producer].remove(product)
                
                # Return the product to the original producer's queue.
                self.queues_locks[producer].acquire()
                self.producers_queues[producer].append(product)
                self.queues_locks[producer].release()

                self.logger.info('Remove from cart: cart_id = %d, product = %s', cart_id, product)
                break

    def place_order(self, cart_id):
        """
        Finalizes an order, returning products and freeing up producer slots.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: A list of all products that were in the cart.
        """
        returned_list = []

        # Block Logic: Iterate through the producers represented in the cart.
        for producer in self.carts[cart_id]:
            for product in self.carts[cart_id][producer]:
                returned_list.append(product)
                
                # Synchronization: Atomically increment the producer's available slots,
                # as the product has now been sold.
                self.slots_locks[producer].acquire()
                self.producers_slots[producer] += 1
                self.slots_locks[producer].release()

        # The cart is now empty and can be deleted.
        del self.carts[cart_id]

        self.logger.info('Place order: cart_id = %d,returned list = %s', cart_id, returned_list)
        return returned_list


class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace class.
    """

    def setUp(self):
        """Set up a new marketplace and some products for each test."""
        self.marketplace = Marketplace(3)
        self.products = []
        self.products.append(Tea('Musetel', 1, 'Herbal'))
        self.products.append(Tea('Coada Soricelului', 3, 'Herbal'))
        self.products.append(Coffee('Espresso', 2, '10.0', 'HIGH'))
        self.products.append(Tea('Urechea boului', 4, 'Non-herbal'))

    def test_register_producer(self):
        """Test that producers are registered with unique, sequential IDs."""
        for i in range(0, 10):
            self.assertEqual(self.marketplace.register_producer(), str(i))

    def test_new_cart(self):
        """Test that new carts are created with unique, sequential IDs."""
        for i in range(0, 10):
            self.assertEqual(self.marketplace.new_cart(), i)

    def test_publish(self):
        """Test that products can be published to different producers' queues."""
        for _ in range(0, 4):
            self.marketplace.register_producer()

        self.marketplace.publish('0', self.products[0])
        self.marketplace.publish('0', self.products[1])
        self.marketplace.publish('2', self.products[0])
        self.marketplace.publish('0', self.products[0])
        self.marketplace.publish('1', self.products[3])

        self.assertEqual(self.marketplace.producers_queues['0'],
                         [self.products[0], self.products[1], self.products[0]])
        self.assertEqual(self.marketplace.producers_queues['1'], [self.products[3]])
        self.assertEqual(self.marketplace.producers_queues['2'], [self.products[0]])
        self.assertEqual(self.marketplace.producers_queues['3'], [])

    def test_publish_fails(self):
        """Test that publishing fails when a producer's queue is full."""
        self.marketplace.register_producer()
        for i in range(0, 3):
            self.assertTrue(self.marketplace.publish(str(0), self.products[i]))
        # The 4th publish should fail as the queue size is 3.
        self.assertFalse(self.marketplace.publish(str(0), self.products[0]))

    def test_add_to_cart(self):
        """Test adding products to the cart and ensure they are removed from producer queues."""
        # Setup: Create carts, producers, and publish products.
        for i in range(0, 3):
            self.marketplace.new_cart()
            self.marketplace.register_producer()
        for i in range(0, 3):
            for _ in range(0, 3):
                self.marketplace.publish(str(i), self.products[i])

        # Test: Add products to carts.
        for i in range(0, 3):
            for _ in range(0, 3):
                self.assertTrue(self.marketplace.add_to_cart(i, self.products[i]))

        # Verification: Adding again should fail as products are gone.
        for i in range(0, 3):
            self.assertFalse(self.marketplace.add_to_cart(i, self.products[i]))
        
        # Verification: Check cart contents.
        for i in range(0, 3):
            self.assertEqual(self.marketplace.carts[i][str(i)],
                             [self.products[i], self.products[i], self.products[i]])

    def test_remove_from_cart(self):
        """Test that removing a product from a cart returns it to the producer's inventory."""
        id_cart = self.marketplace.new_cart()
        id_producer = self.marketplace.register_producer()
        for i in range(0, 3):
            self.marketplace.publish('0', self.products[i])
            self.marketplace.add_to_cart(0, self.products[i])

        self.marketplace.remove_from_cart(id_cart, self.products[1])
        self.assertEqual(self.marketplace.carts[id_cart][id_producer],
                         [self.products[0], self.products[2]])
        self.marketplace.remove_from_cart(id_cart, self.products[0])
        self.assertEqual(self.marketplace.carts[id_cart][id_producer], [self.products[2]])
        self.marketplace.remove_from_cart(id_cart, self.products[2])
        self.assertEqual(self.marketplace.carts[id_cart][id_producer], [])

    def test_place_order(self):
        """Test that placing an order correctly returns the products and frees producer slots."""
        # Setup: Create a cart, a producer, and add items.
        cart_id = self.marketplace.new_cart()
        producer_id = self.marketplace.register_producer()
        for i in range(0, 3):
            self.marketplace.publish(producer_id, self.products[i])
            self.marketplace.add_to_cart(cart_id, self.products[i])

        returned_list = self.marketplace.place_order(cart_id)
        
        # Verification: Check that the returned list matches the cart contents.
        self.assertEqual(returned_list, [self.products[0], self.products[1], self.products[2]])
        
        # Verification: Check that producer slots are restored.
        self.assertEqual(self.marketplace.producers_slots[producer_id], 3)


class Producer(Thread):
    """
    Represents a producer that publishes products to the marketplace.

    A producer thread continuously tries to publish a list of products
    to the marketplace, waiting for a specified time between each publication.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of tuples, where each tuple contains
                             a product, quantity, and publication delay.
            marketplace (Marketplace): The marketplace instance to interact with.
            republish_wait_time (float): Time to wait before retrying to publish
                                         if the marketplace queue is full.
            **kwargs: Keyword arguments for the Thread class.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution logic for the producer thread.

        It registers with the marketplace and then enters an infinite loop,
        publishing its assigned products.
        """
        producer_id = self.marketplace.register_producer()
        
        # Invariant: The producer runs in an infinite loop, continuously supplying products.
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    time.sleep(product[2])
                    success = self.marketplace.publish(producer_id, product[0])
                    # Invariant: If publishing fails (e.g., queue is full),
                    # wait and retry.
                    while not success:
                        time.sleep(self.republish_wait_time)
                        success = self.marketplace.publish(producer_id, product[0])
