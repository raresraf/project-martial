
"""
This module simulates a marketplace with producers and consumers acting in parallel.
It defines the core `Marketplace` logic, `Producer` and `Consumer` threads,
and the data structures for products, booths, and carts. The simulation uses
semaphores for synchronization to handle concurrent access to shared resources.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Represents a consumer thread that simulates a shopping experience in the marketplace.
    It processes a list of carts, each with a set of actions (add/remove products),
    and places an order for each cart.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of cart data structures, each containing a sequence of
                          product operations.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): The time in seconds to wait before retrying to add a
                                     product if it's not available.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def print_order(self, products):
        """
        Formats and prints the details of a placed order.

        Args:
            products (list): A list of products included in the order.
        """
        output = ""
        for product in products:
            output += self.name + ' bought ' + str(product) + '\n'
        output = output[:-1]
        print(output)

    def run(self):
        """
        The main execution loop for the consumer. It iterates through its assigned carts,
        executes the specified actions for each item, and places the order.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for item in cart:
                action = item['type']
                product = item['product']
                quantity = item['quantity']


                for _ in range(quantity):
                    if action == 'add':
                        # Attempt to add the product, retrying if it fails
                        ret_value = self.marketplace.add_to_cart(cart_id, product)
                        while not ret_value:
                            time.sleep(self.retry_wait_time)
                            ret_value = self.marketplace.add_to_cart(cart_id, product)
                    elif action == 'remove':
                        self.marketplace.remove_from_cart(cart_id, product)
            # Place the order after all actions for the current cart are done
            products = self.marketplace.place_order(cart_id)
            self.print_order(products)

import logging.handlers
import time
import unittest
from threading import Semaphore
from tema.product import Tea, Coffee


class Booth:
    """Represents a producer's booth within the marketplace, tracking product counts."""
    def __init__(self, producer):
        """
        Initializes a Booth.

        Args:
            producer: The ID of the producer owning this booth.
        """
        self.producer = producer
        self.num_products = 0
        self.num_products_mutex = Semaphore(1)

    def __eq__(self, other):
        """Checks equality based on the producer ID."""
        if not isinstance(other, Booth):
            return False
        return self.producer == other.producer


class Cart:
    """Represents a consumer's shopping cart, holding a list of products."""
    def __init__(self, cart_id):
        """
        Initializes a Cart.

        Args:
            cart_id: The unique identifier for the cart.
        """
        self.cart_id = cart_id
        self.products = []

    def __eq__(self, other):
        """Checks equality based on the cart ID."""
        if not isinstance(other, Cart):
            return False
        return self.cart_id == other.cart_id


class Marketplace:
    """
    A thread-safe marketplace that manages producers, consumers, and product inventory.
    It uses semaphores to ensure safe concurrent access to shared data structures like
    product lists and carts.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace and sets up logging.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have listed at one time.
        """
        handler = logging.handlers.RotatingFileHandler(filename='marketplace.log',
                                                       mode='a',
                                                       maxBytes=10000,
                                                       backupCount=1)
        logging.Formatter.converter = time.gmtime
        logging.basicConfig(
            handlers=[handler],
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%d-%m-%Y %H:%M:%S')
        logging.basicConfig(
            handlers=[handler],
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.ERROR,
            datefmt='%d-%m-%Y %H:%M:%S')
        self.queue_size_per_producer = queue_size_per_producer
        self.producers = {}
        self.num_producers = 0
        self.carts = {}
        self.num_carts = 0
        self.products = []
        self.shopping_mutex = Semaphore(1)
        self.carts_mutex = Semaphore(1)
        self.register_mutex = Semaphore(1)

    def register_producer(self):
        """
        Registers a new producer with the marketplace, assigning a unique ID.
        This operation is thread-safe.

        Returns:
            str: The unique ID assigned to the new producer.
        """
        logging.info('A producer wants to register')

        # Acquire lock to safely modify producer registry
        self.register_mutex.acquire()
        
        producer_id = self.num_producers
        self.producers[producer_id] = Booth(producer_id)
        self.num_producers += 1
        self.register_mutex.release()

        logging.info('A producer registered with id = ' + str(producer_id))
        return str(producer_id)

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.
        The operation fails if the producer's queue is full.

        Args:
            producer_id (str): The ID of the publishing producer.
            product (Product): The product to be published.

        Returns:
            bool: True if publishing was successful, False otherwise.
        """
        logging.info('Producer ' + producer_id + ' wants to publish product: ' + str(product))

        # Check if producer's booth queue is full
        pid = int(producer_id)
        booth = self.producers[pid]
        booth.num_products_mutex.acquire()
        if booth.num_products < self.queue_size_per_producer:
            # Acquire lock to safely modify the shared product list
            self.shopping_mutex.acquire()
            
            self.products.append((product, pid))
            booth.num_products += 1
            self.shopping_mutex.release()
            booth.num_products_mutex.release()
            logging.info('Producer ' + producer_id
                         + ' published product ' + str(product) + ' successfully')
            return True

        # Log failure if queue is full
        logging.error('Producer ' + producer_id
                      + ' could not publish product, because its queue is full')
        booth.num_products_mutex.release()
        return False

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.
        This operation is thread-safe.

        Returns:
            int: The unique ID of the newly created cart.
        """
        logging.info('A consumer wants a new cart')

        # Acquire lock to safely create a new cart
        self.carts_mutex.acquire()
        
        cart_id = self.num_carts
        self.carts[cart_id] = Cart(cart_id)
        self.num_carts += 1
        self.carts_mutex.release()

        logging.info('The consumer got a new cart')
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a specified product to a consumer's cart.
        It finds the product in the global list and moves it to the cart.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (Product): The product to add.

        Returns:
            bool: True if the product was successfully added, False otherwise.
        """
        logging.info('Consumer with cart id ' + str(cart_id)
                     + ' wants to add to cart the product ' + str(product))

        # Acquire lock to safely search and modify the product list
        self.shopping_mutex.acquire()
        for i in range(len(self.products)):
            if product == self.products[i][0]:
                # Move product from marketplace to cart
                self.carts[cart_id].products.append(self.products[i])
                self.products = self.products[:i] + self.products[i+1:]
                self.shopping_mutex.release()
                logging.info('Consumer with cart id ' + str(cart_id)
                             + ' added to cart the product ' + str(product) + ' successfully')
                return True

        # Release lock if product not found
        self.shopping_mutex.release()
        logging.error('Consumer with cart id ' + str(cart_id)
                      + ' could not add to cart the product ' + str(product))
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a specified product from a consumer's cart, returning it to the
        marketplace's global product list.

        Args:
            cart_id (int): The ID of the cart from which to remove the product.
            product (Product): The product to remove.
        """
        logging.info('Consumer with cart id ' + str(cart_id)
                     + ' wants to remove the product ' + str(product) + ' from the cart')

        # Acquire lock to safely modify cart and product lists
        self.shopping_mutex.acquire()
        for i in range(len(self.carts[cart_id].products)):
            if product == self.carts[cart_id].products[i][0]:
                # Move product from cart back to marketplace
                self.products.append(self.carts[cart_id].products[i])
                self.carts[cart_id].products = self.carts[cart_id].products[:i] \
                                               + self.carts[cart_id].products[i+1:]
                self.shopping_mutex.release()

                logging.info('Consumer with cart id ' + str(cart_id)
                             + ' removed product ' + str(product) + ' from the cart')
                break

    def place_order(self, cart_id):
        """
        Finalizes an order for all items in a cart. This action removes the items
        from the system and updates the producer's product count.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: A list of the products that were in the cart.
        """
        logging.info('Consumer with cart id ' + str(cart_id) + ' wants to place the order')

        cart = self.carts[cart_id].products
        products = []
        for (product, producer_id) in cart:
            products.append(product)
            
            # Update the producer's product count
            booth = self.producers[producer_id]
            booth.num_products_mutex.acquire()
            booth.num_products -= 1
            booth.num_products_mutex.release()

        # Delete the cart after the order is placed
        del self.carts[cart_id]

        logging.info('Consumer with cart id ' + str(cart_id) + ' placed the order successfully')
        return products


class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self) -> None:
        """Initializes a new Marketplace instance before each test."""
        self.marketplace = Marketplace(3)

    def test_register_producer(self):
        """Tests that producer registration assigns sequential IDs correctly."""
        self.assertEqual(self.marketplace.register_producer(), '0')
        self.assertEqual(self.marketplace.register_producer(), '1')
        self.assertEqual(self.marketplace.register_producer(), '2')
        self.assertEqual(self.marketplace.num_producers, 3)
        self.assertEqual(self.marketplace.producers, {0: Booth(0), 1: Booth(1), 2: Booth(2)})

    def test_publish(self):
        """Tests the product publishing logic, including queue limits."""
        p_1 = Tea(name='Linden', type='Herbal', price=9)
        pid = self.marketplace.register_producer()
        products = [(p_1, int(pid)), (p_1, int(pid)), (p_1, int(pid))]

        self.assertEqual(self.marketplace.publish(pid, p_1), True)
        self.assertEqual(self.marketplace.publish(pid, p_1), True)
        self.assertEqual(self.marketplace.publish(pid, p_1), True)
        # Should fail as the queue size (3) is reached
        self.assertEqual(self.marketplace.publish(pid, p_1), False)
        self.assertEqual(self.marketplace.producers[int(pid)].num_products, 3)
        self.assertEqual(self.marketplace.products, products)

    def test_new_cart(self):
        """Tests that new carts are created with sequential IDs."""
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.new_cart(), 1)
        self.assertEqual(self.marketplace.new_cart(), 2)
        self.assertEqual(self.marketplace.num_carts, 3)
        self.assertEqual(self.marketplace.carts, {0: Cart(0), 1: Cart(1), 2: Cart(2)})

    def test_add_to_cart(self):
        """Tests adding products to a cart, including unavailable products."""
        p_1 = Tea(name='Linden', type='Herbal', price=9)
        p_2 = Coffee(name='Indonezia', acidity='5.05', roast_level='MEDIUM', price=1)

        pid = self.marketplace.register_producer()
        self.marketplace.publish(pid, p_1)

        cart_id = self.marketplace.new_cart()
        self.assertEqual(self.marketplace.add_to_cart(cart_id, p_1), True)
        # Fails because the product is already in a cart
        self.assertEqual(self.marketplace.add_to_cart(cart_id, p_1), False)
        # Fails because the product has not been published
        self.assertEqual(self.marketplace.add_to_cart(cart_id, p_2), False)

        products = [(p_1, int(pid))]
        self.assertEqual(self.marketplace.carts[cart_id].products, products)
        self.assertEqual(self.marketplace.products, [])

    def test_remove_from_cart(self):
        """Tests removing a product from a cart."""
        p_1 = Tea(name='Linden', type='Herbal', price=9)
        p_2 = Coffee(name='Indonezia', acidity='5.05', roast_level='MEDIUM', price=1)

        pid = self.marketplace.register_producer()
        self.marketplace.publish(pid, p_1)


        self.marketplace.publish(pid, p_1)
        self.marketplace.publish(pid, p_2)

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, p_1)
        self.marketplace.add_to_cart(cart_id, p_1)
        self.marketplace.add_to_cart(cart_id, p_2)

        products = [(p_1, int(pid)), (p_2, int(pid))]
        self.marketplace.remove_from_cart(cart_id, p_1)
        self.assertEqual(self.marketplace.carts[cart_id].products, products)
        self.assertEqual(self.marketplace.products, [(p_1, int(pid))])

    def test_place_order(self):
        """Tests the order placement logic and verifies inventory updates."""
        p_1 = Tea(name='Linden', type='Herbal', price=9)
        p_2 = Coffee(name='Indonezia', acidity='5.05', roast_level='MEDIUM', price=1)

        pid = self.marketplace.register_producer()
        self.marketplace.publish(pid, p_1)
        self.marketplace.publish(pid, p_1)
        self.marketplace.publish(pid, p_2)

        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, p_1)
        self.marketplace.add_to_cart(cart_id, p_1)
        self.marketplace.add_to_cart(cart_id, p_2)

        self.assertEqual(self.marketplace.place_order(cart_id), [p_1, p_1, p_2])
        self.assertEqual(self.marketplace.producers[int(pid)].num_products, 0)
        self.assertEqual(self.marketplace.carts, {})


from threading import Thread
import time

class Producer(Thread):
    """
    Represents a producer thread that publishes a predefined list of products
    to the marketplace at a specified frequency.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the Producer thread.

        Args:
            products (list): A list of items to produce, where each item is a tuple of
                             (product, quantity, wait_time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): The time to wait before retrying a publish
                                         operation if the queue is full.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        The main execution loop for the producer. It continuously cycles through its
        product list, publishing each item according to its specified quantity and
        waiting period.
        """
        pid = self.marketplace.register_producer()
        while True:
            for item in self.products:
                product = item[0]
                quantity = item[1]
                waiting = item[2]
                for _ in range(quantity):
                    # Attempt to publish, retrying on failure (e.g., full queue)
                    ret_value = self.marketplace.publish(pid, product)
                    while not ret_value:
                        time.sleep(self.republish_wait_time)
                        ret_value = self.marketplace.publish(pid, product)
                    time.sleep(waiting)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base data class for products, containing a name and a price."""
    name: str
    price: int

    def __eq__(self, other):
        pass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A data class representing Tea, a specific type of Product."""
    type: str

    def __eq__(self, other):
        """Custom equality check for Tea objects."""
        if not isinstance(other, Tea):
            return False
        return other.name == self.name and \
                    other.price == self.price and \
                    other.type == self.type


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A data class representing Coffee, with specific attributes."""
    acidity: str
    roast_level: str

    def __eq__(self, other):
        """Custom equality check for Coffee objects."""
        if not isinstance(other, Coffee):
            return False
        return other.name == self.name and \
                    other.price == self.price and \
                    other.acidity == self.acidity and \
                    other.roast_level == self.roast_level
