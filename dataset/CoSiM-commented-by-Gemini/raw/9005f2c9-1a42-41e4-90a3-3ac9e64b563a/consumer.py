"""
This module contains a full simulation of a producer-consumer marketplace.

@note This file appears to be a concatenation of multiple logical files,
including `consumer.py`, `marketplace.py`, `producer.py`, and definitions
for the data model. The documentation will treat it as a single module.

The simulation includes:
- The `Marketplace` class: The central, thread-safe hub for all transactions.
- A `Producer` thread class that creates and publishes products.
- A `Consumer` thread class that simulates users buying products.
- Simple data classes for `Producer`, `Product`, and `Cart`.
- Unit tests for the marketplace functionality.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    Represents a consumer thread that simulates purchasing products from the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping lists for the consumer to process.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying an operation.
            **kwargs: Keyword arguments for the parent Thread class.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution logic for the consumer thread.

        Processes each assigned shopping list by creating a cart, performing add/remove
        operations with retries, and finally placing the order.
        """
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()

            
            for action in cart:

                
                for _ in range(action['quantity']):
                    if action['type'] == "add":
                        return_value = self.marketplace.add_to_cart(cart_id, action['product'])
                    else:
                        return_value = self.marketplace\
                            .remove_from_cart(cart_id, action['product'])
                    time.sleep(self.retry_wait_time)

                    # Invariant: Keep retrying the operation until it succeeds.
                    while return_value == False:
                        time.sleep(self.retry_wait_time)

                        if action['type'] == "add":
                            return_value = self.marketplace\
                                .add_to_cart(cart_id, action['product'])
                        else:
                            return_value = self.marketplace\
                                .remove_from_cart(cart_id, action['product'])

            
            self.marketplace.place_order(cart_id)


from threading import currentThread, Lock


import unittest
import time
import logging
from logging.handlers import RotatingFileHandler
logging.basicConfig(
        handlers=[RotatingFileHandler('./marketplace.log', maxBytes=150000, backupCount=15)],
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S')
logging.Formatter.converter = time.gmtime

class Producer:
    """A simple data class to hold information about a producer."""
    def __init__(self, producer_id, nr_items):
        """
        Initializes a Producer data object.

        Args:
            producer_id (int): The unique ID of the producer.
            nr_items (int): The number of items this producer has published.
        """
        self.producer_id = producer_id 
        self.nr_items = nr_items 

class Product:
    """A simple data class to hold information about a product type in the marketplace."""
    def __init__(self, details, producer_id, quantity):
        """
        Initializes a Product data object.

        Args:
            details: The product information (e.g., name, type).
            producer_id (int): The ID of the producer who published this product.
            quantity (int): The available quantity of this product type.
        """
        self.details = details 
        self.producer_id = producer_id 
        self.quantity = quantity 

class Cart:
    """A simple data class to hold information about a shopping cart."""
    def __init__(self, cart_id, products):
        """
        Initializes a Cart data object.

        Args:
            cart_id (int): The unique ID of the cart.
            products (list): A list of products currently in the cart.
        """
        self.cart_id = cart_id 
        self.products = products 

def get_index_of_product(product, list_of_products):
    """
    Helper function to find the index of a product type in a list.

    Args:
        product: The product details to search for.
        list_of_products (list): The list of Product objects to search within.

    Returns:
        int: The index of the product if found, otherwise -1.
    """
    idx = 0
    for element in list_of_products:
        if product == element.details:
            return idx
        idx += 1
    return -1

class Marketplace:
    """
    A thread-safe marketplace that tracks inventory by product type and quantity.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of items any
                                           single producer can list at a time.
        """
        self.limit_per_producer = queue_size_per_producer  
        self.producers = [] 
        self.products = [] 
        self.carts = [] 
        self.new_producer_lock = Lock()
        self.new_cart_lock = Lock()
        self.add_to_cart_lock = Lock()
        self.remove_from_cart_lock = Lock()
        self.publish_lock = Lock()
        self.pint_lock = Lock()



    def register_producer(self):
        """
        Registers a new producer with the marketplace.

        Returns:
            int: The unique ID for the registered producer.
        """
        with self.new_producer_lock:
            
            self.producers.append(Producer(len(self.producers), 0))
            logging.info(f'FROM "register_producer" ->'
                         f' output: producer_id = {len(self.producers) - 1}')
            return len(self.producers) - 1

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace, increasing its quantity.

        If the product type already exists, its quantity is incremented.
        Otherwise, a new product type is added to the inventory.

        Args:
            producer_id (int): The ID of the publishing producer.
            product: The product details to be published.

        Returns:
            bool: True if successful, False if the producer's limit is reached.
        """
        logging.info(f'FROM "publish" ->'
                     f' input: producer_id = {producer_id}, product = {product}')

        
        if self.producers[producer_id].nr_items >= self.limit_per_producer:
            logging.info(f'FROM "publish" ->'
                         f' output: False')
            return False

        with self.publish_lock:
            
            self.producers[producer_id].nr_items += 1

            
            idx_product = get_index_of_product(product, self.products)
            if idx_product == -1:
                self.products.append(Product(product, producer_id, 1))
            else:
                self.products[idx_product].quantity += 1
                self.products[idx_product].producer_id = producer_id

        logging.info(f'FROM "publish" ->'
                     f' output: True')
        return True


    def new_cart(self):
        """
        Creates a new, empty shopping cart.

        Returns:
            int: The unique ID for the new cart.
        """
        with self.new_cart_lock:
            
            self.carts.append(Cart(len(self.carts), []))
            logging.info(f'FROM "new_cart" ->'
                         f' output: cart_id: {len(self.carts) - 1}')
            return len(self.carts) - 1

    def add_to_cart(self, cart_id, product):
        """
        Adds one unit of a product type to a shopping cart.

        Decrements the available quantity of the product in the marketplace.

        Args:
            cart_id (int): The ID of the target cart.
            product: The product details to add.

        Returns:
            bool: True if the product was available and added, False otherwise.
        """
        logging.info(f'FROM "add_to_cart" ->'
                     f' input: cart_id = {cart_id}, product = {product}')

        idx_product = get_index_of_product(product, self.products)
        
        # Pre-condition: Product type must exist in the marketplace.
        if idx_product == -1:
            logging.info(f'FROM "add_to_cart" ->'
                         f' output: False')
            return False
        
        # Pre-condition: There must be at least one unit of the product available.
        if self.products[idx_product].quantity == 0:
            logging.info(f'FROM "add_to_cart" ->'
                         f' output: False')
            return False

        with self.add_to_cart_lock:
            
            idx_producer = self.products[idx_product].producer_id
            self.producers[idx_producer].nr_items -= 1

            
            self.products[idx_product].quantity -= 1

        
        self.carts[cart_id].products.append(product)

        logging.info(f'FROM "add_to_cart" ->'
                     f' output: True')
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the marketplace inventory.

        Args:
            cart_id (int): The ID of the cart.
            product: The product to remove.
        """
        logging.info(f'FROM "remove_from_cart" ->'
                     f' input: cart_id = {cart_id}, product = {product}')

        
        self.carts[cart_id].products.remove(product)
        with self.remove_from_cart_lock:
            
            idx_product = get_index_of_product(product, self.products)
            self.products[idx_product].quantity += 1

            
            idx_producer = self.products[idx_product].producer_id
            self.producers[idx_producer].nr_items += 1

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart by printing its contents.

        @warning This is not a realistic consumption model. This method only prints
        the items and returns them. It does not remove the cart or the items
        from the system, meaning they are never truly "consumed".

        Args:
            cart_id (int): The ID of the cart being ordered.

        Returns:
            list: The list of products that were in the cart.
        """
        logging.info(f'FROM "place_order" ->'
                     f' input: cart_id = {cart_id}')

        self.pint_lock.acquire()
        for product in self.carts[cart_id].products:
            print(f'{currentThread().name} bought {product}')
        self.pint_lock.release()

        logging.info(f'FROM "place_order" ->'
                     f' input: cart_id = {self.carts[cart_id].products}')
        return self.carts[cart_id].products

class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self):
        """Sets up the test fixture before each test method."""
        self.marketplace = Marketplace(5)

    def test_register_producer(self):
        """Tests that producer registration returns sequential IDs."""
        self.assertEqual(self.marketplace.register_producer(), len(self.marketplace.producers) - 1)
        
        self.assertEqual(len(self.marketplace.producers), 1)

    def test_new_cart(self):
        """Tests that new cart creation returns sequential IDs."""
        self.assertEqual(self.marketplace.new_cart(), len(self.marketplace.carts) - 1)
        
        self.assertEqual(len(self.marketplace.carts), 1)

    def test_publish_success(self):
        """Tests successful product publication."""
        self.marketplace.register_producer()
        product = "product1"
        self.assertEqual(self.marketplace.publish(0, product), True)
        
        self.assertEqual(len(self.marketplace.products), 1)

    def test_publish_fail(self):
        """Tests that publishing fails when the producer's queue is full."""
        self.marketplace.register_producer()
        self.marketplace.producers[0].nr_items = 5
        product = "product1"
        self.assertEqual(self.marketplace.publish(0, product), False)

    def test_add_to_cart_success(self):
        """Tests successful addition of a product to a cart."""
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        product = "product1"
        self.marketplace.publish(0, product)
        self.assertEqual(self.marketplace.add_to_cart(0, product), True)

    def test_add_to_cart_fail_case1(self):
        """Tests that adding a non-existent product to a cart fails."""
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        product = "product1"
        self.marketplace.publish(0, product)
        nonexistent_product = "nonexistent_product"
        self.assertEqual(self.marketplace.add_to_cart(0, nonexistent_product), False)

    def test_add_to_cart_fail_case2(self):
        """Tests that adding a product with zero quantity fails."""
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        product = "product1"
        self.marketplace.publish(0, product)
        self.marketplace.products[0].quantity = 0
        self.assertEqual(self.marketplace.add_to_cart(0, product), False)

    def test_remove_from_cart(self):
        """Tests removing a product from a cart."""
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        product = "product1"
        self.marketplace.publish(0, product)
        self.marketplace.add_to_cart(0, product)
        self.marketplace.remove_from_cart(0, product)
        self.assertEqual(self.marketplace.carts[0].products, [])

    def test_place_order(self):
        """Tests the place_order method."""
        self.marketplace.new_cart()
        products_sample = ["prod1", "prod2", "prod3"]
        self.marketplace.carts[0].products = products_sample
        self.assertEqual(self.marketplace.place_order(0), products_sample)


from threading import Thread
import time

class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list where each element is a tuple of
                             (product, quantity_to_produce, production_time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying a publish
                                         if the producer's queue is full.
            **kwargs: Keyword arguments for the parent Thread class.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        The main execution logic for the producer thread.

        Registers with the marketplace and then enters an infinite loop,
        continuously publishing its assigned products with retries.
        """
        id_producer = self.marketplace.register_producer()
        while True:
            for i in range(len(self.products)):
                product_id = self.products[i][0]
                how_many = self.products[i][1]
                wait_time = self.products[i][2]
                for _ in range(how_many):
                    # Invariant: Keep trying to publish until successful.
                    while True:
                        return_value = self.marketplace.publish(id_producer, product_id)
                        if return_value:
                            time.sleep(wait_time)
                            break
                        time.sleep(self.republish_wait_time)
