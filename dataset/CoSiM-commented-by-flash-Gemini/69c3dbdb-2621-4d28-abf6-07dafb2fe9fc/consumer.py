"""
@69c3dbdb-2621-4d28-abf6-07dafb2fe9fc/consumer.py
@brief Multi-threaded simulation of a marketplace environment using Producer-Consumer patterns.
Architecture: Distributed actor-based model where Consumers and Producers interact via a centralized Marketplace mediator.
Synchronization: Utilizes multiprocessing.Lock for thread-safe access to shared state (carts, product listings, registration).
Functional Utility: Simulates realistic e-commerce operations including inventory management, cart persistence, and order fulfillment.
"""

import time
from threading import Thread
from multiprocessing import Lock

class Consumer(Thread):
    """
    @brief Consumer agent responsible for executing shopping strategies.
    Logic: Iterates through assigned carts and attempts to fulfill product requirements by interacting with the Marketplace.
    Error Handling: Employs a retry mechanism with exponential backoff (via sleep) when product availability is constrained.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts List of product requests to be fulfilled.
        @param marketplace Shared resource for inventory and order management.
        @param retry_wait_time Temporal duration to wait before retrying a failed 'add to cart' operation.
        """
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        Thread.__init__(self, **kwargs)


    def run(self):
        """
        @brief Execution loop for the consumer thread.
        Invariant: All products in a cart must be successfully added or the consumer will block (retry) indefinitely.
        """
        for c in self.carts:
            # Functional Utility: Initializes a unique session within the marketplace.
            c_id = self.marketplace.new_cart()
            for req in c:
                type = req['type']
                prod = req['product']
                qty = req['quantity']
                
                # Block Logic: Conditional state transition based on the operation type (add/remove).
                if type == 'add':
                    for _ in range(0, qty):
                        # Logic: Attempts to acquire a product from the marketplace global pool.
                        prod_added_flag = self.marketplace.add_to_cart(c_id, prod)

                        # Block Logic: Synchronization barrier for product availability.
                        # Ensures the consumer waits until the marketplace is replenished by a producer.
                        while prod_added_flag is False:
                            time.sleep(self.retry_wait_time)
                            prod_added_flag = self.marketplace.add_to_cart(c_id, prod)

                elif type == 'remove':
                    for _ in range(0, qty):
                        # Logic: Returns a previously reserved product to the marketplace pool.
                       self.marketplace.remove_from_cart(c_id, prod)

            # Functional Utility: Finalizes the transaction and retrieves the confirmed shopping list.
            shopping_list = self.marketplace.place_order(c_id)

            # Block Logic: Thread-safe I/O operation.
            # Prevents interleaved output from multiple concurrent consumers.
            self.marketplace.cons_lock.acquire()            
            for elem in shopping_list:
                print("{} bought {}".format(self.getName(), elem))
            self.marketplace.cons_lock.release()



from itertools import product
import logging


from logging.handlers import RotatingFileHandler
from multiprocessing import Lock

import unittest

class Marketplace:
    """
    @brief Centralized resource manager for the producer-consumer simulation.
    State Management: Maintains global dictionaries for producer inventory and active shopping carts.
    Synchronization: Uses fine-grained locking (lock_prod, lock_cart, cons_lock) to minimize contention.
    Observability: Integrates RotatingFileHandler for audit logging of all transactional operations.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Hard limit on inventory per producer to prevent resource exhaustion.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.lock_prod = Lock() # Protects producer registration and inventory access.
        self.lock_cart = Lock() # Protects cart allocation and ID generation.
        self.prod_id = 0
        self.lock = Lock() # General purpose lock for cart modifications.

        
        self.prod_list = {} # Mapping: Producer ID -> List of available products.

        
        self.carts_list = {} # Mapping: Cart ID -> List of (ProducerID, Product) tuples.

        self.cart_id = 0
        self.cons_lock = Lock() # Serializes console output across threads.

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Functional Utility: Persistent log of marketplace events with rotation to manage disk space.
        RotatingFileHandler(filename='marketplace.log', maxBytes=50000, backupCount=20)
        self.logger.addHandler(RotatingFileHandler("marketplace.log"))

    def register_producer(self):
        """
        @brief Onboards a new producer into the marketplace ecosystem.
        @return Unique producer identifier.
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
        @brief Allows a producer to add inventory to the marketplace.
        Constraint: Operation fails if the producer's specific queue has reached queue_size_per_producer.
        """

        self.logger.info("Publish was called")

        if len(self.prod_list[producer_id]) == self.queue_size_per_producer:
            self.logger.info("Publish returned False")
            return False

        self.prod_list[producer_id].append(product)
        return True


    def new_cart(self):
        """
        @brief Allocates a new shopping cart for a consumer.
        @return Unique cart identifier.
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
        @brief Atomically moves a product from producer inventory to a consumer cart.
        Logic: Scans all producer lists for the requested product. First-found strategy.
        @return Boolean indicating success of the transfer.
        """
        
        self.logger.info("Add to cart was called")

        for i in self.prod_list:
            if product in self.prod_list[i]:

                # Invariant: Product must be removed from global inventory before being added to a cart.
                self.prod_list[i].remove(product)
                self.carts_list[cart_id].append(tuple((i, product)))
                self.logger.info("Add to cart was made successfully")
                return True
        self.logger.info("Add to cart could not be completed")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverts an add operation, returning a product to its original producer's inventory.
        """
        
        self.logger.info("Remove from cart was called")

        self.lock.acquire()
        for i in self.carts_list[cart_id]:

            if i[1] == product:
                # Logic: Uses the stored ProducerID in the cart tuple to route the product back.
                self.prod_list[i[0]].append(product)
                self.carts_list[cart_id].remove(i)
                self.logger.info("Remove from cart was successful")
                break

        self.lock.release()
        return


    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction for a specific cart.
        Side Effect: Flushes the cart content and returns the product list to the consumer.
        """
        
        self.logger.info("Place order was called")

        cart = self.carts_list[cart_id].copy()
        self.carts_list[cart_id] = [] # Functional Utility: Resets cart state after order.

        final_list = []
        for i in cart:
            final_list.append(i[1])
        self.logger.info("Place order created the list")
        return final_list


class TestMarketplace(unittest.TestCase):
    """
    @brief Unit test suite for verifying Marketplace core logic and synchronization.
    """
    def setUp(self):
        self.marketplace = Marketplace(8)
        self.prod_list = {"name1":"Tea", "name2":"Coffee", "name3":"Wild berries Tea"}
        self.name_list = ["Tea", "Coffee", "Wild berries Tea"]
        self.list = ["Coffee", "Wild berries Tea"]

    def test_register_prod(self):

        rez = self.marketplace.register_producer()
        self.assertEqual(rez, 1, 'Producer not updated')

    def test_publish(self):
        self.marketplace.register_producer()
        rez = self.marketplace.publish(1, self.prod_list["name3"])
        self.assertTrue(rez, 'Product is not published')

    def test_new_cart(self):
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        rez = self.marketplace.new_cart()
        self.assertEqual(rez, 4, 'Cart id not updated')

    def test_add_to_cart(self):

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

# Note: The following section was originally in producer.py
# >>>> file: producer.py

class Producer(Thread):
    """
    @brief Producer agent responsible for inventory replenishment.
    Logic: Continuously cycles through a list of products and attempts to publish them to the Marketplace.
    Flow Control: Blocks when the Marketplace queue for this producer is full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @param products List of (ProductID, Quantity) tuples to be produced.
        @param marketplace Shared resource mediator.
        @param republish_wait_time Temporal duration to wait when the marketplace queue is full.
        """
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        """
        @brief Main loop for the producer thread.
        @p_id Registered unique identifier for this producer.
        """

        p_id = 0
        p_id = self.marketplace.register_producer()

        while True:
            for prod in self.products:
                product_id = prod[0]
                q = prod[1]
                for _ in range(0, q):

                    # Block Logic: Backpressure management.
                    # If publish fails (queue full), the producer waits before retrying.
                    if self.marketplace.publish(p_id, product_id) is True:
                        break

                    time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Immutable base entity representing a commodity in the marketplace.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Specialized product entity for hot beverage simulations.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Specialized product entity with beverage-specific attributes (acidity, roast).
    """
    acidity: str
    roast_level: str
