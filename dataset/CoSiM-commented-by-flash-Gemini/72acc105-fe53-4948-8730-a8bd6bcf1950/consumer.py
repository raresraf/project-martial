"""
@72acc105-fe53-4948-8730-a8bd6bcf1950/consumer.py
@brief Concurrent marketplace simulation implementing the Producer-Consumer pattern.

Functional Intent: Provides a thread-safe environment for multiple Producers to 
publish products and Consumers to acquire them via virtual shopping carts. 
Features fine-grained synchronization using multiple mutexes (Locks) for 
distinct shared resources like registration, cart allocation, and inventory access. 
Implements robust retry logic for handling stock depletion or quota limits.

Domain: Production Systems, Concurrency and Synchronization.
"""


from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    @brief Represents a consumer entity that operates in its own execution thread.
    
    Logic: Iteratively creates shopping sessions (carts) and performs batch 
    add/remove operations until all requested quantities are met.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes the consumer with specific shopping tasks.
        @param carts List of carts containing operation sequences (add/remove).
        @param marketplace Reference to the central Marketplace instance.
        @param retry_wait_time Duration to wait when polling for unavailable items.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs
        self.cart_id = 0

    def run(self):
        """
        @brief Core execution loop for the consumer thread.
        
        Algorithm: Sequential cart processing with persistent item polling.
        """
        for current_cart in self.carts:
            # Block Logic: Registration of a new shopping session.
            self.cart_id = self.marketplace.new_cart()
            for operation in current_cart:
                
                if operation["type"] == "add":
                    # Block Logic: Batch acquisition loop.
                    for _ in range(operation["quantity"]):
                        res = self.marketplace.add_to_cart(self.cart_id, operation["product"])
                        
                        # Invariant: Continues polling until the marketplace grants the reservation.
                        while not res:
                            sleep(self.retry_wait_time)
                            res = self.marketplace.add_to_cart(self.cart_id, operation["product"])
                else:
                    
                    if operation["type"] == "remove":
                        # Block Logic: Batch restoration pass.
                        for _ in range(operation["quantity"]):
                            self.marketplace.remove_from_cart(self.cart_id, operation["product"])
            
            # Functional Intent: Commits the transaction and logs the purchase results.
            products = self.marketplace.place_order(self.cart_id)
            for product in products:
                print(self.kwargs["name"]+ " bought " + str(product))


from logging.handlers import RotatingFileHandler
import random
from threading import Lock
import time
import unittest
import logging

class Marketplace:
    """
    @brief Central coordinator for thread-safe item publishing and transactions.
    
    Functional Utility: Manages global inventory, producer registration, and 
    consumer carts. Utilizes multiple locks to prevent race conditions during 
    concurrent access to distinct shared data structures.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the marketplace with capacity limits and diagnostic logging.
        """
        self.size = queue_size_per_producer
        self.products_producer = {} # Logic: Maps producer IDs to their active published items.
        self.products = [] # Logic: Global pool of items currently available for sale.
        self.carts = {} # Logic: Maps cart IDs to their respective reserved item lists.
        
        # Synchronization: Discrete locks for granular concurrency control.
        self.lock_register = Lock()
        self.lock_cart = Lock()
        self.lock_products = Lock()
        self.lock_place_order = Lock()

        # Configuration: Rotating file logger for ISO-formatted activity logs.
        self.logger = logging.getLogger('marketplace.logger')
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('marketplace.log', maxBytes=10000, backupCount=10)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)8s: %(message)s')
        handler.setFormatter(formatter)
        logging.Formatter.converter = time.gmtime
        self.logger.addHandler(handler)
        self.logger.info("MARKETPLACE CONSTRUCTOR")
        self.logger.info("queue_size =%s", str(queue_size_per_producer))

    def register_producer(self):
        """
        @brief Onboards a new producer and initializes its inventory context.
        """
        self.logger.info("------------------")
        self.logger.info("REGISTER PRODUCER")

        # Synchronization: Critical section for protected ID generation.
        with self.lock_register:
            id_producer = len(self.products_producer.keys())
            self.products_producer[id_producer] = []
        self.logger.info("id_producer =%s", str(id_producer))
        return id_producer


    def publish(self, producer_id, product):
        """
        @brief Adds a product to the shop if the producer's quota allows.
        """
        self.logger.info("------------------")
        self.logger.info("PUBLISH")
        self.logger.info("producer_id = %s product %s", str(producer_id), str(product))

        # Block Logic: Quota verification.
        # Invariant: Producer cannot have more than 'size' items published simultaneously.
        if len(self.products_producer[producer_id]) < self.size:
            self.products_producer[producer_id].append(product)
            self.products.append(product)

            self.logger.info("PRODUSE IN MAGAZIN")
            self.logger.info("PUBLISH RES = TRUE")

            return True
        self.logger.info("PUBLISH RES = FALSE")
        return False

    def new_cart(self):
        """
        @brief Spawns a new shopping cart session for a consumer.
        """
        self.logger.info("------------------")
        self.logger.info("NEW_CART")
        
        # Synchronization: Mutex for thread-safe session allocation.
        with self.lock_cart:
            id_cart = len(self.carts.keys())
            self.carts[id_cart] = []
            self.logger.info("NEW_CART_ID = %s", str(id_cart))
            return id_cart

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically transfers an item from the global inventory to a specific cart.
        """
        self.logger.info("------------------")
        self.logger.info("ADD_TO_CART ARGS")
        self.logger.info("id = %s  product = %s", str(cart_id), str(product))

        # Synchronization: Critical section for global inventory mutation.
        with self.lock_products:
            if product in self.products:
                self.carts[cart_id].append(product)
                # Invariant: Item must be removed from global pool to prevent over-selling.
                self.products.remove(product)

                self.logger.info("ADD_TO_CART RES = TRUE")
                self.logger.info("CURRENT CART = %s", str(self.carts[cart_id]))

                return True
        self.logger.info("ADD_TO_CART RES = FALSE")
        return False


    def remove_from_cart(self, cart_id, product):
        """
        @brief Returns a reserved item from a cart back to the global marketplace.
        """
        self.logger.info("------------------")
        self.logger.info("REMOVE_FROM_CART ARGS")
        self.logger.info("id = %s  product = %s", str(cart_id), str(product))

        self.carts[cart_id].remove(product)
        self.logger.info("CURRENT CART = %s", str(self.carts[cart_id]))
        self.products.append(product)

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction, updating producer active counts and closing session.
        """
        self.logger.info("------------------")
        self.logger.info("PLACE_ORDER ARGS")
        self.logger.info("id = %s", str(cart_id))

        # Synchronization: Protects cross-producer inventory reconciliation during checkout.
        with self.lock_place_order:
            for product in self.carts[cart_id]:
                for id_producer in self.products_producer:
                    
                    if product in self.products_producer[id_producer]:
                        # Logic: Successfully identified the origin producer to release their quota slot.
                        self.products_producer[id_producer].remove(product)
                        break

            self.logger.info("PLACE_ORDER RES")
            self.logger.info(self.carts[cart_id])

            return self.carts[cart_id]

class TestMarketplace(unittest.TestCase):
    """
    @brief Unit tests for validating Marketplace synchronization and data integrity.
    """
    def setUp(self):
        self.marketplace = Marketplace(10)

    def test_register_producer(self):
        self.marketplace.products_producer[0] = []
        self.marketplace.products_producer[1] = []
        self.marketplace.products_producer[2] = []
        self.marketplace.products_producer[3] = []
        self.marketplace.products_producer[4] = []
        self.assertEqual(self.marketplace.register_producer(), 5, "Incorrect value, must be 6")

    def test_publish1(self):
        test_list = []
        self.marketplace.register_producer()
        for i in range(10):
            if i < 10 / 2:
                product = {"product_type": "Coffee"}
                name = "Columbia"
                product["name"] = name
                product["acidity"] = round(random.uniform(3, 4.8), 2)
                product["roast_level"] = "LIGHT"
            else:
                product = {"product_type": "Tea"}
                tea = "English Breakfast"
                product["name"] = tea
                product["type"] = "Black"

            product["price"] = random.randint(1, 10)
            test_list.append(product)
            self.marketplace.publish(0, product)
        self.assertListEqual(self.marketplace.products, test_list, "Publish FAILED")

    def test_publish2(self):
        test_list = []
        self.marketplace.register_producer()
        for i in range(20):
            if i < 10 / 2:
                product = {"product_type": "Coffee"}
                name = "Columbia"
                product["name"] = name
                product["acidity"] = round(random.uniform(3, 4.8), 2)
                product["roast_level"] = "LIGHT"
            else:
                product = {"product_type": "Tea"}
                tea = "English Breakfast"
                product["name"] = tea
                product["type"] = "Black"

            product["price"] = random.randint(1, 10)
            if i < 10:
                test_list.append(product)
            self.marketplace.publish(0, product)
        self.assertListEqual(self.marketplace.products, test_list, "publish FAILED")

    def test_new_cart(self):
        self.marketplace.carts[0] = []
        self.marketplace.carts[1] = []
        self.marketplace.carts[2] = []
        self.marketplace.carts[3] = []
        self.assertEqual(self.marketplace.new_cart(), 4, "new_cart FAILED")

    def test_add_to_cart1(self):
        cart_id = self.marketplace.new_cart()
        self.marketplace.register_producer()
        test_list = []
        for i in range(10):
            if i < 10 / 2:
                product = {"product_type": "Coffee"}
                name = "Columbia"
                product["name"] = name
                product["acidity"] = round(random.uniform(3, 4.8), 2)
                product["roast_level"] = "LIGHT"
            else:
                product = {"product_type": "Tea"}
                tea = "English Breakfast"
                product["name"] = tea
                product["type"] = "Black"

            product["price"] = random.randint(1, 10)
            test_list.append(product)
            self.marketplace.publish(0, product)

        for product in test_list:
            self.marketplace.add_to_cart(cart_id, product)
        self.assertListEqual(self.marketplace.carts[cart_id], test_list, "add_to_cart FAILED")

    def test_add_to_cart2(self):
        cart_id = self.marketplace.new_cart()
        self.marketplace.register_producer()
        for i in range(10):
            if i < 10 / 2:
                product = {"product_type": "Coffee"}
                name = "Columbia"
                product["name"] = name
                product["acidity"] = round(random.uniform(3, 4.8), 2)
                product["roast_level"] = "LIGHT"
            else:
                product = {"product_type": "Tea"}
                tea = "English Breakfast"
                product["name"] = tea
                product["type"] = "Black"

            product["price"] = random.randint(1, 10)
            self.marketplace.publish(0, product)

        product = {"product_type": "Fruit"}
        name = "Apple"
        product["name"] = name
        product["acidity"] = 0
        product["roast_level"] = "LIGHT"
        product["price"] = random.randint(1, 10)
        self.marketplace.add_to_cart(cart_id, product)
        self.assertListEqual(self.marketplace.carts[cart_id], [], "add_to_cart FAILED")

    def test_remove_from_cart(self):
        cart_id = self.marketplace.new_cart()
        self.marketplace.register_producer()
        test_list = []
        for i in range(10):
            if i < 10 / 2:
                product = {"product_type": "Coffee"}
                name = "Columbia"
                product["name"] = name
                product["acidity"] = round(random.uniform(3, 4.8), 2)
                product["roast_level"] = "LIGHT"
            else:
                product = {"product_type": "Tea"}
                tea = "English Breakfast"
                product["name"] = tea
                product["type"] = "Black"

            product["price"] = random.randint(1, 10)
            test_list.append(product)
            self.marketplace.publish(0, product)

        for product in test_list:
            self.marketplace.add_to_cart(cart_id, product)

        for i in range(4):
            test_list.remove(test_list[i])
            product = self.marketplace.carts[cart_id][i]
            self.marketplace.remove_from_cart(cart_id, product)

        self.assertListEqual(self.marketplace.carts[cart_id], test_list, "remove_from_cart FAILED")

    def test_plac_order(self):
        cart_id = self.marketplace.new_cart()
        self.marketplace.register_producer()
        test_list = []
        for i in range(10):
            if i < 10 / 2:
                product = {"product_type": "Coffee"}
                name = "Columbia"
                product["name"] = name
                product["acidity"] = round(random.uniform(3, 4.8), 2)
                product["roast_level"] = "LIGHT"
            else:
                product = {"product_type": "Tea"}
                tea = "English Breakfast"
                product["name"] = tea
                product["type"] = "Black"

            product["price"] = random.randint(1, 10)
            test_list.append(product)
            self.marketplace.publish(0, product)

        for product in test_list:
            self.marketplace.add_to_cart(cart_id, product)

        for i in range(3):
            test_list.remove(test_list[i])
            product = self.marketplace.carts[cart_id][i]
            self.marketplace.remove_from_cart(cart_id, product)

        for i in range(2):
            test_list.append(self.marketplace.products[i])
            self.marketplace.add_to_cart(cart_id, self.marketplace.products[i])

        products = self.marketplace.place_order(cart_id)
        self.assertListEqual(products, test_list, "place_order FAILED")

if __name__ == '__main__':
    unittest.main()


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    @brief Represents a producer entity that generates and publishes products.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes the producer with its catalog and production schedule.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.id_producer = self.marketplace.register_producer()

    def run(self):
        """
        @brief Core production cycle.
        
        Logic: Continuously iterates through the product list, publishing items 
        according to specified quantities and simulating manufacturing latency.
        """
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    res = self.marketplace.publish(self.id_producer, product[0])
                    if res:
                        # Optimization: Simulation of device manufacturing time.
                        sleep(product[2])
                    else:
                        # Block Logic: Persistent publishing retry.
                        # Invariant: Re-polls until the marketplace accepts the item.
                        while not res:
                            sleep(self.republish_wait_time)
                            res = self.marketplace.publish(self.id_producer, product[0])


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base immutable representation of a market item.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Specialized product type for tea.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Specialized product type for coffee.
    """
    acidity: str
    roast_level: str
