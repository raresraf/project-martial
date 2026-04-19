
"""
@file consumer.py
@brief Event-driven multi-threaded Marketplace simulation.

This implementation provides a concurrent framework where producers and consumers 
interact via a central Marketplace buffer. It uses thread-safe mechanisms to 
manage product availability, cart state, and order fulfillment.

Algorithm: Concurrent state management with centralized mutual exclusion.
Domain: Multi-threaded Production Systems.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Independent consumer thread that executes a series of shopping tasks.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        :param carts: A list of shopping task definitions.
        :param marketplace: Reference to the shared transaction hub.
        :param retry_wait_time: Backoff interval for resource contention.
        """
        Thread.__init__(self, **kwargs)
        self.name = kwargs["name"]
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        Main execution loop: Processes multiple carts sequentially.
        """
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()

            # Block Logic: Instruction execution sequence for the current cart.
            for cart_op in cart:
                quantity = cart_op.get("quantity")

                
                if cart_op.get("type") == "add":
                    
                    while quantity > 0:
                        
                        # Logic: Retries addition until space/product availability permits.
                        while not self.marketplace.add_to_cart(cart_id, cart_op.get("product")):
                            time.sleep(self.retry_wait_time)
                        quantity -= 1
                elif cart_op.get("type") == "remove":
                    
                    while quantity > 0:
                        self.marketplace.remove_from_cart(cart_id, cart_op.get("product"))
                        quantity -= 1

            
            # Finalization: Converts cart state into a permanent order record.
            for product in self.marketplace.place_order(cart_id):
                print(f"{self.name} bought {product}")


from threading import Lock
import unittest
import logging
from tema.product import Tea, Coffee

class Marketplace:
    """
    Thread-safe synchronization hub that mediates access to global product listings 
    and private consumer carts.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the state with partitioned locks for producer and consumer logic tiers.
        """
        self.queue_size_per_producer = queue_size_per_producer
        
        self.nb_producers = 0
        
        self.nb_consumers = 0
        
        self.producers = {} # Storage mapping producer IDs to their available products.
        
        self.consumers = {} # Storage mapping consumer IDs to their cart contents.
        
        self.producer_lock = Lock()
        
        self.consumer_lock = Lock()

        # Logging: Initialized for persistent tracing of marketplace events.
        logging.basicConfig(filename="marketplace.log", filemode='w',
                            level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            datefmt='%d/%m/%Y %H:%M:%S')

    def register_producer(self):
        """
        Registers a new producer and initializes its inventory buffer.
        """
        logging.info("producer registered with id %s", self.nb_producers)
        
        self.producers[self.nb_producers] = []
        self.nb_producers += 1

        
        return self.nb_producers - 1

    def publish(self, producer_id, product):
        """
        Allows a producer to add a product to the global inventory.
        Logic: Enforces per-producer queue size constraints to prevent memory exhaustion.
        """
        logging.info("producer %s published product %s", producer_id, product)
        
        if len(self.producers[producer_id]) == self.queue_size_per_producer:
            logging.info("publish returned False")
            return False

        # Critical Section: Thread-safe update of the producer's available stock.
        with self.producer_lock:
            self.producers[producer_id].append(product)
        logging.info("publish returned True")
        return True

    def new_cart(self):
        """
        Allocates a new unique cart identifier for a consumer session.
        """
        logging.info("cart registered with id %s", self.nb_consumers)
        
        self.consumers[self.nb_consumers] = []
        self.nb_consumers += 1

        
        return self.nb_consumers - 1

    def add_to_cart(self, cart_id, product):
        """
        Attempts to move a product from any producer's stock into the target cart.
        Logic: Performs a global search across all producer inventories.
        """
        logging.info("cart %s added to cart %s", cart_id, product)
        
        for producer_id in range(self.nb_producers):
            
            for prd in self.producers[producer_id]:
                
                if prd == product:
                    # Critical Section: Atomic transfer between global stock and private cart.
                    with self.consumer_lock:
                        
                        self.producers[producer_id].remove(product)
                        self.consumers[cart_id].append([product, producer_id])
                    logging.info("add_to_cart returned True")
                    return True
        logging.info("add_to_cart returned False")
        return False


    def remove_from_cart(self, cart_id, product):
        """
        Reverts a product from a cart back to the original producer's stock.
        """
        logging.info("cart %s removed from cart %s", cart_id, product)
        
        for [prd, producer_id] in self.consumers[cart_id]:
            
            if prd == product:
                with self.consumer_lock:
                    # Critical Section: Restoration of inventory state.
                    self.consumers[cart_id].remove([product, producer_id])
                    self.producers[producer_id].append(product)
                break

    def place_order(self, cart_id):
        """
        Finalizes the order by listing all acquired items and clearing the cart state.
        """
        
        products = [product for [product, _] in self.consumers[cart_id]]
        logging.info("cart %s placed order: %s", cart_id, products)

        return products


class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        
        self.marketplace = Marketplace(3)
        self.producers = []
        self.consumers = []
        self.products = []

        
        self.products.append(Tea(name='Linden', price=9, type='Herbal'))
        self.products.append(Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM'))

        
        for _ in range(10):
            self.producers.append(self.marketplace.register_producer())

        
        for _ in range(5):
            self.consumers.append(self.marketplace.new_cart())

    def test_register_producer(self):
        
        
        self.assertEqual(self.marketplace.register_producer(), 10)

    def test_publish(self):
        
        
        for _ in range(3):


            self.assertEqual(self.marketplace.publish(0, self.products[0]), True)
        self.assertEqual(self.marketplace.publish(0, self.products[0]), False)

    def test_new_cart(self):
        
        
        self.assertEqual(self.marketplace.new_cart(), 5)

    def test_add_to_cart1(self):
        
        
        for i in range(2):
            self.marketplace.publish(0, self.products[i])

        self.assertEqual(self.marketplace.add_to_cart(0, self.products[0]), True)


        self.assertEqual(self.marketplace.add_to_cart(0, self.products[1]), True)
        
        self.assertEqual(self.marketplace.add_to_cart(0, self.products[0]), False)

    def test_add_to_cart2(self):
        
        
        for i in range(2):
            for j in range(2):
                self.marketplace.publish(i, self.products[j])

        


        self.assertEqual(self.marketplace.add_to_cart(0, self.products[0]), True)
        self.assertEqual(self.marketplace.add_to_cart(0, self.products[1]), True)
        self.assertEqual(self.marketplace.add_to_cart(0, self.products[0]), True)

    def test_add_to_cart3(self):
        
        
        for i in range(2):
            self.marketplace.publish(0, self.products[i])

        self.assertEqual(self.marketplace.add_to_cart(0, self.products[0]), True)
        self.assertEqual(self.marketplace.add_to_cart(1, self.products[1]), True)
        
        self.assertEqual(self.marketplace.add_to_cart(1, self.products[0]), False)

    def test_remove_from_cart1(self):
        
        
        for i in range(2):
            self.marketplace.publish(0, self.products[i])



        self.marketplace.add_to_cart(0, self.products[0])
        self.marketplace.remove_from_cart(0, self.products[0])
        
        self.assertEqual(self.marketplace.add_to_cart(0, self.products[0]), True)

    def test_remove_from_cart2(self):
        
        
        for i in range(2):
            self.marketplace.publish(0, self.products[i])
            self.marketplace.add_to_cart(0, self.products[i])
            self.marketplace.remove_from_cart(0, self.products[i])
            
            self.assertEqual(self.marketplace.add_to_cart(0, self.products[i]), True)

    def test_place_order(self):
        
        for i in range(2):
            self.marketplace.publish(0, self.products[i])
            self.marketplace.add_to_cart(0, self.products[i])

        
        self.assertEqual(self.marketplace.place_order(0), self.products)


from threading import Thread
import time


class Producer(Thread):
    """
    Supply-side background thread that continuously produces items for the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        :param products: Catalog of product metadata (id, quantity, production_time).
        :param marketplace: Reference to the shared buffer.
        :param republish_wait_time: Throttling delay for full inventory.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        Infinite production lifecycle.
        Logic: Iterates over the assigned product list, producing and publishing 
        each item while respecting production times and marketplace capacity.
        """
        while True:
            for product in self.products:
                
                [product_id, quantity, wait_time] = product
                time.sleep(wait_time)

                
                while quantity > 0:
                    
                    # Logic: Retries publishing until the marketplace buffer has space.
                    while not self.marketplace.publish(self.producer_id, product_id):
                        time.sleep(self.republish_wait_time)
                    quantity -= 1


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Base data class for marketplace items.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Tea specialization with variety metadata.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Coffee specialization with acidity and roast level.
    """
    acidity: str
    roast_level: str
