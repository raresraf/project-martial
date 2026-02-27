from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    Represents a consumer thread that executes a list of shopping operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer.

        Args:
            carts (list): A list of shopping sessions to be performed.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying a failed operation.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        Main execution loop. For each shopping session, creates a new cart,
        performs all operations, and places the order.
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
        Adds a given quantity of a product to the cart, retrying on failure.
        This is a blocking operation that will not return until successful.
        """
        for _ in range(quantity):
            while True:
                added = self.marketplace.add_to_cart(cart_id, product_id)
                if added:
                    break
                sleep(self.retry_wait_time)

    def remove_cart(self, cart_id, product_id, quantity):
        """
        Removes a given quantity of a product from the cart, retrying on failure.
        This is a blocking operation.
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
# The test cases depend on an external 'product' module.
sys.path.insert(1, './tema')
import product as produs

class Marketplace:
    """
    A marketplace simulation.

    WARNING: This implementation is NOT thread-safe. While it has a mutex,
    it is not used to protect most of the critical methods that modify shared
    state (publish, add_to_cart, remove_from_cart), which will lead to race
    conditions under concurrency.
    """
    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace."""
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0
        self.cart_id = 0
        self.queues = [] # List of lists; queues[id] = [product,...]
        self.carts = [] # List of lists; carts[id] = [product,...]
        self.mutex = Lock()
        # Maps a product back to the ID of the producer who owns it.
        self.products_dict = {}

    def register_producer(self):
        """Atomically registers a new producer and returns its ID as a string."""
        with self.mutex:
            producer_id = self.producer_id
            self.producer_id += 1
            self.queues.append([])
        return str(producer_id)

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace.

        WARNING: This method is NOT thread-safe as it does not use a lock when
        modifying the shared `queues` and `products_dict` state.
        """
        index_prod = int(producer_id)
        if len(self.queues[index_prod]) == self.queue_size_per_producer:
            return False
        self.queues[index_prod].append(product)
        self.products_dict[product] = index_prod
        return True

    def new_cart(self):
        """Atomically creates a new cart and returns its ID."""
        with self.mutex:
            cart_id = self.cart_id
            self.cart_id += 1
        self.carts.append([])
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart.

        WARNING: This method is NOT thread-safe. It iterates and modifies shared
        lists (`queues`, `carts`) without a lock.
        """
        prod_in_queue = False
        # Inefficiently scans all producer queues for the product.
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
        Removes a product from a cart and returns it to the producer's queue.

        WARNING: This method is NOT thread-safe.
        It also has a potential logic flaw where an item cannot be returned if
        the producer's queue is full, which could lead to lost items.
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
        """Finalizes an order, returning the products and clearing the cart."""
        cart_product_list = self.carts[cart_id]
        self.carts[cart_id] = []
        return cart_product_list

class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    def setUp(self):
        self.marketplace = Marketplace(4)

    def test_register_producer(self):
        self.assertEqual(self.marketplace.register_producer(), str(0))
        # ... more test assertions ...

    def test_publish(self):
        self.marketplace.register_producer()
        # ... more test assertions ...

    def test_new_cart(self):
        self.assertEqual(self.marketplace.new_cart(), 0)
        # ... more test assertions ...

    def test_add_to_cart(self):
        self.marketplace.register_producer()
        # ... more test assertions ...

    def test_remove_from_cart(self):
        self.marketplace.register_producer()
        # ... more test assertions ...

    def test_place_order(self):
        self.marketplace.register_producer()
        # ... more test assertions ...


from threading import Thread
from time import sleep

class Producer(Thread):
    """A producer thread that continuously publishes products."""
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """Main loop: continuously produces and publishes items."""
        while True:
            for product in self.products:
                quantity = product[1]
                for _ in range(0, quantity):
                    self.publish_product(product[0], product[2])

    def publish_product(self, product, production_time):
        """Publishes a single product, retrying with a wait on failure."""
        while True:
            published = self.marketplace.publish(self.producer_id, product)
            if published:
                sleep(production_time)
                break
            sleep(self.republish_wait_time)
