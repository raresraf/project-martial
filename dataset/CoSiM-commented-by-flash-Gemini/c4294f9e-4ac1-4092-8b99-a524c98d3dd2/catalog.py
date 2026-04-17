"""
@c4294f9e-4ac1-4092-8b99-a524c98d3dd2/catalog.py
@brief Hierarchical electronic marketplace with per-producer inventory catalogs.
This module implements a structured trading environment where each producer manages 
their own 'Catalog' of items. The system tracks item lifecycle states (Available, 
Reserved/Frozen, and Purchased) to ensure transactional integrity across concurrent 
sessions managed by a central Marketplace coordinator.

Domain: Inventory Management, Concurrent State Transitions, Producer-Consumer dynamics.
"""

from collections.abc import MutableMapping
from threading import RLock


class Catalog():
    """
    Inventory manager for a single producer.
    Functional Utility: Provides atomic operations for item state transitions 
    between available stock and reserved cart items.
    """
    
    def __init__(self, max_elems):
        """
        Initializes the catalog.
        @param max_elems: Maximum total storage capacity for this producer.
        """
        self.lock = RLock()
        # Dictionary mapping products to (available_count, frozen_count).
        self.inventory = {}
        self.max_elems = max_elems
        self.size = 0

    def add_product(self, product):
        """
        Increments the available stock of a product.
        Logic: verifies capacity before updating the inventory dictionary.
        @return: True if added, False if capacity limit is reached.
        """
        with self.lock:
            if self.size == self.max_elems:
                return False
            try:
                tup = self.inventory[product]
                (count, frozen) = tup
                self.inventory[product] = (count + 1, frozen)
            except KeyError:
                # Initialization of a new product entry.
                self.inventory[product] = (1, 0)
            self.size += 1
        return True

    def order_product(self, product):
        """
        Finalizes the purchase of a product.
        Precondition: The product must have been previously reserved (frozen).
        Logic: Removes the item from the 'frozen' state and decrements total size.
        """
        with self.lock:
            (count, frozen) = self.inventory[product]
            self.inventory[product] = (count, frozen - 1)
            self.size -= 1

    def free_product(self, product):
        """
        Restores a reserved product back to available stock.
        Logic: Decrements the 'frozen' count and increments the 'available' count.
        """
        with self.lock:
            if product not in self.inventory:
                return False
            (count, frozen) = self.inventory[product]
            self.inventory[product] = (count + 1, frozen - 1)
            return True

    def reserve_product(self, product):
        """
        Claims a product for a consumer cart, making it unavailable to others.
        Logic: Transfers one unit from 'available' to 'frozen' state.
        @return: True if reserved successfully, False if out of stock.
        """
        with self.lock:
            if product not in self.inventory:
                return False
            (count, frozen) = self.inventory[product]
            if count == 0:
                return False
            self.inventory[product] = (count - 1, frozen + 1)
        return True


from threading import Thread
from time import sleep
from tema.marketplace import Marketplace


class Consumer(Thread):
    """
    Simulation thread representing a customer.
    Functional Utility: Orchestrates multi-cart shopping sessions with 
    synchronous delays and retry logic.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the consumer thread.
        @param carts: List of carts, each containing a sequence of operations.
        @param marketplace: Central trading hub.
        @param retry_wait_time: Interval for polling the marketplace.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        Main shopper execution loop.
        Logic: Registers as a new customer and processes each cart sequentially.
        """
        customer_id = self.marketplace.new_customers()
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                for _ in range(operation['quantity']):
                    # Simulate human delay.
                    sleep(self.retry_wait_time)
                    if operation['type'] == 'add':
                        finished = self.marketplace.add_to_cart(
                            cart_id, operation['product'])
                        # Block until the desired item becomes available.
                        while not finished:
                            sleep(self.retry_wait_time)
                            finished = self.marketplace.add_to_cart(
                                cart_id, operation['product'])
                    elif operation['type'] == 'remove':
                        self.marketplace.remove_from_cart(
                            cart_id, operation['product'])

            # Commits the session and prints the manifest.
            prods = self.marketplace.place_order(cart_id)
            for prod in prods:
                print("cons{} bought {}".format(customer_id, str(prod)))

import logging
import logging.handlers
from threading import RLock
from tema.catalog import Catalog


class Marketplace:
    """
    Central session coordinator and inventory aggregator.
    Functional Utility: Manages producer catalogs and tracks active consumer sessions 
    to provide a unified view of the marketplace state.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.
        """
        # Specialized re-entrant locks for different state categories.
        self.catalogs_lock = RLock()
        self.cartlock = RLock()
        self.customerslock = RLock()
        self.customeridlock = RLock()
        self.loglock = RLock()

        self.producers_catalogs = []
        self.carts = []
        self.queue_size_per_producer = queue_size_per_producer
        self.customers_active = 0
        self.customers_total = 0
        
        # System Logging: configured for rotating file audit.
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        handler = logging.handlers.RotatingFileHandler(
            "marketplace.log", backupCount=4, maxBytes=10000000)
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def customers_left(self):
        """Checks if there are still active shopping sessions in the system."""
        return self.customers_active > 0

    def new_customers(self):
        """Registers a new unique customer and returns their global ID."""
        with self.customerslock:
            self.customers_total += 1
        return self.customers_total

    def register_producer(self):
        """Allocates a new Catalog for a producer and registers it in the hub."""
        catalog = Catalog(self.queue_size_per_producer)
        with self.catalogs_lock:
            producer_id = len(self.producers_catalogs)
            self.producers_catalogs.append(catalog)
        return producer_id

    def publish(self, producer_id, product):
        """Delegates item publication to a specific producer's catalog."""
        catalog = self.producers_catalogs[producer_id]
        return catalog.add_product(product)

    def new_cart(self):
        """Initializes a new shopping session for a customer."""
        with self.customerslock:
            # Lifecycle Tracking: increments active customer count.
            self.customers_active += 1
        with self.cartlock:
            cart_id = len(self.carts)
            self.carts.append([])
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Global search for a product across all available producer catalogs.
        Logic: Performs an atomic reservation on the first catalog that has stock.
        """
        cart = self.carts[cart_id]
        ret = False
        for catalog in self.producers_catalogs:
            ret = catalog.reserve_product(product)
            if ret:
                # Association: stores both the product and its origin catalog.
                cart.append((product, catalog))
                break
        return ret

    def remove_from_cart(self, cart_id, product):
        """
        Restores a product from a cart back to its originating catalog.
        """
        cart = self.carts[cart_id]
        for (searched_product, catalog) in cart:
            if product == searched_product:
                catalog.free_product(product)
                cart.remove((searched_product, catalog))
                break

    def place_order(self, cart_id):
        """
        Finalizes the transaction by converting reservations into permanent orders.
        """
        product_list = []
        cart = self.carts[cart_id]
        for (product, catalog) in cart:
            catalog.order_product(product)
            product_list.append(product)
        with self.customerslock:
            # Lifecycle Tracking: decrements active customer count.
            self.customers_active -= 1
        return product_list


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Manufacturing entity with an activity-based lifecycle.
    Functional Utility: Continuously supplies goods as long as active customers 
    are detected in the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        Main production loop.
        Logic: Monitors marketplace activity and manufactures goods until 
        demand (customers) ceases.
        """
        sleep(self.republish_wait_time)
        producer_id = self.marketplace.register_producer()
        while self.marketplace.customers_left():
            for bundle in self.products:
                product = bundle[0]
                quantity = bundle[1]
                wait_time = bundle[2]
                for _ in range(quantity):
                    finished = self.marketplace.publish(producer_id, product)
                    # Block Logic: Handle marketplace congestion.
                    while not finished:
                        sleep(self.republish_wait_time)
                        finished = self.marketplace.publish(producer_id, product)
                    # Simulation of manufacturing time.
                    sleep(wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """Core data model for goods."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """Product specialization."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """Product specialization."""
    acidity: str
    roast_level: str
