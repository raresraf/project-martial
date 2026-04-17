"""
@c4294f9e-4ac1-4092-8b99-a524c98d3dd2/catalog.py
@brief Hierarchical marketplace simulation with producer-specific catalogs and state-based inventory reservation.
* Algorithm: Resource state tracking using (Available, Reserved/Frozen) tuples with re-entrant locking for nested operations.
* Functional Utility: Orchestrates a multi-threaded virtual market where inventory is partitioned across producer catalogs and reserved during consumer interactions.
"""

from collections.abc import MutableMapping
from threading import RLock

class Catalog():
    """
    @brief Producer-specific inventory manager that tracks item availability and reservations.
    """
    
    def __init__(self, max_elems):
        """
        @brief Initializes the catalog with a fixed capacity limit.
        """
        self.lock = RLock()
        self.inventory = {} # Intent: Maps product to tuple (available_count, frozen_count).
        self.max_elems = max_elems
        self.size = 0 # Domain: Current total items (available + frozen) in this catalog.

    def add_product(self, product):
        """
        @brief Increases the available count of a product if catalog capacity allows.
        """
        with self.lock:
            if self.size == self.max_elems:
                return False
            try:
                tup = self.inventory[product]
                (count, frozen) = tup
                self.inventory[product] = (count + 1, frozen)
            except KeyError:
                self.inventory[product] = (1, 0)
            self.size += 1
        return True

    def order_product(self, product):
        """
        @brief Permanently removes a frozen item from the catalog after a completed purchase.
        Pre-condition: Product must have been previously reserved (frozen).
        """
        with self.lock:
            (count, frozen) = self.inventory[product]
            self.inventory[product] = (count, frozen - 1)
            self.size -= 1

    def free_product(self, product):
        """
        @brief Restores a previously reserved item back to the available pool.
        """
        with self.lock:
            if product not in self.inventory:
                return False
            (count, frozen) = self.inventory[product]
            self.inventory[product] = (count + 1, frozen - 1)
            return True

    def reserve_product(self, product):
        """
        @brief Transitions an item from 'available' to 'frozen' (reserved) state.
        Algorithm: Inventory reservation to prevent double-selling during cart operations.
        """
        with self.lock:
            if product not in self.inventory:
                # Debug Logic: Records missing inventory states for post-mortem analysis.
                with open("inventory.txt", "w") as f:
                    f.write(str(product) + " not found in " +
                            str(self.inventory))
                return False
            (count, frozen) = self.inventory[product]
            if count == 0:
                return False
            self.inventory[product] = (count - 1, frozen + 1)
        return True

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    @brief Consumer agent that executes multi-step shopping transactions.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes the consumer with its transaction list.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        @brief Main execution loop for shopping activities.
        Algorithm: Iterative processing of multiple carts with operation-level retries.
        """
        customer_id = self.marketplace.new_customers()
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                for _ in range(operation['quantity']):
                    sleep(self.retry_wait_time)
                    if operation['type'] == 'add':
                        # Logic: Attempts to reserve product; retries on failure.
                        finished = self.marketplace.add_to_cart(
                            cart_id, operation['product'])
                        while not finished:
                            # Functional Utility: Throttles attempts to handle stock contention.
                            sleep(self.retry_wait_time)
                            finished = self.marketplace.add_to_cart(
                                cart_id, operation['product'])
                    elif operation['type'] == 'remove':
                        self.marketplace.remove_from_cart(
                            cart_id, operation['product'])

            # Post-condition: Completes the transaction and logs successful purchases.
            prods = self.marketplace.place_order(cart_id)
            for prod in prods:
                print("cons{} bought {}".format(customer_id, str(prod)))

import logging
import logging.handlers
from threading import RLock

class Marketplace:
    """
    @brief Centralized marketplace controller managing multiple producer catalogs and consumer sessions.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes market with various sub-locks to maximize concurrency.
        """
        self.catalogs_lock = RLock()
        self.cartlock = RLock()
        self.customerslock = RLock()
        self.customeridlock = RLock()
        self.loglock = RLock()

        self.producers_catalogs = [] # Intent: Registry of Catalog instances per producer.
        self.carts = []              # Intent: Registry of active consumer sessions.
        self.queue_size_per_producer = queue_size_per_producer
        self.customers_active = 0
        self.customers_total = 0
        
        # Block Logic: Configuration for rotating file logger.
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        handler = logging.handlers.RotatingFileHandler(
            "marketplace.log", backupCount=4, maxBytes=10000000)
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def customers_left(self):
        """
        @brief Checks if any consumer sessions are currently active.
        """
        return self.customers_active > 0

    def new_customers(self):
        """
        @brief Registers a new customer and returns their global sequence ID.
        """
        with self.customerslock:
            self.customers_total += 1
        return self.customers_total

    def register_producer(self):
        """
        @brief Onboards a new producer and creates its dedicated catalog.
        """
        catalog = Catalog(self.queue_size_per_producer)
        with self.catalogs_lock:
            producer_id = len(self.producers_catalogs)
            self.producers_catalogs.append(catalog)

        self.log(f"{producer_id} = register_producer()")
        return producer_id

    def publish(self, producer_id, product):
        """
        @brief Routes a product publication request to the appropriate producer catalog.
        """
        catalog = self.producers_catalogs[producer_id]
        ret = catalog.add_product(product)
        self.log(f"{ret} = publish({producer_id}, {product})")
        return ret

    def new_cart(self):
        """
        @brief Initializes a new shopping cart session.
        Invariant: Increments active customer count to coordinate with producer shutdown logic.
        """
        with self.customerslock:
            self.customers_active += 1
        with self.cartlock:
            cart_id = len(self.carts)
            self.carts.append([])
        self.log(f"{cart_id} = new_cart()")
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Attempts to reserve a product across all known catalogs.
        Algorithm: Distributed inventory search - iterates catalogs until reservation succeeds.
        """
        cart = self.carts[cart_id]
        ret = False
        for catalog in self.producers_catalogs:
            ret = catalog.reserve_product(product)
            if ret:
                # Logic: Stores product and its source catalog for consistent returns/completion.
                cart.append((product, catalog))
                break
        self.log(f"{ret} = add_to_cart({cart_id}, {product})")
        return ret

    def remove_from_cart(self, cart_id, product):
        """
        @brief Identifies and returns a reserved product back to its source catalog.
        """
        cart = self.carts[cart_id]
        for (searched_product, catalog) in cart:
            if product == searched_product:
                # Logic: Atomic state transition in the source catalog.
                catalog.free_product(product)
                cart.remove((searched_product, catalog))
                break
        self.log(f"remove_from_cart({cart_id}, {product})")

    def place_order(self, cart_id):
        """
        @brief Permanently commits all reserved items in a cart and closes the session.
        """
        product_list = []
        cart = self.carts[cart_id]
        for (product, catalog) in cart:
            # Logic: Transitions items from 'frozen' to 'ordered' (removed from catalog).
            catalog.order_product(product)
            product_list.append(product)
        with self.customerslock:
            self.customers_active -= 1
        self.log(f"{product_list} = place_order({cart_id})")
        return product_list

    def log(self, message):
        """
        @brief Stub for audit logging functionality.
        """
        pass

from threading import Thread
from time import sleep

class Producer(Thread):
    """
    @brief Producer agent that generates goods as long as active consumers exist.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes the producer with its product portfolio.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        @brief Main production lifecycle.
        Algorithm: Conditional production based on marketplace consumer activity.
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
                    while not finished:
                        # Logic: Blocks during catalog saturation.
                        sleep(self.republish_wait_time)
                        finished = self.marketplace.publish(producer_id, product)
                    # Domain: Product creation latency.
                    sleep(wait_time)

from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Immutable base schema for marketplace goods.
    """
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Specialized tea product schema.
    """
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Specialized coffee product schema.
    """
    acidity: str
    roast_level: str
