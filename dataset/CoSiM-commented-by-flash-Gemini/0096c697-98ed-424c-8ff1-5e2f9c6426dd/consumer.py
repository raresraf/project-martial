


"""
This module simulates a multi-threaded marketplace system.
It defines `Consumer` threads that perform shopping actions, a `Marketplace`
class that manages products from producers and consumer carts with thread-safety,
and `Producer` threads that continuously publish products.
Unit tests for the `Marketplace` functionality are also included.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the Marketplace.
    Each consumer can manage multiple shopping carts, perform add/remove
    operations, and place orders. Operations involving adding items
    include retry logic if the item is not immediately available.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        :param carts: A list of cart action sequences. Each sequence is a list
                      of dictionaries, where each dictionary defines an "add"
                      or "remove" operation for a product and quantity.
        :param marketplace: The Marketplace instance with which the consumer interacts.
        :param retry_wait_time: The time in seconds to wait before retrying
                                an 'add to cart' operation if it fails.
        :param kwargs: Arbitrary keyword arguments passed to the Thread constructor,
                       e.g., 'name' for thread identification.
        """
        super().__init__(kwargs=kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs
        self._id = 0 # Stores the ID of the current cart being managed.

    def run(self):
        """
        The main execution method for the Consumer thread.
        It iterates through the predefined cart action sequences. For each sequence,
        it creates a new cart, performs all specified add/remove operations,
        and then places the order, printing the items bought.
        """
        # Iterate through each sequence of cart actions.
        for act_cart in self.carts:
            # Create a new cart for the current sequence of actions.
            self._id = self.marketplace.new_cart()

            items_count = 0 # Tracks the number of items currently in the cart.

            # Process each individual action (add or remove) within the current cart sequence.
            for act_op in act_cart:
                # Handle "add" operations.
                if act_op["type"] == "add":
                    count = 0
                    # Attempt to add the desired quantity of the product.
                    while count < act_op["quantity"]:
                        # If adding to cart is successful.
                        if self.marketplace.add_to_cart(self._id, act_op["product"]):
                            count += 1
                            items_count += 1
                        else:
                            # If adding fails (product not available), wait and retry.
                            sleep(self.retry_wait_time)

                # Handle "remove" operations.
                elif act_op["type"] == "remove":
                    count = 0
                    # Perform the remove operation for the desired quantity.
                    while count < act_op["quantity"]:
                        self.marketplace.remove_from_cart(self._id, act_op["product"])
                        count += 1
                        items_count -= 1

            # Place the final order for the current cart.
            items = self.marketplace.place_order(self._id)

            # Print the list of items that were successfully bought by this consumer.
            for item in items:
                print(self.kwargs["name"] + " bought " + str(item))


from threading import Lock
import time
import unittest
import logging
import logging.handlers

class TestMarketplace(unittest.TestCase):
    """
    Unit test class for the Marketplace functionality.
    It verifies the correctness and thread-safety of various marketplace
    operations such as producer registration, product publishing, cart
    management, and order placement.
    """
    def setUp(self):
        """
        Sets up the test environment by initializing a new Marketplace instance
        before each test method is executed.
        """
        self.marketplace = Marketplace(15)

    def test_register_producer(self):
        """
        Tests the `register_producer` method to ensure it assigns unique
        and sequential producer IDs.
        """
        for i in range(0, 100):
            act_id = self.marketplace.register_producer()
            self.assertEqual(i, act_id)

    def test_publish(self):
        """
        Tests the `publish` method, verifying that products are correctly
        added to a producer's goods list within the marketplace.
        """
        from product import Product # Assuming Product class is defined in product.py or is available

        prod_id = self.marketplace.register_producer()
        new_prod = Product("Ceai verde", 10)

        return_val = self.marketplace.publish(prod_id, new_prod)

        if return_val:
            # Asserts that the published product is the last item in the producer's goods.
            self.assertEqual(new_prod, self.marketplace.goods[prod_id][-1][0])

    def test_new_cart(self):
        """
        Tests the `new_cart` method to ensure it assigns unique and sequential
        cart IDs.
        """
        for i in range(0, 100):
            act_id = self.marketplace.new_cart()
            self.assertEqual(i, act_id)

    def test_add_to_cart(self):
        """
        Tests the `add_to_cart` method, verifying that a product can be moved
        from the marketplace goods to a consumer's cart.
        """
        from product import Product

        cart_id = self.marketplace.new_cart()
        new_prod = Product("Ceai verde", 10)

        # Note: For this test to pass reliably, the product must first be published.
        # Adding a mock publish here for completeness or assuming it's pre-published.
        prod_id = self.marketplace.register_producer()
        self.marketplace.publish(prod_id, new_prod)


        return_val = self.marketplace.add_to_cart(cart_id, new_prod)

        if return_val:
            # Asserts that the added product is the last item in the cart.
            self.assertEqual(new_prod, self.marketplace.carts[cart_id][-1][0])

    def test_remove_from_cart(self):
        """
        Tests the `remove_from_cart` method, verifying that a product can be
        successfully removed from a cart and returned to the marketplace goods.
        """
        from product import Product

        cart_id = self.marketplace.new_cart()
        new_prod = Product("Ceai verde", 10)

        # Publish the product and then add it to the cart for removal test.
        prod_id = self.marketplace.register_producer()
        self.marketplace.publish(prod_id, new_prod)
        return_val = self.marketplace.add_to_cart(cart_id, new_prod)

        if not return_val:
            # If product couldn't be added to cart, the test cannot proceed to remove it.
            return

        self.marketplace.remove_from_cart(cart_id, new_prod)

        # Asserts that the cart is empty after removal.
        self.assertEqual(0, len(self.marketplace.carts[cart_id]))

    def test_place_order(self):
        """
        Tests the `place_order` method, verifying that an order can be placed
        and the list of ordered products is returned.
        """
        from product import Product

        cart_id = self.marketplace.new_cart()
        new_prod = Product("Ceai verde", 10)

        # Publish the product and add it to the cart for order placement test.
        prod_id = self.marketplace.register_producer()
        self.marketplace.publish(prod_id, new_prod)
        return_val = self.marketplace.add_to_cart(cart_id, new_prod)

        if not return_val:
            return

        prod_list = self.marketplace.place_order(cart_id)

        # Asserts that the ordered product is present in the returned list.
        self.assertEqual(new_prod, prod_list[0])

class Marketplace:
    """
    The central Marketplace class managing products from multiple producers
    and consumer shopping carts. It ensures thread-safe operations for
    registering producers, publishing products, creating carts, adding/removing
    items from carts, and placing orders. It also includes comprehensive logging.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace with its internal data structures,
        ID counters, synchronization locks, and logging configuration.

        :param queue_size_per_producer: The maximum number of items (per product type)
                                        that a single producer can have in the marketplace's
                                        available goods list at any given time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        # Dictionary to store products available from each producer.
        # Key: producer_id, Value: list of (product, availability_status) tuples.
        self.goods = {}
        # Dictionary to store consumer carts.
        # Key: cart_id, Value: list of (product, original_producer_id) tuples.
        self.carts = {}

        self.id_prod = 0 # Counter for assigning unique producer IDs.
        self.id_cart = 0 # Counter for assigning unique cart IDs.

        # Locks for thread-safe access to ID counters.
        self.id_prod_lock = Lock()
        self.id_cart_lock = Lock()
        # Dictionary to store locks for each producer's goods list,
        # ensuring thread-safe access to individual producer inventories.
        self.goods_locks = {}

        # Setup logging for the marketplace.
        self.logger = logging.getLogger("marketLogger")
        self.logger.setLevel(logging.INFO)

        handler = logging.handlers.RotatingFileHandler(filename="marketplace.log", backupCount=15, maxBytes=5242880)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s %(message)s')
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def register_producer(self):
        """
        Registers a new producer with the marketplace.
        Assigns a unique producer ID, initializes an empty goods list for it,
        and creates a specific lock to manage access to this producer's goods.

        :return: The newly assigned unique integer ID for the producer.
        """
        self.logger.info("Entry in function register_producer.")

        # Acquire lock to ensure atomic update of producer ID and goods structure.
        with self.id_prod_lock:
            new_id = self.id_prod
            self.id_prod += 1
            self.goods[new_id] = [] # Initialize an empty list for this producer's goods.
            self.goods_locks[new_id] = Lock() # Create a specific lock for this producer's goods.

        self.logger.info("Leave from function register_producer with return value %d.", new_id)
        return new_id

    def publish(self, producer_id, product):
        """
        Publishes a product by adding it to the specified producer's goods list
        in the marketplace. The operation is successful only if the producer's
        goods list has not reached its maximum capacity (`queue_size_per_producer`).

        :param producer_id: The ID of the producer publishing the product.
        :param product: The product object to be published.
        :return: True if the product was successfully published, False otherwise (e.g., buffer full).
        """
        self.logger.info("Entry in function publish with params producer_id = %d, product = %s.", producer_id, str(product))
        successful_publish = True

        # Acquire the lock specific to this producer's goods to ensure thread-safe modification.
        with self.goods_locks[producer_id]:
            # Check if the producer's goods list has reached its capacity.
            if len(self.goods[producer_id]) == self.queue_size_per_producer:
                successful_publish = False # Cannot publish, buffer is full.
            else:
                # Add the product to the producer's goods list, initially marked as available (1).
                self.goods[producer_id].append((product, 1))

        self.logger.info("Leave from function publish with return value %d.", successful_publish)
        return successful_publish

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.
        Assigns a unique cart ID and initializes an empty list for its contents.

        :return: The newly assigned unique integer ID for the cart.
        """
        self.logger.info("Entry in function new_cart.")

        # Acquire lock to ensure atomic update of cart ID and carts structure.
        with self.id_cart_lock:
            new_id = self.id_cart
            self.id_cart += 1
            self.carts[new_id] = [] # Initialize an empty list for this new cart.

        self.logger.info("Leave from function new_cart with return value %d", new_id)
        return new_id

    def add_to_cart(self, cart_id, product):
        """
        Attempts to add a specified product to a consumer's cart.
        It searches across all producers' goods lists for an available instance
        of the product. If found, the product is moved from the marketplace goods
        to the cart and marked as unavailable in the goods list.

        :param cart_id: The ID of the cart to which the product should be added.
        :param product: The product object to add.
        :return: True if the product was successfully added, False if not found or unavailable.
        """
        self.logger.info("Entry in function add_to_cart with params cart_id = %d, product = %s.", cart_id, product)

        found = False

        # Iterate through all producers to find the product.
        for id_p in self.goods:
            # Acquire the lock for the current producer's goods to prevent race conditions.
            self.goods_locks[id_p].acquire()

            # Iterate through the products offered by the current producer.
            for i in range(0, len(self.goods[id_p])):
                prod_entry = self.goods[id_p][i]

                # Check if the product matches and is available (status 1).
                if (prod_entry[1] == 1) & (prod_entry[0] == product):
                    # Add the product to the consumer's cart.
                    self.carts[cart_id].append((product, id_p))
                    # Mark the product as unavailable (status 0) in the producer's goods.
                    self.goods[id_p][i] = (prod_entry[0], 0)
                    found = True
                    break # Product found and added, exit inner loop.

            self.goods_locks[id_p].release() # Release the lock for this producer's goods.
            if found == True:
                break # Product found and added, exit outer loop.

        self.logger.info("Leave from function add_to_cart with return value %d.", found)
        return found

    def remove_from_cart(self, cart_id, product):
        """
        Removes a specific product from a consumer's cart and returns it to
        its original producer's goods list, marking it as available again.

        :param cart_id: The ID of the cart from which the product should be removed.
        :param product: The product object to remove.
        """
        self.logger.info("Entry in function remove_from_cart with params cart_id = %d, product = %s.", cart_id, str(product))

        _item_to_remove = None
        _producer_of_item = None

        # Iterate through the items in the specified cart.
        for item in self.carts[cart_id]:
            if item[0] == product:
                _producer_of_item = item[1] # Get the original producer ID of the item.
                _item_to_remove = item # Store the item to be removed.
                break # Assuming one instance removed per call.

        if _item_to_remove is not None:
            self.carts[cart_id].remove(_item_to_remove) # Remove the item from the cart.

            # Acquire the lock for the original producer's goods.
            self.goods_locks[_producer_of_item].acquire()

            # Find the item in the producer's goods that was previously marked unavailable (0)
            # and mark it as available (1) again.
            for i in range(0, len(self.goods[_producer_of_item])):
                item_in_goods = self.goods[_producer_of_item][i]

                if (item_in_goods[0] == product) & (item_in_goods[1] == 0):
                    self.goods[_producer_of_item][i] = (product, 1) # Mark as available.
                    break

            self.goods_locks[_producer_of_item].release()
        self.logger.info(f"Leave from function remove_from_cart.")

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.
        It compiles a list of ordered product names and permanently removes
        these specific items from their respective producers' goods lists in the marketplace.

        :param cart_id: The ID of the cart for which to place the order.
        :return: A list of product objects that were successfully ordered.
        """
        self.logger.info("Entry in function place_order with params cart_id = %d.", cart_id)

        ordered_items = []
        # Iterate through all items in the consumer's cart.
        for item in self.carts[cart_id]:
            ordered_items.append(item[0]) # Add product object to the ordered list.
            original_producer_id = item[1] # Get the original producer ID.

            # Acquire the lock for the original producer's goods to modify it.
            self.goods_locks[original_producer_id].acquire()

            # Find and permanently remove the specific (product, 0) entry from the goods list.
            # This represents the item being officially sold.
            _item_to_remove_from_goods = None
            for it in self.goods[original_producer_id]:
                if (it[0] == item[0]) & (it[1] == 0): # Match product and unavailable status.
                    _item_to_remove_from_goods = it
                    break
            
            if _item_to_remove_from_goods is not None:
                self.goods[original_producer_id].remove(_item_to_remove_from_goods)

            self.goods_locks[original_producer_id].release()

        self.logger.info("Leave from function place_order with return value %s.", str(ordered_items))
        return ordered_items


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer thread that continuously generates and publishes
    products to the Marketplace. Each producer has a defined list of products
    to offer, including quantities and delays between publications.
    It includes retry logic if the marketplace buffer is full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        :param products: A list of products this producer will offer. Each element
                         is a tuple: (product_object, quantity_to_publish, time_to_sleep_after_each_publish).
        :param marketplace: The Marketplace instance with which the producer interacts.
        :param republish_wait_time: The time in seconds to wait before retrying
                                    to publish a product if the marketplace buffer is full.
        :param kwargs: Arbitrary keyword arguments passed to the Thread constructor,
                       e.g., 'name' for thread identification.
        """
        super().__init__(kwargs=kwargs, daemon=True) # Set as daemon thread for graceful exit.

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self._id = 0 # Stores the unique ID assigned by the marketplace.

    def run(self):
        """
        The main execution method for the Producer thread.
        It first registers itself with the marketplace, then enters an infinite
        loop to continuously publish its predefined list of products, respecting
        quantities and delays, and retrying if the marketplace buffer is full.
        """
        # Register the producer with the marketplace to obtain a unique ID.
        self._id = self.marketplace.register_producer()

        # Infinite loop to continuously publish products.
        while True:
            # Iterate through each product type the producer offers.
            for prod_info in self.products:
                count_published = 0
                # Publish the specified quantity of the current product type.
                while count_published < prod_info[1]: # prod_info[1] is the quantity to publish.
                    # Attempt to publish the product to the marketplace.
                    if self.marketplace.publish(self._id, prod_info[0]): # prod_info[0] is the product object.
                        count_published += 1
                        # Wait for the specified time after successfully publishing one item.
                        sleep(prod_info[2]) # prod_info[2] is the sleep time.
                    else:
                        # If publishing fails (marketplace buffer full), wait and retry.
                        sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Base class for all products in the marketplace.
    Uses Python's `dataclass` decorator for concise data class definition.
    Instances are immutable (`frozen=True`).
    """
    name: str  # The name or type of the product (e.g., "Laptop", "Coffee").
    price: int # The price of a single unit of the product.


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Represents a specific product: Tea.
    Inherits from `Product` and adds a specific attribute for tea type.
    """
    type: str  # The variety or blend of the tea (e.g., "Green", "Black", "Herbal").


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Represents a specific product: Coffee.
    Inherits from `Product` and adds attributes specific to coffee characteristics.
    """
    acidity: str      # Describes the acidity level of the coffee (e.g., "Low", "Medium", "High").
    roast_level: str  # Describes the roast level of the coffee (e.g., "Light", "Medium", "Dark").
