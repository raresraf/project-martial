"""
This module implements a simulation of a marketplace with producers and consumers.

It defines classes for:
- `Consumer`: Represents an entity that places orders for products in a marketplace.
- `Marketplace`: Manages products, shopping carts, and order processing, handling concurrency with locks.
- `Producer`: Represents an entity that supplies products to a marketplace.
- `Product`, `Tea`, `Coffee`: Data structures for various product types.
"""


from threading import Thread
import time


class Consumer(Thread):
    """
    Represents a consumer thread that simulates adding and removing products from a shopping cart
    in a marketplace, and then placing the order.
    """
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping carts, where each cart is a list of product dictionaries.
                          Each product dictionary contains "quantity", "type" (add/remove), and "product".
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (float): The time in seconds to wait before retrying an action if it fails.
            **kwargs: Arbitrary keyword arguments passed to the Thread.__init__.
        """
        
        Thread.__init__(self, **kwargs)
        self.carts = carts 
        self.marketplace = marketplace 
        self.retry_wait_time = retry_wait_time


    def run(self):
        """
        Executes the consumer's purchasing logic.
        For each cart defined in `self.carts`, it simulates the process of:
        1. Creating a new shopping cart in the marketplace.
        2. Iteratively adding or removing products based on the cart's specifications.
           - If an add/remove operation fails (e.g., product unavailable), it retries after `retry_wait_time`.
        3. Finally, placing the completed order through the marketplace.
        """
        for cart in self.carts:
            # Block Logic: Initializes a new shopping cart for processing the current consumer's requests.
            # Invariant: A unique cart_id is obtained from the marketplace, ensuring isolation of purchases.
            cart_id = self.marketplace.new_cart()
            for products in cart:
                now_quantity = 0
                # Block Logic: Processes each product type (add/remove) based on the specified quantity.
                # Pre-condition: 'products' dictionary contains "quantity", "type", and "product" keys.
                # Invariant: 'now_quantity' tracks the successfully processed items for the current product.
                while now_quantity < products["quantity"]:
                    if products["type"] == "add":
                        # Block Logic: Attempts to add a product to the cart.
                        # Outcome: 'check' is True if successful, False otherwise.
                        check = self.marketplace.add_to_cart(cart_id, products["product"])
                    if products["type"] == "remove":
                        # Block Logic: Attempts to remove a product from the cart.
                        # Outcome: 'check' is True if successful, False otherwise.
                        check = self.marketplace.remove_from_cart(cart_id, products["product"])
                    if check is False:
                        # Block Logic: If the cart operation fails, the consumer waits and retries.
                        # Rationale: Simulates temporary unavailability or contention in the marketplace.
                        time.sleep(self.retry_wait_time)
                    else:
                        # Block Logic: Increments count on successful cart operation.
                        now_quantity += 1
            # Block Logic: Places the order for all items successfully added to the current cart.
            # Post-condition: The cart is finalized and removed from active processing.
            self.marketplace.place_order(cart_id)


from threading import Lock, currentThread

class Marketplace:
    """
    Manages products from multiple producers, handles consumer shopping carts,
    and facilitates order placement in a thread-safe manner.
    It utilizes locks to synchronize access to shared resources like producer registration,
    carts, and product availability.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace with necessary data structures and locks for concurrency control.

        Args:
            queue_size_per_producer (int): The maximum number of products a single producer can publish
                                           to the marketplace at any given time.
        """
        
        self.queue_size_per_producer = queue_size_per_producer
        self.no_of_producers = 0
        self.producers = {} 
        self.no_of_carts = 0
        self.carts = {}
        self.producers_products = {} 
        self.available_products = [] 
        
        
        self.lock_reg_producers = Lock() 
        self.lock_carts = Lock() 
        self.lock_producers = Lock() 
        
        

    def register_producer(self):
        """
        Registers a new producer with the marketplace, assigning a unique ID
        and initializing their product count to zero.
        Ensures thread-safe registration using a lock.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        
        self.lock_reg_producers.acquire()
        self.no_of_producers += 1
        producer_id = self.no_of_producers
        self.producers[producer_id] = 0
        self.lock_reg_producers.release()
        return producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer to the marketplace.
        The product is added to the list of available products and associated with the producer.
        This operation is conditional on the producer not exceeding its queue size limit.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Product): The product object to be published.

        Returns:
            bool: True if the product was successfully published, False otherwise (if queue is full).
        """
        
        if self.producers[int(producer_id)] >= self.queue_size_per_producer:
            return False

        self.producers[int(producer_id)] += 1
        self.producers_products[product] = int(producer_id)
        self.available_products.append(product)
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart in the marketplace and assigns it a unique ID.
        Ensures thread-safe cart creation using a lock.

        Returns:
            int: The unique ID assigned to the new cart.
        """
        
        self.lock_carts.acquire()
        self.no_of_carts += 1
        cart_id = self.no_of_carts
        self.carts[cart_id] = []
        self.lock_carts.release()
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a specified product to a consumer's cart.
        This operation requires the product to be available and decrements the producer's count for that product.
        Ensures thread-safe product addition using a lock.

        Args:
            cart_id (int): The ID of the cart to which the product will be added.
            product (Product): The product object to add.

        Returns:
            bool: True if the product was successfully added, False if the product is not available.
        """
        
        self.lock_producers.acquire()
        if product not in self.available_products:
            self.lock_producers.release()
            return False

        prod_id = self.producers_products[product]


        self.producers[prod_id] -= 1
        self.available_products.remove(product)
        self.carts[cart_id].append(product)
        self.lock_producers.release()
        return True


    def remove_from_cart(self, cart_id, product):
        """
        Removes a specified product from a consumer's cart and makes it available again in the marketplace.
        Increments the producer's product count.
        Ensures thread-safe product removal using a lock.

        Args:
            cart_id (int): The ID of the cart from which the product will be removed.
            product (Product): The product object to remove.
        """
        
        self.carts[cart_id].remove(product)
        self.available_products.append(product)
        self.lock_producers.acquire()
        self.producers[self.producers_products[product]] += 1
        self.lock_producers.release()


    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart, removes the cart from the marketplace,
        and prints the products bought by the current thread.

        Args:
            cart_id (int): The ID of the cart to place an order for.
        """
        
        prod_list = self.carts.pop(cart_id)
        for product in prod_list:
            print("{} bought {}".format(currentThread().getName(), product))


from threading import Thread
import time


class Producer(Thread):
    """
    Represents a producer thread that continuously publishes products to the marketplace.
    It attempts to publish products based on a predefined list and retries if the marketplace's
    queue limit for the producer is reached.
    """
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products to be published, where each item is a tuple
                             containing (product_object, quantity, publish_interval).
            marketplace (Marketplace): The marketplace instance to interact with.
            republish_wait_time (float): The time in seconds to wait before retrying to publish
                                         a product if the marketplace queue is full.
            **kwargs: Arbitrary keyword arguments passed to the Thread.__init__.
        """
        
        Thread.__init__(self, **kwargs)
        self.products = products 
        self.marketplace = marketplace 
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        Executes the producer's product publishing logic.
        Continuously attempts to publish products from its defined list to the marketplace.
        - The outer `while True` loop ensures the producer runs indefinitely.
        - The inner `for sublist in self.products` iterates through the list of products
          to be published by this producer.
        - The `while count < sublist[1]` loop attempts to publish the specified quantity of a product.
          If `marketplace.publish` returns True, the product is successfully published,
          and the producer waits for `sublist[2]` seconds before publishing the next unit.
          If `marketplace.publish` returns False (e.g., the producer's queue is full),
          the producer waits for `self.republish_wait_time` before retrying the same product.
        """
        while True:
            # Block Logic: Iterates through each product type and quantity specified for this producer.
            # Invariant: 'sublist' contains [product_object, quantity_to_publish, publish_interval].
            for sublist in self.products:
                count = 0
                # Block Logic: Attempts to publish a specific quantity of a single product.
                # Pre-condition: 'sublist[1]' defines the target quantity for the current product.
                # Invariant: 'count' tracks the number of units successfully published for the current product.
                while count < sublist[1]:
                    # Block Logic: Attempts to publish a single unit of the product to the marketplace.
                    # Outcome: 'check' is True if successful, False if the producer's queue is full.
                    check = self.marketplace.publish(str(self.producer_id), sublist[0])
                    if check:
                        # Block Logic: On successful publication, waits for the specified interval.
                        time.sleep(sublist[2])
                        count += 1
                    else:
                        # Block Logic: If publication fails (queue full), waits before retrying.
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Represents a generic product with a name and a price.
    Uses dataclass for concise definition, immutability (frozen=True),
    and automatic __init__, __repr__, and ordering methods.
    """
    
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Represents a type of Product specifically for tea, adding a 'type' attribute.
    Inherits from Product and uses dataclass for concise definition.
    """
    
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Represents a type of Product specifically for coffee, adding 'acidity' and 'roast_level' attributes.
    Inherits from Product and uses dataclass for concise definition.
    """
    
    acidity: str
    roast_level: str