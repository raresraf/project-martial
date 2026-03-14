"""
@65c11c60-12e9-4538-ad14-fbb4456b4f62/consumer.py
@brief Implements a simulated producer-consumer marketplace system using multi-threading.

This module defines classes for:
- `Consumer`: Represents a buyer who places orders on the marketplace.
- `Marketplace`: The central hub where producers publish products and consumers place orders.
- `Producer`: Represents a seller who publishes products to the marketplace.
- `Product`: Base class for marketplace items, with `Tea` and `Coffee` as specific types.

The system uses `threading.Thread` for concurrency, `threading.Lock` for protecting
shared data access within the `Marketplace`, and `time.sleep` for simulating delays.

Algorithm:
- Producers register with the marketplace and continuously publish products up to a queue size limit.
- Consumers create carts, add/remove products to/from their carts, handling retry logic if products are unavailable.
- The Marketplace manages product availability, producer queues, and cart contents,
  ensuring thread-safe operations through various locks.

Time Complexity:
- `Consumer.run`: O(C * P * Q * R) where C is carts, P is products per cart, Q is quantity, R is retries.
- `Marketplace.publish`: O(1) amortized.
- `Marketplace.add_to_cart`: O(N) due to `list.remove`, N is available products.
- `Producer.run`: O(P * R) where P is products a producer offers, R is republish retries.
Space Complexity:
- `Consumer`: O(C * P) for carts.
- `Marketplace`: O(P_total + C_total) for producers_products, available_products, carts.
- `Producer`: O(P) for products.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    @brief Represents a consumer (buyer) in the marketplace.

    Each consumer operates as a separate thread, simulating placing orders
    by adding and removing products from a cart managed by the Marketplace.
    It includes retry logic for adding/removing products if they are
    temporarily unavailable.
    """
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a new Consumer thread.

        @param carts: A list of carts, where each cart is a list of product dictionaries
                      specifying product, quantity, and type ("add" or "remove").
        @param marketplace: The Marketplace instance the consumer interacts with.
        @param retry_wait_time: The time (in seconds) to wait before retrying an
                                 add/remove operation if it fails.
        @param kwargs: Additional keyword arguments for the Thread constructor.
        """
        
        Thread.__init__(self, **kwargs)
        self.carts = carts # List of shopping carts this consumer will process.
        self.marketplace = marketplace # Reference to the shared marketplace.
        self.retry_wait_time = retry_wait_time # Time to wait on failed add/remove attempts.


    def run(self):
        """
        @brief The main execution loop for the Consumer thread.

        Iterates through each assigned cart, creates a new cart on the marketplace,
        and attempts to add or remove products as specified. It handles retries
        for unavailable products and finally places the order.
        """
        for cart in self.carts: # Processes each cart assigned to this consumer.
            # Block Logic: Create a new cart ID in the marketplace for this shopping session.
            cart_id = self.marketplace.new_cart() 
            
            for products in cart: # Iterates through each product type and quantity specified in the current cart.
                now_quantity = 0 # Counter for the quantity of the current product successfully added/removed.
                
                # Block Logic: Loop until the desired quantity of the current product is added/removed.
                while now_quantity < products["quantity"]:
                    check = False # Flag to indicate if the add/remove operation was successful.
                    
                    # Block Logic: Perform 'add' or 'remove' operation based on product type.
                    if products["type"] == "add":
                        check = self.marketplace.add_to_cart(cart_id, products["product"])
                    if products["type"] == "remove":
                        check = self.marketplace.remove_from_cart(cart_id, products["product"])
                    
                    # Block Logic: Handle retry if the operation failed.
                    if check is False:
                        time.sleep(self.retry_wait_time) # Wait before retrying.
                    else:
                        now_quantity += 1 # Increment successful quantity.
            self.marketplace.place_order(cart_id) # Place the final order for the filled cart.


from threading import Lock, currentThread

class Marketplace:
    """
    @brief The central marketplace managing products, producers, and consumer carts.

    This class handles the core logic of product publication, producer registration,
    cart creation, adding/removing products to/from carts, and order placement.
    It uses locks to ensure thread-safe access to shared data structures.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes a new Marketplace instance.

        @param queue_size_per_producer: The maximum number of products a single
                                         producer can have published at any time.
        """
        
        self.queue_size_per_producer = queue_size_per_producer
        self.no_of_producers = 0 # Counter for registered producers.
        self.producers = {} # Dictionary mapping producer_id to their current published product count.
        self.no_of_carts = 0 # Counter for created carts.
        self.carts = {} # Dictionary mapping cart_id to a list of products in that cart.
        self.producers_products = {} # Dictionary mapping product to the producer_id that published it.
        self.available_products = [] # List of currently available products in the marketplace.
        
        # Locks for protecting shared data access.
        self.lock_reg_producers = Lock() # Protects `no_of_producers` and `producers` during registration.
        self.lock_carts = Lock() # Protects `no_of_carts` and `carts` during cart creation.
        self.lock_producers = Lock() # Protects `producers`, `producers_products`, and `available_products` during publish/add/remove.
        
        

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.

        Assigns a unique ID to the producer and initializes its product count.
        Ensures thread-safe registration.

        @return The unique ID assigned to the new producer.
        """
        
        self.lock_reg_producers.acquire() # Acquire lock for producer registration.
        self.no_of_producers += 1 # Increment producer count.
        producer_id = self.no_of_producers # Assign new ID.

        self.producers[producer_id] = 0 # Initialize published product count for new producer.
        self.lock_reg_producers.release() # Release lock.
        return producer_id

    def publish(self, producer_id, product):
        """
        @brief Publishes a product to the marketplace by a producer.

        The product is added to the list of available products if the producer
        has not exceeded its `queue_size_per_producer` limit.

        @param producer_id: The ID of the producer publishing the product.
        @param product: The product object to publish.
        @return True if the product was successfully published, False otherwise.
        """
        
        # Block Logic: Check if the producer has exceeded its publishing limit.
        if self.producers[int(producer_id)] >= self.queue_size_per_producer:
            return False

        self.producers[int(producer_id)] += 1 # Increment producer's published product count.
        self.producers_products[product] = int(producer_id) # Record which producer published this product.
        self.available_products.append(product) # Add product to general available list.
        return True

    def new_cart(self):
        """
        @brief Creates a new shopping cart and assigns a unique cart ID.

        Ensures thread-safe cart creation.

        @return The unique ID of the newly created cart.
        """
        
        self.lock_carts.acquire() # Acquire lock for cart creation.
        self.no_of_carts += 1 # Increment cart count.
        cart_id = self.no_of_carts # Assign new ID.
        self.carts[cart_id] = [] # Initialize an empty list for the new cart.
        self.lock_carts.release() # Release lock.
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a consumer's cart.

        Ensures that the product is available and then transfers it from
        the general available list to the specified cart. Decrements the
        producer's published count and removes the product from available_products.
        Ensures thread-safe operation.

        @param cart_id: The ID of the cart to add the product to.
        @param product: The product object to add.
        @return True if the product was successfully added, False if not available.
        """
        
        self.lock_producers.acquire() # Acquire lock for product availability.
        
        # Block Logic: Check if the product is currently available.
        if product not in self.available_products:
            self.lock_producers.release() # Release lock if product not available.
            return False

        prod_id = self.producers_products[product] # Get the ID of the producer.

        self.producers[prod_id] -= 1 # Decrement producer's published count.
        self.available_products.remove(product) # Remove product from general available list.
        self.carts[cart_id].append(product) # Add product to the consumer's cart.
        self.lock_producers.release() # Release lock.
        return True


    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a consumer's cart and makes it available again.

        Adds the product back to the `available_products` list and increments
        the publishing producer's count. This method is not fully thread-safe in
        its current form as it modifies `carts` and `available_products` outside a lock.

        @param cart_id: The ID of the cart from which to remove the product.
        @param product: The product object to remove.
        """
        
        # NOTE: This method lacks proper locking around `self.carts[cart_id].remove(product)`
        # and `self.available_products.append(product)`, which can lead to race conditions
        # if multiple threads try to modify these lists simultaneously.
        self.carts[cart_id].remove(product) # Remove product from the cart.
        self.available_products.append(product) # Add product back to available list.
        self.lock_producers.acquire() # Acquire lock for producer-related updates.
        self.producers[self.producers_products[product]] += 1 # Increment producer's published count.
        self.lock_producers.release() # Release lock.


    def place_order(self, cart_id):
        """
        @brief Places an order for the specified cart.

        Removes the cart from the marketplace and prints a message for each
        product bought. This method is not fully thread-safe as `carts.pop` is not
        protected by `lock_carts`.

        @param cart_id: The ID of the cart to place an order for.
        """
        
        prod_list = self.carts.pop(cart_id) # Remove cart from active carts.
        for product in prod_list: # For each product in the ordered list.
            # Prints the name of the ordering thread and the product bought.
            print("{} bought {}".format(currentThread().getName(), product))


from threading import Thread
import time


class Producer(Thread):
    """
    @brief Represents a producer (seller) in the marketplace.

    Each producer operates as a separate thread, continuously publishing a
    set of products to the Marketplace. It handles retry logic if the
    marketplace's queue for that producer is full.
    """
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a new Producer thread.

        @param products: A list of products this producer offers. Each item is a
                         sublist: `[product_object, quantity_to_publish, publish_delay]`.
        @param marketplace: The Marketplace instance the producer interacts with.
        @param republish_wait_time: The time (in seconds) to wait before retrying
                                     to publish a product if the marketplace queue is full.
        @param kwargs: Additional keyword arguments for the Thread constructor.
        """
        
        Thread.__init__(self, **kwargs)
        self.products = products # List of products to be published.
        self.marketplace = marketplace # Reference to the shared marketplace.
        self.republish_wait_time = republish_wait_time # Time to wait on failed publish attempts.
        self.producer_id = self.marketplace.register_producer() # Registers with the marketplace and gets a unique ID.

    def run(self):
        """
        @brief The main execution loop for the Producer thread.

        Continuously attempts to publish its products to the marketplace.
        It respects per-product publish delays and retries if the marketplace's
        queue for this producer is full.
        """
        while True: # Loop indefinitely to simulate continuous production.
            for sublist in self.products: # Iterates through each type of product the producer offers.
                count = 0 # Counter for how many of the current product have been published.
                
                # Block Logic: Loop until the desired quantity of the current product is published.
                while count < sublist[1]:
                    # Inline: Attempt to publish the product to the marketplace.
                    check = self.marketplace.publish(str(self.producer_id), sublist[0])
                    if check: # If publishing was successful.
                        time.sleep(sublist[2]) # Wait for the specified publish delay.
                        count += 1 # Increment published count.
                    else: # If publishing failed (e.g., marketplace queue full).
                        time.sleep(self.republish_wait_time) # Wait before retrying.


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base class for products in the marketplace.

    Uses `dataclasses` to automatically generate `__init__`, `__repr__`,
    and `__eq__`/`__hash__` methods.
    - `init=True`: Generates an `__init__` method.
    - `repr=True`: Generates a `__repr__` method.
    - `order=False`: Does not generate comparison methods (`<`, `<=`, etc.).
    - `frozen=True`: Makes instances immutable (hashable).

    @field name: The name of the product (string).
    @field price: The price of the product (integer).
    """
    
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Represents a Tea product, subclass of Product.

    @field type: The type of tea (e.g., "Green", "Black", "Herbal").
    """
    
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Represents a Coffee product, subclass of Product.

    @field acidity: The acidity level of the coffee.
    @field roast_level: The roast level of the coffee.
    """
    
    acidity: str
    roast_level: str
