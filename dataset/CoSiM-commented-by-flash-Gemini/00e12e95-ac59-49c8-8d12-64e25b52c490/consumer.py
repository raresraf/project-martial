
"""
This module simulates a multi-threaded marketplace system.
It defines `Consumer` threads that perform shopping operations and `Producer`
threads that generate and publish products. The `Marketplace` class manages
the interactions between consumers and producers, ensuring thread-safe operations
for product availability, cart management, and order placement. This version
leverages `UUID` for unique identifiers and `dataclasses` for structured data.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Thread
from time import sleep
from typing import Dict, List, Tuple

from .marketplace import Marketplace
from .product import Product

class Consumer(Thread):
    """
    Represents a consumer thread that performs shopping operations
    (adding/removing products) from a marketplace based on a predefined
    list of actions. Each consumer operates on a series of carts.
    """

    @dataclass
    class Operation():
        """
        Represents a single shopping operation to be performed by a consumer.
        """
        type: str    # Type of operation: 'add' or 'remove'.
        product: Product # The product involved in the operation.
        quantity: int  # The quantity of the product for this operation.

        @classmethod
        def from_dict(
            cls,
            dict: Dict
        ) -> Operation:
            """
            Creates an Operation object from a dictionary representation.
            :param dict: A dictionary containing 'type', 'product', and 'quantity'.
            :return: An initialized Operation instance.
            """
            return cls(
                type=dict['type'],
                product=dict['product'],
                quantity=dict['quantity']
            )


    def __init__(
        self,
        carts: List[List[Dict]],
        marketplace: Marketplace,
        retry_wait_time: int,
        **kwargs
    ):
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
        Thread.__init__(self, **kwargs)
        # Convert dictionary-based cart operations into nested Operation dataclass objects.
        self.operations = [[self.Operation.from_dict(op) for op in cart] for cart in carts]
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution method for the Consumer thread.
        It iterates through the predefined cart action sequences. For each sequence,
        it creates a new cart, performs all specified add/remove operations,
        and then places the order, printing the items bought.
        """
        # Iterate through each sequence of operations, effectively representing different carts.
        for cart_ops_sequence in self.operations:
            # Create a new unique cart ID for the current sequence of operations.
            c_id = self.marketplace.new_cart()

            # Process each individual operation (add or remove) within the current cart's sequence.
            for op in cart_ops_sequence:
                # Handle 'add' operations.
                if op.type == 'add':
                    # Continuously attempt to add the product until the desired quantity is met.
                    while op.quantity:
                        if self.marketplace.add_to_cart(c_id, op.product):
                            op.quantity -= 1  # Decrement remaining quantity if successful.
                        else:
                            sleep(self.retry_wait_time)  # Wait and retry if adding fails (product unavailable).
                # Handle 'remove' operations.
                elif op.type == 'remove':
                    # Continuously attempt to remove the product until the desired quantity is met.
                    while op.quantity:
                        self.marketplace.remove_from_cart(c_id, op.product)
                        op.quantity -= 1  # Decrement remaining quantity after removal.

            # After all operations for a cart are processed, place the order.
            # Then, print the list of products that were successfully bought.
            for p in self.marketplace.place_order(c_id):
                print(f'{self.name} bought {p}')


from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Dict, List, NamedTuple, Optional
from uuid import UUID, uuid4

from .product import Product

class Marketplace:
    """
    The central Marketplace class managing products from multiple producers
    and consumer shopping carts. It ensures thread-safe operations for
    registering producers, publishing products, creating carts, adding/removing
    items from carts, and placing orders. Uses UUIDs for unique IDs.
    """

    class BrandedProduct(NamedTuple):
        """
        A NamedTuple representing a product along with the UUID of its producer.
        Used to track ownership of products within consumer carts.
        """
        producer_id: UUID
        product: Product

    def __init__(
        self,
        queue_size_per_producer: int
    ):
        """
        Initializes the Marketplace with its internal data structures and locks.

        :param queue_size_per_producer: The maximum number of products (per product type)
                                        that a single producer can have in the marketplace's
                                        available goods list at any given time.
        """
        # Maximum number of products a single producer can have in stock.
        self.queue_size_per_producer: int = queue_size_per_producer
        # Dictionary storing products available from each producer.
        # Key: producer_id (UUID), Value: List of Product objects.
        self.producer_lot: Dict[UUID, List[Product]] = defaultdict(list)
        # Dictionary storing consumer carts.
        # Key: cart_id (UUID), Value: List of BrandedProduct objects in the cart.
        self.consumers: Dict[UUID, List[self.BrandedProduct]] = defaultdict(list)

        # Lock to protect access to `producer_lot` during publish operations.
        self.p_lock = Lock()
        # Lock to protect access during add_to_cart operations to prevent race conditions.
        self.add_to_cart_lock = Lock()

    def register_producer(self) -> UUID:
        """
        Registers a new producer with the marketplace, assigning it a unique UUID.
        This UUID will be used to identify the producer's products.

        :return: A unique UUID assigned to the new producer.
        """
        return uuid4()

    def publish(
        self,
        producer_id: UUID,
        product: Product
    ) -> bool:
        """
        Publishes a product by adding it to the specified producer's lot
        in the marketplace. The operation is successful only if the producer's
        lot has not reached its maximum capacity (`queue_size_per_producer`).

        :param producer_id: The UUID of the producer publishing the product.
        :param product: The product object to be published.
        :return: True if the product was successfully published, False otherwise (e.g., lot full).
        """
        # Acquire lock to ensure thread-safe modification of producer lots.
        with self.p_lock:
            # Check if the producer's lot has reached its capacity.
            if len(self.producer_lot[producer_id]) == self.queue_size_per_producer:
                return False  # Cannot publish, lot is full.

            self.producer_lot[producer_id].append(product)  # Add the product to the producer's lot.
            return True

    def new_cart(self) -> UUID:
        """
        Creates a new, empty shopping cart and assigns it a unique UUID.

        :return: A unique UUID for the newly created cart.
        """
        return uuid4()

    def add_to_cart(
        self,
        cart_id: UUID,
        product: Product
    ) -> bool:
        """
        Attempts to add a specified product to a consumer's cart.
        It searches across all producers' lots for an available instance
        of the product. If found, the product is moved from the producer's lot
        to the cart (as a BrandedProduct) and removed from the producer's lot.

        :param cart_id: The UUID of the cart to which the product should be added.
        :param product: The product object to add.
        :return: True if the product was successfully added, False if not found or unavailable.
        """
        # Acquire lock to ensure atomic operations during cart modification and product movement.
        with self.add_to_cart_lock:
            # Iterate through all producers' lots to find the product.
            for p_id, products in self.producer_lot.items():
                # Check if the product exists in the current producer's lot.
                if product in products:
                    # Add the product (with its producer's UUID) to the consumer's cart.
                    self.consumers[cart_id].append(self.BrandedProduct(p_id, product))
                    products.remove(product)  # Remove the product from the producer's lot.
                    return True  # Product found and added.

            return False  # Product not found in any producer's lot.

    def remove_from_cart(
        self,
        cart_id: UUID,
        product: Product
    ):
        """
        Removes a specific product from a consumer's cart and returns it to
        its original producer's lot, making it available again.

        :param cart_id: The UUID of the cart from which the product should be removed.
        :param product: The product object to remove.
        """
        cart = self.consumers[cart_id]  # Get the consumer's cart.
        # Iterate through items in the cart to find the matching BrandedProduct.
        for bp in cart:
            # If the product matches.
            if bp.product == product:
                # Return the product to its original producer's lot.
                self.producer_lot[bp.producer_id].append(bp.product)
                cart.remove(bp)  # Remove the BrandedProduct from the cart.
                break  # Assuming one instance removed per call.

    def place_order(
        self,
        cart_id: UUID
    ) -> List[Product]:
        """
        Finalizes an order for a given cart.
        It compiles a list of ordered product objects from the cart.
        Note: Products are *not* removed from producer_lot here as they are
        already removed during `add_to_cart`.

        :param cart_id: The UUID of the cart for which to place the order.
        :return: A list of product objects that were successfully ordered.
        """
        # Extract only the Product objects from the BrandedProduct NamedTuples in the cart.
        return [bp.product for bp in self.consumers[cart_id]]


from __future__ import annotations


from dataclasses import dataclass
from threading import Thread
from time import sleep
from typing import List, Tuple


from .marketplace import Marketplace
from .product import Product


class Producer(Thread):
    """
    Represents a producer thread that continuously generates and publishes
    products to the Marketplace. Each producer has a defined list of products
    to offer, including quantities and delays between publications.
    It includes retry logic if the marketplace buffer is full.
    """

    @dataclass
    class ProductionLine():
        """
        Represents a single product production line for a producer.
        """
        product: Product # The product object to produce.
        count: int     # The quantity of this product to produce in one cycle.
        time: float    # The time delay (in seconds) after publishing one item.

        @classmethod
        def from_tuple(
            cls,
            tup: Tuple[Product, int, float]
        ):
            """
            Creates a ProductionLine object from a tuple representation.
            :param tup: A tuple containing (product, count, time).
            :return: An initialized ProductionLine instance.
            """
            return cls(
                product=tup[0],
                count=tup[1],
                time=tup[2]
            )


    def __init__(
        self,
        products: List[Tuple[Product, int, float]],
        marketplace: Marketplace,
        republish_wait_time: float,
        **kwargs
    ):
        """
        Initializes a Producer thread.

        :param products: A list of tuples, where each tuple describes a product
                         production line: (product_object, quantity_to_publish_per_cycle, time_delay_after_each_publish).
        :param marketplace: The Marketplace instance with which the producer interacts.
        :param republish_wait_time: The time in seconds to wait before retrying
                                    to publish a product if the marketplace lot is full.
        :param kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        # Convert tuple-based product lines into ProductionLine dataclass objects.
        self.production = [self.ProductionLine.from_tuple(pl) for pl in products]
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution method for the Producer thread.
        It first registers itself with the marketplace, then enters an infinite
        loop to continuously publish its predefined list of products, respecting
        quantities and delays, and retrying if the marketplace lot is full.
        """
        # Register the producer with the marketplace to obtain a unique UUID.
        p_id = self.marketplace.register_producer()

        # Infinite loop for continuous product publishing.
        while True:
            # Iterate through each defined production line for this producer.
            for prod_line in self.production:
                # Wait for the initial production time delay.
                sleep(prod_line.time)

                # Publish the specified quantity of the current product type.
                _count = prod_line.count
                while _count:
                    if self.marketplace.publish(p_id, prod_line.product):
                        _count -= 1  # Decrement remaining quantity if successful.
                    else:
                        sleep(self.republish_wait_time)  # Wait and retry if publishing fails (lot full).
