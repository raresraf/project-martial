

import time


from threading import Thread
from typing import List

from tema.marketplace import Marketplace
from tema.product import Product


class Operation:
    def __init__(self, op_type: str, product: Product, quantity: int):
        self.op_type: str = op_type
        self.product: Product = product
        self.quantity: int = quantity


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        


        super(Consumer, self).__init__(**kwargs)
        self.carts: List[List[Operation]] = []
        for cart_operations in carts:
            ops = []
            for operation in cart_operations:
                ops.append(Operation(
                    operation['type'],
                    operation['product'],
                    operation['quantity'])
                )
            self.carts.append(ops)

        self.marketplace: Marketplace = marketplace
        self.retry_wait_time: float = retry_wait_time

        
        self.products: List[Product] = []

    def run(self):
        
        for operations in self.carts:
            cart_id: int = self.marketplace.new_cart()

            
            while operations:
                if operations[0].op_type == 'add':
                    
                    if self.marketplace.add_to_cart(cart_id, operations[0].product):
                        operations[0].quantity -= 1
                    else:
                        
                        time.sleep(self.retry_wait_time)
                        continue

                    
                    
                    if operations[0].quantity == 0:
                        operations = operations[1:]
                elif operations[0].op_type == 'remove':
                    
                    
                    while operations[0].quantity > 0:
                        self.marketplace.remove_from_cart(cart_id, operations[0].product)
                        operations[0].quantity -= 1
                    
                    operations = operations[1:]

            
            final_products: List[Product] = self.marketplace.place_order(cart_id)
            for product in final_products:
                print(self.name + " bought " + str(product))



from threading import Lock
from typing import List, Dict

from tema.product import Product


class MarketplaceProduct:
    def __init__(self, producer_id: int, product: Product):
        self.producer_id = producer_id
        self.product = product
        self.lock: Lock = Lock()


class Cart:
    def __init__(self, cart_id: int):
        self.cart_id: int = cart_id
        self.products: List[MarketplaceProduct] = []

    def add_product(self, product: MarketplaceProduct):
        self.products.append(product)

    def remove_product(self, product: MarketplaceProduct):
        if product in self.products:
            self.products.remove(product)

    def find_product_in_cart(self, product) -> [MarketplaceProduct]:
        for market_product in self.products:
            if market_product.product == product:
                return market_product

        return None


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        
        self.max_items: int = queue_size_per_producer

        
        self.consumers: List[int] = []
        self.producers: List[int] = []

        
        self.products: Dict[int, List[MarketplaceProduct]] = {}
        self.carts: Dict[int, Cart] = {}

    def register_producer(self) -> int:
        
        
        producer_id = len(self.producers)

        self.producers.append(producer_id)
        self.products[producer_id] = []

        return producer_id

    def publish(self, producer_id, product):
        
        if len(self.products[producer_id]) > self.max_items:
            return False
        else:
            self.products[producer_id].append(MarketplaceProduct(producer_id, product))
            return True

    def new_cart(self):
        
        
        cart_id = len(self.carts) + 1

        self.carts[cart_id] = Cart(cart_id)
        return cart_id

    def add_to_cart(self, cart_id, product):
        
        
        for producer_products in self.products.values():
            for market_product in producer_products:
                
                if market_product.product == product and not market_product.lock.locked():
                    market_product.lock.acquire()  
                    self.carts[cart_id].add_product(market_product)  
                    return True

        
        return False

    def remove_from_cart(self, cart_id, product):
        
        
        market_product: MarketplaceProduct = self.carts[cart_id].find_product_in_cart(product)
        self.carts[cart_id].remove_product(market_product)
        market_product.lock.release()

    def place_order(self, cart_id) -> List[Product]:
        
        
        final_products = [marketProduct.product for marketProduct in self.carts[cart_id].products]

        
        for market_product in self.carts[cart_id].products:
            for producer_product in self.products[market_product.producer_id]:
                if producer_product == market_product:
                    self.products[market_product.producer_id].remove(market_product)
                    break

        return final_products

import time


from threading import Thread
from typing import List

from tema.marketplace import Marketplace
from tema.product import Product



class Production:
    def __init__(self, product: Product, quantity: int, wait_time: float):
        self.product: Product = product
        self.quantity: int = quantity
        self.wait_time: float = wait_time

        self.number_produced = 0


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        


        super(Producer, self).__init__(**kwargs)
        self.productions: List[Production] = []
        for prod in products:
            self.productions.append(Production(prod[0], prod[1], prod[2]))
        self.marketplace: Marketplace = marketplace
        self.republish_wait_time: float = republish_wait_time

        
        self.producer_id: int = self.marketplace.register_producer()

    def run(self):
        
        
        while True:
            
            if self.productions[0].number_produced < self.productions[0].quantity:
                
                if self.marketplace.publish(self.producer_id, self.productions[0].product):
                    self.productions[0].number_produced += 1
                
                time.sleep(self.productions[0].wait_time)
            else:
                
                self.productions[0].number_produced = 0
                self.productions = self.productions[1:] + [self.productions[0]]
