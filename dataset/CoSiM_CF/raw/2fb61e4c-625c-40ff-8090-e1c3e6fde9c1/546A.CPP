#include <iostream>
using namespace std;
int main() {
  unsigned long int k, n, w;
  cin >> k >> n >> w;
  unsigned long int req_money = k * w * (w + 1) / 2;
  int borrow = n > req_money ? 0 : req_money - n;
  cout << borrow;
  return 0;
}