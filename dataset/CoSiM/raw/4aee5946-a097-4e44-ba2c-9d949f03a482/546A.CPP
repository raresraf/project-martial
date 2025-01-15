#include <iostream>
using namespace std;
int main() {
  int k, w;
  long n, amountToBuyWBananas = 0;
  cin >> k >> n >> w;
  amountToBuyWBananas = ((w * (w + 1)) / 2) * k;
  if (n - amountToBuyWBananas < 0)
    cout << abs(n - amountToBuyWBananas);
  else {
    cout << "0";
  }
}