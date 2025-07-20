#include <iostream>

using namespace std;

int main() {
  int k, w;
  long long int n;
  cin >> k >> n >> w;
  long long int bill = 0;

  for (auto i = 1; i <= w; i++) {
    bill += i * k;
  }

  const auto need_to_borrow = n < bill ? bill - n : 0;

  cout << need_to_borrow << endl;

  return 0;
}
