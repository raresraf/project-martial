#include <iostream>

using namespace std;

int main() {
  int k, n, w, sum = 0;
  cin >> k >> n >> w;
  sum += k;

  for (int i = 2; i <= w; i++) {
    sum += i * k;
  }

  if (sum > n) {
    cout << sum - n;
  } else {
    cout << 0;
  }

  return 0;
}