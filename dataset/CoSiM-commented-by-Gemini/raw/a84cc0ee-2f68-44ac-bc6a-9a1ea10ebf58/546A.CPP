#include <iostream>
using namespace std;

int main() {
  int n, k, w, counter = 1, sum = 0;
  cin >> k >> n >> w;
  for (int i = 1; i <= w; i++) {
    sum += k * counter;
    counter++;
  }
  if (sum > n)
    cout << sum - n << "\n";

  else
    cout << "0\n";

  return 0;
}
