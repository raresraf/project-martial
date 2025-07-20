#include <iostream>
using namespace std;

int main() {
  int k, n, w;
  cin >> k >> n >> w;
  int cost = 0;
  for (int i = 1; i <= w; i++)
    cost += i * k;
  cout << max(0, cost - n) << "\n";
}
