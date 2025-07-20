#include <iostream>

using namespace std;

int getCost(int w, int k) {
  int cost = 0;
  for (int i = 0; i < w; i++) {
    cost += (i + 1) * k;
  }
  return cost;
}

int main() {
  int k, n, w;
  cin >> k >> n >> w;
  int cost = getCost(w, k);
  if (n >= cost) {
    cout << 0 << endl;
  } else {
    cout << (cost - n) << endl;
  }

  return 0;
}