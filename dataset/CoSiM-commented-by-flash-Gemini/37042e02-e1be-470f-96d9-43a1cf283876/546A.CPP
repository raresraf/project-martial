#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

int main() {
  int k, n, w;
  int total_cost = 0;
  int next_banana;
  cin >> k >> n >> w;
  for (int i = 1; i <= w; ++i) {
    total_cost += k * i;
  }
  if (n >= total_cost) {
    cout << 0;
  } else {
    cout << total_cost - n;
  }
}