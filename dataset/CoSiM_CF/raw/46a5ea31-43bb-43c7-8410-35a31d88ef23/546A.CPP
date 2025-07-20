#include <algorithm>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <vector>

using namespace std;

int main() {
  int k, n, w;
  cin >> k >> n >> w;
  int basePrice = k;
  int total = k;
  for (int i = 2; i <= w; i++) {
    total += i * k;
  }
  cout << max(total - n, 0);
  return 0;
}