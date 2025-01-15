#include <iostream>

using namespace std;

int main() {
  int n, k, score = -1, cur = -1, i;
  cin >> n >> k;
  for (i = 1; i <= n; ++i) {
    cin >> cur;
    if (i == k && cur != 0)
      score = cur;
    else if (cur == 0 || (i >= k && cur != score))
      break;
  }
  cout << i - 1;
}