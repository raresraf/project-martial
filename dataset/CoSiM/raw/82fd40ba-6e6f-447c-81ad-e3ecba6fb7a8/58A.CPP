#include <bits/stdc++.h>

using namespace std;

int main(void) {
  int n;
  cin >> n;
  for (int i = 0; i < n; i++) {
    int min_limit, max_limit, interval;
    cin >> min_limit >> max_limit >> interval;

    if (interval < min_limit)
      cout << interval << "\n";
    else {
      int q = max_limit / interval;
      cout << interval * (q + 1) << "\n";
    }
  }
}
