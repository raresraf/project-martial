#include <bits/stdc++.h>

using namespace std;

int main() {
  int k, w;
  long int n, count = 0, cost, i;
  scanf("%d %d %d", &k, &n, &w);
  for (i = 1; i <= w; i++) {
    count = count + (i * k);
  }
  cost = n - count;
  if (cost < 0) {
    cout << abs(cost) << endl;
  } else {
    cout << "0" << endl;
  }
}
