#include <algorithm>
#include <cstdio>
#include <iostream>
#include <vector>
using namespace std;
int main() {
  int n, k;
  cin >> n >> k;
  int count = 0;
  int x, i;
  for (i = 0; i < k; i++) {
    scanf("%d", &x);
    if (x > 0)
      count++;
  }
  for (; i < n; i++) {
    int y;
    scanf("%d", &y);
    if (y > 0 && y == x)
      count++;
  }
  cout << count << "\n";
}