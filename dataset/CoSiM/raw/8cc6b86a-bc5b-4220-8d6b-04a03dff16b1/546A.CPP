#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;
int main() {
  int k, n, w;
  cin >> k >> n >> w;
  int cost = 0;
  for (int i = 1; i <= w; i++) {
    cost = cost + (i * k); // 1*3+2*3+3*3+4*3
  }

  if (cost > n) {
    cout << cost - n;
  } else {
    cout << 0;
  }
}
