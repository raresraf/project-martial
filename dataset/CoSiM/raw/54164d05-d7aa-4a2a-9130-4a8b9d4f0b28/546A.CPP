#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace std;

int main() {
  int k, n, w, temp = 0;
  cin >> k >> n >> w;
  temp = k * (w * (w + 1) / 2);
  if (n >= temp) {
    cout << 0 << endl;
  } else {
    cout << temp - n << endl;
  }
  return 0;
}