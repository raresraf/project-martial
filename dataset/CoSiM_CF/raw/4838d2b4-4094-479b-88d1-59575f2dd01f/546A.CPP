#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <map>
#include <math.h>
#include <set>
#include <stdio.h>
#include <string>

using namespace std;

int main() {
  int k, n, w;
  cin >> k >> n >> w;
  int total = k * w * (w + 1) / 2;
  cout << max(total - n, 0);

  return 0;
}