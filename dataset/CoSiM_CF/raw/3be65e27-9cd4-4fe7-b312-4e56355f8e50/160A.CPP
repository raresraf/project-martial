#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <ext/hash_map>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <memory.h>
#include <numeric>
#include <set>
#include <sstream>
#include <stack>
#include <utility>
#include <vector>

using namespace std;

int main() {
  vector<int> n;
  int x, i, sum, v = 0, c = 0;
  cin >> x;
  n.resize(x);
  for (i = 0; i < x; ++i) {
    cin >> n[i];
  }
  sort(n.rbegin(), n.rend());
  sum = accumulate(n.begin(), n.end(), 0);
  for (i = 0; i < x; ++i) {
    v += n[i];
    c++;
    if (v > (sum - v))
      break;
  }
  cout << c << endl;
  return 0;
}
