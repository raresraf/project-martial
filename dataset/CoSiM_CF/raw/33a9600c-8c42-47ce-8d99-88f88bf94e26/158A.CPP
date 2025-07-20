#include <algorithm>
#include <bits/stdc++.h>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>
using namespace std;

int main() {

  int n, k, input, requiredscore, cunt = 0;
  cin >> n >> k;
  vector<int> v;
  for (int i = 0; i < n; i++) {
    cin >> input;
    v.push_back(input);
    if (i + 1 == k) {
      requiredscore = v[i];
    }
  }
  for (int i = 0; i < n; i++) {
    if ((requiredscore <= v[i]) && (0 < v[i])) {
      cunt++;
    }
  }
  cout << cunt;

  return 0;
}
