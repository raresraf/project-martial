#include <algorithm>
#include <array>
#include <assert.h>
#include <bitset>
#include <cassert>
#include <cmath>
#include <deque>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <math.h>
#include <numeric>
#include <queue>
#include <search.h>
#include <set>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;

int main() {
  int n, t, ind = 0;
  string s;
  char b = 'B', g = 'G';
  cin >> n >> t >> s;
  for (int i = 0; i < t; i++) {
    while (ind < s.size() - 1) {
      if (s[ind] == b && s[ind + 1] == g) {
        s[ind] = g;
        s[ind + 1] = b;
        ind += 2;
      } else {
        ind++;
      }
    }
    ind = 0;
  }
  cout << s;
  return 0;
}
