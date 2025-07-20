#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <ctype.h>
#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <vector>
#define ll long long

using namespace std;

int main() {
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  while (t--) {
    for (int i = 0; i < s.size() - 1; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        i += 1;
      }
    }
  }
  cout << s << "\n";
  return 0;
}
