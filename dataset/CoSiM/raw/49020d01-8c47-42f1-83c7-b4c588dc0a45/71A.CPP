#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <vector>
using namespace std;

int main() {
  int n;
  cin >> n;
  for (int i = 0; i < n; i++) {
    string t;
    cin >> t;
    if (t.length() > 10) {
      printf("%c%d%c\n", t[0], t.length() - 2, t[t.length() - 1]);
    } else
      cout << t << endl;
  }

  return 0;
}
