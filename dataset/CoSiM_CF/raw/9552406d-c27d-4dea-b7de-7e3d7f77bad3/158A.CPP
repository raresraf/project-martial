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
  int n, k;
  int s[110];
  cin >> n >> k;
  for (int i = 1; i <= n; i++) {
    cin >> s[i];
  }

  int ans = 0;
  for (int i = 1; i <= n; i++) {
    if (s[i] > 0 && s[i] >= s[k])
      ans++;
    else
      break;
  }
  cout << ans << endl;

  return 0;
}
