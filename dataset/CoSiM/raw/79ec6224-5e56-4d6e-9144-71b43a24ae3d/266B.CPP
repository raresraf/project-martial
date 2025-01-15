// http://codeforces.com/problemset/problem/266/B
#include <bits/stdc++.h>
using namespace std;

int main() {
  string queue;
  int n, t;
  cin >> n;
  cin >> t;
  cin >> queue;
  for (int i = 0; i < t; i++) {
    // run the loop t times (means t seconds)
    for (int j = 0; j < n - 1; j++) {
      if (queue[j] == 'B' && queue[j + 1] == 'G') {
        queue[j] = 'G';
        queue[j + 1] = 'B';
        j++;
      }
    }
  }
  cout << queue << endl;
}
