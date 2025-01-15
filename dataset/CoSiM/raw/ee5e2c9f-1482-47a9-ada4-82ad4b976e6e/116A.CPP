#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  int entry, exit;
  int ans = 0;
  int final = INT_MIN;
  for (int i = 0; i < n; i++) {
    cin >> exit >> entry;
    ans += entry - exit;
    final = max(ans, final);
  }
  cout << final << endl;
}