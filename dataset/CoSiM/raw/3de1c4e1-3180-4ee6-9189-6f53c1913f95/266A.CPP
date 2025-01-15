#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  int t, count = 0;
  cin >> t >> s;
  for (int i = 0; i < t - 1; i++) {
    if (s[i] == s[i + 1]) {
      count++;
    }
  }
  cout << count << endl;
}
