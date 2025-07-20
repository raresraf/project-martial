#include <bits/stdc++.h>
using namespace std;
int main() {
  char s;
  vector<char> v;
  int t, n;
  cin >> n;
  cin >> t;
  for (int k = 0; k < n; k++) {
    cin >> s;
    v.push_back(s);
  }
  for (int i = 1; i <= t; i++) {
    for (int j = 0; j < (v.size() - 1); j++) {
      if (v[j] == 'B' && v[j + 1] == 'G') {
        swap(v[j + 1], v[j]);
        j++;
      }
    }
  }
  for (int k = 0; k < n; k++) {
    cout << v[k];
  }
  cout << endl;
}
