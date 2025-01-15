#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;

  int total = 0;

  vector<int> v;
  for (int i = 0; i < n; i++) {
    int check = 0;
    for (int j = 0; j < 3; j++) {
      int val;
      cin >> val;
      if (val == 1) {
        check++;
      }
      v.push_back(val);
    }
    if (check >= 2) {
      total++;
    }
  }
  cout << total << endl;
  return 0;
}