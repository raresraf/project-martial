#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k;
  cin >> n >> k;
  vector<int> vals;
  for (int i = 0; i < n; i++) {
    int temp;
    cin >> temp;
    vals.push_back(temp);
  }
  int count = 0;
  for (int i = k - 1; i < n; i++) {
    if (vals[i] != 0 && vals[i] == vals[k - 1]) {
      count++;
    } else {
      break;
    }
  }
  for (int i = k - 2; i >= 0; i--) {
    if (vals[i] != 0) {
      count++;
    }
  }
  cout << count;
}
