#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  string x;
  int result = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> x;
    if (x.find("++") != string::npos) {
      result++;
    } else {
      result--;
    }
  }
  cout << result;
  return 0;
}
