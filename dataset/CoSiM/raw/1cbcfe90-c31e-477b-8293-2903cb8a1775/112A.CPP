#include <bits/stdc++.h>
using namespace std;

int main() {
  string s1, s2;
  int cmp1, cmp2;
  int result = 0;
  cin >> s1;
  cin >> s2;
  for (int i = 0; i < s1.size(); i++) {
    if (s1[i] >= 97) {
      s1[i] -= 32;
    }
    if (s2[i] >= 97) {
      s2[i] -= 32;
    }
    if (s1[i] > s2[i]) {
      result = 1;
      break;
    }
    if (s1[i] < s2[i]) {
      result = -1;
      break;
    }
  }
  cout << result;
  return 0;
}
