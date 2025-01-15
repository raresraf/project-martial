#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, dif = 0;
  cin >> n;
  string s;
  cin >> s;
  int size = s.size();
  for (int i = 0; i < size - 1; i++) {
    if (s[i] == s[i + 1]) {
      dif += 1;
    }
  }
  cout << dif << endl;
}
