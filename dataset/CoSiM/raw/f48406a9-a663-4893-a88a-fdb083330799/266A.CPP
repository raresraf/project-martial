#include <bits/stdc++.h>

using namespace std;

int main() {
  string str;
  int count = 0, n;
  cin >> n >> str;
  for (int i = 0; i < n; i++) {
    if (str[i] == str[i - 1])
      count++;
  }
  cout << count << endl;

  return 0;
}
