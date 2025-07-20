#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;
  string str;
  cin >> str;
  int count = 0;
  for (int i = 1; i < str.length(); i++) {
    if (str[i - 1] == str[i]) {
      count++;
    }
  }

  cout << count << endl;
  return 0;
}