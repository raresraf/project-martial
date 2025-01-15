#include <bits/stdc++.h>
using namespace std;

int main() {
  string str;
  cin >> str;

  regex pattern(".*h.*e.*l.*l.*o.*");

  if (regex_match(str, pattern)) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
