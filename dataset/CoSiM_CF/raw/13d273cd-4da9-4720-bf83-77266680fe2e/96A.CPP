#include <bits/stdc++.h>
using namespace std;

int main() {
  string str;
  cin >> str;

  regex zeros(".*0000000.*");
  regex ones(".*1111111.*");
  if (regex_match(str, zeros) || regex_match(str, ones)) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
