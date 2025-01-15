#include <bits/stdc++.h>
#include <string.h>
using namespace std;
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  string str1, str2;

  getline(cin, str1);
  getline(cin, str2);
  transform(str1.begin(), str1.end(), str1.begin(), ::tolower);
  transform(str2.begin(), str2.end(), str2.begin(), ::tolower);
  if (str1.compare(str2) > 0) {
    cout << "1";
  } else if (str1.compare(str2) == 0) {
    cout << "0";
  } else if (str1.compare(str2) < 0) {
    cout << "-1";
  }

  return 0;
}
