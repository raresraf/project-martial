#include <algorithm>
#include <iostream>
#include <string>
using namespace std;
int main() {
  string s1, s2;
  cin >> s1 >> s2;
  int n(s1.length()), i(0);
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
  for (i = 0; i < n; i++) {
    if ((int)s1[i] > (int)s2[i]) {
      cout << 1 << endl;
      return 0;
    } else if ((int)s1[i] < (int)s2[i]) {
      cout << -1 << endl;
      return 0;
    } else
      continue;
  }
  cout << 0 << endl;
  return 0;
}
